"""流式语料管道的测试：全部用注入的数据源，一次网络都不发。

`corpus_stream` 是"内存上限"和"抽样语义"的唯一体现场，所以这里的断言集中在
两件事：缓冲区是否真的按字节封顶，以及抽样/早停是否按预算生效。
"""

from __future__ import annotations

import pytest

from bbpe.config import VocabSpec
from bbpe.corpus_stream import (
    ByteBudgetBuffer,
    CorpusSource,
    Sampler,
    accumulate_corpus,
    build_sources,
    hf_text_iterator,
    iter_sources,
    keep_probability,
    process_batch,
)
from bbpe.merge_engine import MergeEngine
from bbpe.pretokenize import segment, text_to_bytes
from bbpe.text_clean import CleanConfig


class TestByteBudgetBuffer:
    def test_flushes_at_capacity_using_utf8_bytes(self):
        buf = ByteBudgetBuffer(capacity_bytes=10)
        assert buf.push("你好") is None            # 6 字节
        assert buf.used_bytes == 6
        batch = buf.push("世界")                    # 再 6 字节 -> 12 >= 10
        assert batch is not None and len(batch) == 2
        assert buf.used_bytes == 0

    def test_capacity_is_byte_not_char(self):
        """中文一字三字节：按字符计会把内存上限算成实际值的三分之一。"""
        buf = ByteBudgetBuffer(capacity_bytes=20)
        # 7 个汉字只有 7 个字符，却是 21 字节 >= 20，必须触发刷写；
        # 如果缓冲区按字符计数，这里就会安静地放过，内存上限形同虚设。
        assert buf.push("字" * 7) is not None
        assert buf.used_bytes == 0

    def test_drain_returns_remainder_and_clears(self):
        buf = ByteBudgetBuffer(capacity_bytes=1000)
        buf.push("还没有满")
        assert buf.drain() == ["还没有满"]
        assert buf.drain() == []

    def test_len_tracks_pending_records(self):
        buf = ByteBudgetBuffer(capacity_bytes=1000)
        buf.push("a")
        buf.push("b")
        assert len(buf) == 2

    def test_rejects_non_positive_capacity(self):
        with pytest.raises(ValueError):
            ByteBudgetBuffer(capacity_bytes=0)

    def test_batch_never_exceeds_capacity_plus_one_record(self):
        """单条超长记录会独自成批，这是可接受的：clean_text 那边已经有长度闸门。"""
        buf = ByteBudgetBuffer(capacity_bytes=32)
        peak = 0
        for i in range(200):
            batch = buf.push(f"第 {i} 条记录，内容长度可控。")
            if batch is not None:
                peak = max(peak, sum(len(text_to_bytes(x)) for x in batch))
        assert peak <= 32 + 60


class TestKeepProbability:
    def test_ratio_of_target_to_dataset(self):
        assert keep_probability(1000, 100) == pytest.approx(0.1)

    def test_clamped_to_one_when_target_larger(self):
        assert keep_probability(100, 1000) == 1.0

    @pytest.mark.parametrize("scale", [None, 0, -5])
    def test_unknown_scale_means_no_sampling(self, scale):
        """规模未知时绝不赌小概率：那样可能把语料抽到只剩千分之一而毫不知情。"""
        assert keep_probability(scale, 100) == 1.0


class TestSampler:
    def test_deterministic_with_same_seed(self):
        def run(seed):
            s = Sampler({"zh": 0.5}, seed=seed)
            return [s.keep("zh", 10) for _ in range(200)]

        assert run(1) == run(1)
        assert run(1) != run(2)

    def test_probability_is_roughly_honored(self):
        s = Sampler({"en": 0.2}, seed=99)
        kept = sum(1 for _ in range(20000) if s.keep("en", 1))
        assert 3200 < kept < 4800          # 期望 4000，留足波动避免抖

    def test_prob_one_keeps_everything_without_rng(self):
        s = Sampler({"en": 1.0}, seed=7)
        assert all(s.keep("en", 5) for _ in range(100))

    def test_stop_after_budget(self):
        s = Sampler({"zh": 1.0}, stop_after_bytes={"zh": 100}, seed=7)
        for _ in range(5):
            s.keep("zh", 30)
        assert s.exhausted("zh")
        assert not s.keep("zh", 30)

    def test_budget_is_per_language(self):
        s = Sampler({"zh": 1.0, "en": 1.0}, stop_after_bytes={"zh": 50}, seed=7)
        while not s.exhausted("zh"):
            s.keep("zh", 20)
        assert not s.exhausted("en")
        assert s.keep("en", 20)

    def test_collected_snapshot_is_a_copy(self):
        s = Sampler({"zh": 1.0}, seed=7)
        s.keep("zh", 42)
        snapshot = s.collected
        snapshot["zh"] = 0
        assert s.collected["zh"] == 42


class TestProcessBatch:
    def test_words_are_byte_ids_of_segments(self, spec):
        engine = MergeEngine(spec)
        from bbpe.corpus_stream import PrepareStats

        stats = PrepareStats()
        process_batch(["ab cd"], engine, stats, clean_config=CleanConfig())
        assert set(engine.word_freq) == {(ord("a"), ord("b")), (ord(" "), ord("c"), ord("d"))}
        assert stats.pieces_seen == 2

    def test_cleaning_happens_inside_the_pipeline(self, spec):
        """符号刷屏在清洗阶段就被塌缩，不会带着噪声进统计。"""
        engine = MergeEngine(spec)
        from bbpe.corpus_stream import PrepareStats

        process_batch(["!!!" + "!" * 40], engine, PrepareStats(), clean_config=CleanConfig())
        assert all(len(w) <= 3 for w in engine.word_freq)

    def test_empty_and_blank_records_are_skipped(self, spec):
        engine = MergeEngine(spec)
        from bbpe.corpus_stream import PrepareStats

        stats = PrepareStats()
        process_batch(["", "   ", "\n"], engine, stats, clean_config=CleanConfig())
        assert engine.word_freq == {}


class TestAccumulateCorpus:
    def test_word_freq_matches_manual_count(self, spec):
        engine = MergeEngine(spec)
        items = [("zh", "机器学习"), ("en", "the low low"), ("zh", "机器学习")]
        accumulate_corpus(items, engine, buffer_bytes=10_000_000)
        # 中文整句是一个片段，出现两次；" low" 出现两次
        assert engine.word_freq[tuple("机器学习".encode())] == 2
        assert engine.word_freq[tuple(b" low")] == 2

    def test_stats_are_reported(self, spec):
        engine = MergeEngine(spec)
        items = [("zh", "中文"), ("en", "hello world")]
        stats = accumulate_corpus(items, engine, buffer_bytes=10_000_000)
        assert stats.raw_records == 2
        assert stats.kept_records == 2
        assert stats.kept_bytes == sum(len(text_to_bytes(t)) for _, t in items)
        assert stats.pieces_seen == len(engine.word_freq)
        assert stats.summary().count("MB") >= 1

    def test_tiny_buffer_forces_multiple_flushes(self, spec):
        engine = MergeEngine(spec)
        items = [("zh", "数据" * 20) for _ in range(50)]
        stats = accumulate_corpus(items, engine, buffer_bytes=64)
        assert stats.flushes > 10
        assert sum(engine.word_freq.values()) == 50  # 每条都是同一个片段，计数应等于条数

    def test_per_language_buffers_are_independent(self, spec):
        engine = MergeEngine(spec)
        items = [("zh", "中" * 10), ("en", "a" * 10), ("zh", "中" * 10), ("en", "b" * 10)]
        stats = accumulate_corpus(items, engine, buffer_bytes=31)
        # 两条 zh 各 10 个汉字 = 60 字节，两条 en 各 10 字节；分缓冲区互不掺和
        assert stats.per_lang_bytes == {"zh": 60, "en": 20}

    def test_tail_leftover_is_not_dropped(self, spec):
        """最后不满一缓冲区的数据必须也被统计，否则"抽样 100MB 实际只用 92MB"。"""
        engine = MergeEngine(spec)
        items = [("zh", "只有一条很短的记录")]
        stats = accumulate_corpus(items, engine, buffer_bytes=10_000)
        assert stats.flushes == 1
        assert engine.word_freq

    def test_on_flush_callback_fires_each_flush(self, spec):
        engine = MergeEngine(spec)
        seen: list[int] = []
        accumulate_corpus(
            [("zh", "字" * 100) for _ in range(10)],
            engine,
            buffer_bytes=200,
            on_flush=lambda stats: seen.append(stats.flushes),
        )
        assert seen and seen == sorted(seen)

    def test_sampler_gates_throughput(self, spec):
        engine = MergeEngine(spec)
        items = [("zh", "同一条很长的中文记录" * 10) for _ in range(20)]
        sampler = Sampler({"zh": 1.0}, stop_after_bytes={"zh": 500}, seed=1)
        stats = accumulate_corpus(items, engine, sampler=sampler, buffer_bytes=10_000)
        assert stats.kept_bytes <= 500 + 300
        assert stats.raw_records == 20          # 所有记录都被"看过"，只是没被保留


class TestUniquePieceCap:
    """内存的主导项是"唯一片段数"而不是语料字节数，所以必须有这道闸门。"""

    ITEMS = [("zh", f"记录{i} 机器学习 训练 数据 算法 模型 分词 词表" ) for i in range(400)]

    def test_cap_stops_collection(self, spec):
        engine = MergeEngine(spec)
        stats = accumulate_corpus(self.ITEMS, engine, buffer_bytes=64, max_unique_pieces=120)
        assert stats.stopped_by_piece_cap
        assert len(engine.word_freq) >= 120      # 到线就停，不会远超
        assert len(engine.word_freq) < 200

    def test_no_cap_collects_everything(self, spec):
        engine = MergeEngine(spec)
        stats = accumulate_corpus(self.ITEMS, engine, buffer_bytes=64)
        assert not stats.stopped_by_piece_cap
        assert stats.kept_records == len(self.ITEMS)

    def test_tail_buffer_still_processed_after_stop(self, spec):
        """提前停止时，已经缓冲但没满的数据必须照常统计，否则"用了多少语料"是假的。"""
        engine = MergeEngine(spec)
        stats = accumulate_corpus(self.ITEMS, engine, buffer_bytes=4096, max_unique_pieces=150)
        assert stats.flushes >= 1
        assert stats.pieces_seen > 0
        assert "提前停止" in stats.summary()

    def test_blank_records_count_as_raw_but_not_kept(self, spec):
        engine = MergeEngine(spec)
        stats = accumulate_corpus([("zh", ""), ("zh", "有内容")], engine, buffer_bytes=999)
        assert stats.raw_records == 2 and stats.kept_records == 1


class TestCorpusSource:
    SOURCE = CorpusSource(name="t", repo_id="x/y", lang="zh")

    def test_text_key_preferred(self):
        assert self.SOURCE.extract_text({"text": "正文"}) == "正文"

    def test_falls_back_to_content(self):
        assert self.SOURCE.extract_text({"content": "备选字段"}) == "备选字段"

    def test_explicit_field_wins(self):
        source = CorpusSource(name="t", repo_id="x/y", lang="zh", text_key="answer")
        assert source.extract_text({"text": "不是这个", "answer": "要这个"}) == "要这个"

    def test_explicit_field_missing_returns_blank(self):
        source = CorpusSource(name="t", repo_id="x/y", lang="zh", text_key="answer")
        assert source.extract_text({"text": "没有 answer 字段"}) == ""

    def test_non_dict_record(self):
        assert self.SOURCE.extract_text("直接给字符串") == "直接给字符串"
        assert self.SOURCE.extract_text(None) == ""

    def test_wrong_type_field(self):
        assert self.SOURCE.extract_text({"text": 12345}) == ""

    def test_finewiki_source_declared_as_english_subset(self):
        source = next(s for s in _default_sources() if s.lang == "en")
        assert source.config == "en"


def _default_sources():
    from bbpe.corpus_stream import DEFAULT_SOURCES

    return DEFAULT_SOURCES


class TestHfIterator:
    def test_fake_loader_is_used_instead_of_network(self, fake_records):
        calls = []

        def fake_loader(repo_id, config, split, streaming):
            calls.append((repo_id, config, split, streaming))
            return iter(fake_records)

        source = CorpusSource(name="t", repo_id="some/repo", lang="zh", config="zh_cfg")
        texts = list(hf_text_iterator(source, loader=fake_loader))
        assert calls == [("some/repo", "zh_cfg", "train", True)]
        assert [lang for lang, _ in texts] == ["zh"] * len(fake_records)
        assert texts[0][1] == "机器学习的任务是研究算法。"

    def test_iter_sources_chains_multiple_sources(self, fake_records):
        def fake_loader(repo_id, config, split, streaming):
            return [fake_records[0]] if repo_id == "a" else [fake_records[1]]

        sources = [
            CorpusSource(name="a", repo_id="a", lang="zh"),
            CorpusSource(name="b", repo_id="b", lang="en"),
        ]
        out = list(iter_sources(sources, loader=fake_loader))
        assert out == [("zh", "机器学习的任务是研究算法。"), ("en", "The main task is to study algorithms.")]

    def test_default_loader_is_not_touched_when_injected(self, fake_records, monkeypatch):
        """证明"注入 loader 就绝不会 import datasets"，这是离线可跑通的关键。"""
        import sys

        monkeypatch.setitem(sys.modules, "datasets", None)  # 一旦被 import 就炸
        source = CorpusSource(name="t", repo_id="x", lang="zh")
        list(hf_text_iterator(source, loader=lambda *a: iter(fake_records)))


class TestBuildSources:
    def test_probabilities_from_scale_fn(self):
        sources = [
            CorpusSource(name="a", repo_id="big", lang="zh"),
            CorpusSource(name="b", repo_id="small", lang="en"),
        ]
        scales = {"big": 1000, "small": 100}
        probs = build_sources(
            sources,
            scale_fn=lambda repo, cfg, split: scales[repo],
            target_bytes_per_lang={"zh": 100, "en": 100},
        )
        assert probs["zh"] == pytest.approx(0.1)
        assert probs["en"] == 1.0          # 目标比数据集体量还大 -> 全保留

    def test_none_scale_gives_one(self):
        sources = [CorpusSource(name="a", repo_id="x", lang="zh")]
        probs = build_sources(sources, scale_fn=lambda *a: None, target_bytes_per_lang={"zh": 10})
        assert probs["zh"] == 1.0

    def test_no_scale_fn_means_no_network(self):
        sources = [CorpusSource(name="a", repo_id="x", lang="zh")]
        assert build_sources(sources, scale_fn=None, target_bytes_per_lang={"zh": 10}) == {"zh": 1.0}

    def test_two_sources_same_language_take_last(self):
        """同语言多源时概率会被覆盖 —— 记下来，将来加源时要显式处理。"""
        sources = [
            CorpusSource(name="a", repo_id="x", lang="zh"),
            CorpusSource(name="b", repo_id="y", lang="zh"),
        ]
        probs = build_sources(
            sources,
            scale_fn=lambda repo, cfg, split: 1000 if repo == "x" else 2000,
            target_bytes_per_lang={"zh": 100},
        )
        assert probs["zh"] == pytest.approx(0.05)
