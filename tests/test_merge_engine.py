"""合并引擎测试。

核心是一条**等价性**断言：增量算法每一步的结果，必须和"每轮全量重扫"的暴力参考实现
逐条一致。增量实现最容易出的错是静默的（词表质量下降但程序不报错），
所以这里不测"看起来对"，而是拿一个慢但显然正确的实现当尺子。
"""

from __future__ import annotations

import random

import pytest

from bbpe.config import VocabSpec
from bbpe.merge_engine import MergeEngine, merge_occurrences
from bbpe.pretokenize import segment, text_to_bytes


def word(text: str) -> tuple[int, ...]:
    return tuple(text.encode("utf-8"))


def build_engine(pairs, spec: VocabSpec, min_frequency: int = 2) -> MergeEngine:
    engine = MergeEngine(spec, min_frequency=min_frequency)
    for piece, count in pairs:
        engine.add_word(word(piece), count)
    engine.build_index()
    return engine


def reference_bpe(word_freq, spec: VocabSpec, steps: int, min_frequency: int = 2):
    """暴力参考实现：每轮从零重扫全部词，不做任何增量维护。

    选择"每轮全量重算"正是因为它简单到不可能错：没有堆、没有倒排、没有减法。
    平局规则与引擎一致 —— 先按频率降序、再按 flatten key 升序，这样两边可以逐条比对。
    """
    words = dict(word_freq)
    merges: list[tuple[int, int]] = []
    for _ in range(steps):
        stats: dict[int, int] = {}
        for seq, count in words.items():
            for i in range(len(seq) - 1):
                key = spec.flatten(seq[i], seq[i + 1])
                stats[key] = stats.get(key, 0) + count
        if not stats:
            break
        key, freq = min(stats.items(), key=lambda kv: (-kv[1], kv[0]))
        if freq < min_frequency:
            break
        left, right = spec.unflatten(key)
        new_id = spec.first_merge_id + len(merges)
        merges.append((left, right))
        nxt: dict[tuple[int, ...], int] = {}
        for seq, count in words.items():
            merged = merge_occurrences(seq, left, right, new_id)
            nxt[merged] = nxt.get(merged, 0) + count
        words = nxt
    return merges


CORPUS = [
    ("low", 5), ("lowest", 2), ("newest", 2), ("widest", 2),
    ("er", 1), ("est", 1),
]


class TestBuildIndex:
    def test_pair_counts_are_exact(self, spec):
        engine = build_engine([("ab", 3), ("abc", 2)], spec)
        # "ab" x3 贡献 (a,b) 3 次；"abc" x2 贡献 (a,b) 2 次 + (b,c) 2 次
        assert engine.counter[spec.flatten(ord("a"), ord("b"))] == 5
        assert engine.counter[spec.flatten(ord("b"), ord("c"))] == 2
        assert len(engine.counter) == 2

    def test_repeat_occurrences_are_counted_with_multiplicity(self, spec):
        """(x,x,x) 含两个 (x,x)，必须都算进去 —— 后续增量减法用同一套重数。"""
        engine = build_engine([("aaa", 4)], spec)
        assert engine.counter[spec.flatten(ord("a"), ord("a"))] == 8

    def test_counter_is_empty_before_build(self, spec):
        engine = MergeEngine(spec)
        engine.add_word(word("ab"), 10)
        assert engine.counter == {}

    def test_add_word_merges_duplicate_pieces(self, spec):
        engine = MergeEngine(spec)
        engine.add_word(word("hi"), 2)
        engine.add_word(word("hi"), 5)
        assert engine.word_freq[word("hi")] == 7

    def test_non_positive_count_ignored(self, spec):
        engine = MergeEngine(spec)
        engine.add_word(word("hi"), 0)
        engine.add_word(word("hi"), -3)
        assert engine.word_freq == {}

    def test_vocab_starts_with_bytes_and_eot(self, spec):
        engine = MergeEngine(spec)
        assert engine.vocab_size == 257
        assert engine.id_word_map[256] == b"<|EndOfText|>"
        assert engine.word_id_map[b"<|EndOfText|>"] == 256
        assert engine.next_id == 257


class TestMergeStep:
    def test_first_merge_is_the_most_frequent_pair(self, spec):
        engine = build_engine(CORPUS, spec)
        result = engine.step()
        assert result is not None
        left, right = result.pair
        assert engine.id_word_map[left] + engine.id_word_map[right] == result.new_token
        # "low low low low low lowest" 里 (o,w) 或 (l,o) 最高频，取决于 bpe 经典例子；
        # 这里只要求它确实是当前最大频率
        assert result.frequency == max(engine.counter.values() or [result.frequency])

    def test_new_id_equals_first_merge_id_plus_rank(self, spec):
        engine = build_engine(CORPUS, spec, min_frequency=1)
        for expected_rank in range(5):
            result = engine.step()
            if result is None:
                pytest.skip("语料太小，跑不到 5 步")
            assert result.new_id == spec.first_merge_id + expected_rank
            assert result.rank == expected_rank
            assert engine.id_word_map[result.new_id] == result.new_token

    def test_merged_pair_disappears_from_counter(self, spec):
        engine = build_engine(CORPUS, spec)
        result = engine.step()
        assert spec.flatten(*result.pair) not in engine.counter

    def test_counter_matches_full_recompute_after_every_step(self, spec):
        """核心不变式：增量 counter == 全量重算，每一步都必须成立。"""
        engine = build_engine(CORPUS + [("中文分词", 3), ("机器学习模型", 2)], spec, min_frequency=1)
        for _ in range(40):
            if engine.step() is None:
                break
            assert engine.counter == engine.recompute_pair_freq()
            engine.consistency_check()

    def test_inverted_index_matches_counter(self, spec):
        """pair_words 与 counter 必须互相印证：桶里的词数不能多也不能少。"""
        engine = build_engine(CORPUS, spec, min_frequency=1)
        for _ in range(10):
            if engine.step() is None:
                break
            derived: dict[int, int] = {}
            for key, bucket in engine.pair_words.items():
                total = sum(engine.word_freq[w] for w in bucket)
                assert total > 0, "空桶没被回收，说明删除路径漏了"
                derived[key] = total
            assert derived == engine.counter

    def test_overlapping_merge_is_left_to_right_non_overlapping(self, spec):
        """(x,x,x) 合并 (x,x) 只能得到 (c,x)，与推理端 merge_seq 必须同构。"""
        engine = build_engine([("aaa", 5)], spec)
        result = engine.step()
        assert result is not None
        merged_words = list(engine.word_freq)
        assert any(len(w) == 2 for w in merged_words)
        assert merge_occurrences(word("aaaaa"), ord("a"), ord("a"), 999) == (999, 999, ord("a"))

    def test_word_total_frequency_is_conserved(self, spec):
        """合并绝不产生或消灭频次：词频表总和必须恒定。"""
        engine = build_engine(CORPUS, spec, min_frequency=1)
        before = sum(engine.word_freq.values())
        for _ in range(15):
            engine.step()
        assert sum(engine.word_freq.values()) == before

    def test_stops_at_min_frequency(self, spec):
        engine = build_engine([("ab", 1), ("cd", 1)], spec, min_frequency=2)
        assert engine.step() is None

    def test_stops_when_vocab_full(self, tiny_spec):
        """用小词表上限触发容量终止路径（生产里 60000 很难跑满，必须单独测）。

        语料必须"够杂"：合并次数上界约等于 Σ(词长-1)，所以用 60 个互不相同、
        长度 24 的词，保证能撑满 300-257=43 次合并。
        """
        rng = random.Random(1234)
        pairs = [
            ("".join(rng.choice("abcdefghijklmnopqrstuvwxyz012345") for _ in range(24)), 3)
            for _ in range(60)
        ]
        engine = build_engine(pairs, tiny_spec, min_frequency=1)
        count = 0
        while engine.step() is not None:
            count += 1
        assert engine.vocab_full
        assert engine.vocab_size == tiny_spec.max_vocabulary_size
        assert count == tiny_spec.max_vocabulary_size - tiny_spec.base_vocabulary_size
        # 词表满之后必须坚决停手，不能再分配新 id（否则 uint16 落盘会溢出）
        assert engine.step() is None
        assert engine.next_id == tiny_spec.max_vocabulary_size

    def test_heap_does_not_grow_without_bound(self, spec):
        """惰性删除会让堆积累陈旧项；必须有重建闸门，否则长跑就是纯内存泄漏。"""
        engine = build_engine([("abcdefgh" * 6, 30), ("zyxwvuts" * 6, 30)], spec, min_frequency=1)
        for _ in range(60):
            if engine.step() is None:
                break
        assert len(engine._heap) <= 4 * len(engine.counter) + 8


class TestEquivalenceWithReference:
    @pytest.mark.parametrize(
        "pairs,steps",
        [
            (CORPUS, 12),
            ([("aabcd", 7), ("abacd", 5), ("aacd", 3), ("dbca", 2)], 15),
            ([(piece, 3) for piece in ["你好世界", "世界你好", "机器学习", "习器机学"]], 20),
            ([("ababab" * 2, 4), ("bababa" * 2, 5), ("abbaba", 1)], 18),
        ],
    )
    def test_merge_sequence_identical(self, spec, pairs, steps):
        engine = build_engine(pairs, spec, min_frequency=1)
        for _ in range(steps):
            engine.step()
        word_freq = {word(piece): count for piece, count in pairs}
        assert engine.merges == reference_bpe(word_freq, spec, steps, min_frequency=1)

    def test_merge_sequence_identical_on_generated_corpus(self, spec):
        """在随机生成的语料上比对，覆盖手工用例想不到的形状。"""
        from bbpe.devdata import synthetic_items

        from bbpe.text_clean import clean_text

        freq: dict[tuple[int, ...], int] = {}
        for _, text in synthetic_items(300, seed=7):
            for piece in segment(clean_text(text)):
                w = tuple(text_to_bytes(piece))
                freq[w] = freq.get(w, 0) + 1
        engine = MergeEngine(spec, min_frequency=1)
        for w, c in freq.items():
            engine.add_word(w, c)
        engine.build_index()
        for _ in range(60):
            engine.step()
        assert engine.merges == reference_bpe(freq, spec, 60, min_frequency=1)


class TestMergeSeqForInference:
    def test_trained_words_are_attractors(self, spec):
        """训练结束后，词频表里的每个词再走一遍 merge_seq 必须原样返回。

        这是"训练与推理一致"最硬的证据：训练得到的最终形态就是推理会得到的形态。
        """
        engine = build_engine(CORPUS + [("lower", 3), ("newest", 4)], spec, min_frequency=2)
        for _ in range(12):
            engine.step()
        for seq in engine.word_freq:
            assert engine.merge_seq(seq) == seq

    def test_merge_seq_lossless_and_all_tokens_known(self, spec):
        engine = build_engine(CORPUS, spec, min_frequency=1)
        for _ in range(10):
            engine.step()
        text = "the lowest new widest low"
        produced: list[int] = []
        for piece in segment(text):
            merged = engine.merge_seq(tuple(text_to_bytes(piece)))
            produced.extend(merged)
            # 拼回去必须等于原片段的字节。注意不能直接 bytes(merged)：
            # 合并后的 id 大于 255，必须经 id_word_map 还原成 token 字面再拼接。
            assert b"".join(engine.id_word_map[i] for i in merged) == text_to_bytes(piece)
        assert all(i in engine.id_word_map for i in produced)

    def test_merge_seq_with_no_rules_returns_bytes(self, spec):
        engine = MergeEngine(spec)
        assert engine.merge_seq(word("hi")) == word("hi")

    def test_merge_seq_is_idempotent(self, spec):
        engine = build_engine(CORPUS, spec, min_frequency=1)
        for _ in range(8):
            engine.step()
        for piece in ["lowest", "newest", "widest"]:
            once = engine.merge_seq(word(piece))
            assert engine.merge_seq(once) == once


class TestLoadState:
    def test_load_state_rebuilds_counter_and_passes_check(self, spec):
        engine = build_engine(CORPUS, spec, min_frequency=1)
        for _ in range(6):
            engine.step()
        restored = MergeEngine(spec, min_frequency=1)
        restored.load_state(
            id_word_map=engine.id_word_map,
            merges=engine.merges,
            word_freq=engine.word_freq,
        )
        assert restored.counter == engine.counter
        assert restored.merges == engine.merges
        assert restored.next_id == engine.next_id
        assert restored.vocab_size == engine.vocab_size
        restored.consistency_check()

    def test_load_state_rejects_inconsistent_counter(self, spec):
        """counter 与词频表不是同一进度时必须拒绝，而不是"先加载了再说"。"""
        engine = build_engine(CORPUS, spec, min_frequency=1)
        for _ in range(4):
            engine.step()
        bad_counter = dict(engine.counter)
        key = next(iter(bad_counter))
        bad_counter[key] += 12345
        restored = MergeEngine(spec, min_frequency=1)
        with pytest.raises(AssertionError):
            restored.load_state(
                id_word_map=engine.id_word_map,
                merges=engine.merges,
                word_freq=engine.word_freq,
                counter_to_verify=bad_counter,
            )

    def test_consistency_check_detects_tampering(self, spec):
        engine = build_engine(CORPUS, spec, min_frequency=1)
        for _ in range(5):
            engine.step()
        engine.consistency_check()
        # 人为制造"counter 少了一项"，模拟增量逻辑出错或被截断的状态
        key = next(iter(engine.counter))
        del engine.counter[key]
        with pytest.raises(AssertionError):
            engine.consistency_check()


class TestSpecValidation:
    def test_base_vocab_must_be_257(self):
        with pytest.raises(ValueError):
            VocabSpec(base_vocabulary_size=256)

    def test_eot_must_be_256(self):
        with pytest.raises(ValueError):
            VocabSpec(end_of_text_id=300)

    def test_max_vocab_must_exceed_base(self):
        with pytest.raises(ValueError):
            VocabSpec(max_vocabulary_size=100)

    def test_max_vocab_must_fit_uint16(self):
        """超过 65535 会让 words.npz 的 uint16 溢出，必须在这里挡住而不是写出坏文件。"""
        with pytest.raises(ValueError):
            VocabSpec(max_vocabulary_size=70_000)

    def test_flatten_is_bijective(self, spec):
        for a in (0, 32, 255, 256, 257, 59_999):
            for b in (0, 65, 255, 59_999):
                assert spec.unflatten(spec.flatten(a, b)) == (a, b)
