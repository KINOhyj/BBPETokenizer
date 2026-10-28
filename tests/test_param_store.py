"""参数存取的测试。

重点验证"提交点"行为：骨架最担心的不是存不下，而是**中断之后留下三份进度不一致的
产物还能被当成有效断点加载**。所以这里系统性地注入各种"坏文件"，
要求每一种都被拒绝。
"""

from __future__ import annotations

import json

import pytest

from bbpe.config import (
    COUNTER_FILE,
    META_FILE,
    MERGES_FILE,
    PARAM_FORMAT_VERSION,
    VOCAB_FILE,
    WORDS_FILE,
    ParamDirs,
    VocabSpec,
)
from bbpe.merge_engine import MergeEngine
from bbpe.param_store import (
    CheckpointError,
    decode_merges,
    describe,
    encode_merges,
    encode_vocab,
    load_checkpoint,
    save_checkpoint,
)
from bbpe.pretokenize import PRETOKEN_VERSION, display_to_token, token_to_display


def trained_engine(spec: VocabSpec, steps: int = 8) -> MergeEngine:
    engine = MergeEngine(spec, min_frequency=1)
    for piece, count in [
        ("low", 5), ("lowest", 2), ("newest", 2), ("widest", 2),
        ("机器学习", 4), ("词表分词", 3),
    ]:
        engine.add_word(tuple(piece.encode("utf-8")), count)
    engine.build_index()
    for _ in range(steps):
        if engine.step() is None:
            break
    return engine


@pytest.fixture
def saved(dirs):
    """一个已经落盘的训练状态，返回 (engine, dirs)。"""
    engine = trained_engine(VocabSpec())
    save_checkpoint(engine, dirs, generation=1, extra_meta={"note": "单元测试"})
    return engine, dirs


class TestRoundTrip:
    def test_everything_survives(self, dirs):
        engine = trained_engine(VocabSpec(), steps=10)
        save_checkpoint(engine, dirs, generation=3)
        loaded = load_checkpoint(dirs, engine.spec)
        assert loaded is not None
        assert loaded.id_word_map == engine.id_word_map
        assert loaded.merges == engine.merges
        assert loaded.word_freq == engine.word_freq
        assert loaded.counter == engine.counter
        assert loaded.generation == 3

    def test_restored_engine_continues_identically(self, dirs):
        """续训必须和"一路训下来"得到完全相同的规则序列 —— 否则断点毫无意义。"""
        first = trained_engine(VocabSpec(), steps=6)
        save_checkpoint(first, dirs, generation=1)

        resumed = MergeEngine(VocabSpec(), min_frequency=1)
        loaded = load_checkpoint(dirs, resumed.spec)
        loaded.apply_to(resumed)
        for _ in range(4):
            resumed.step()

        direct = trained_engine(VocabSpec(), steps=10)
        assert resumed.merges == direct.merges
        assert resumed.id_word_map == direct.id_word_map
        assert resumed.counter == direct.counter

    def test_no_temp_files_left_behind(self, saved):
        engine, dirs = saved
        leftovers = list(dirs.param_dir.glob("*.tmp"))
        assert leftovers == []

    def test_meta_records_all_four_artifacts(self, saved):
        engine, dirs = saved
        meta = json.loads((dirs.param_dir / META_FILE).read_text("utf-8"))
        assert set(meta["files"]) == {VOCAB_FILE, MERGES_FILE, COUNTER_FILE, WORDS_FILE}
        assert meta["format_version"] == PARAM_FORMAT_VERSION
        assert meta["counts"]["merges"] == len(engine.merges)
        assert meta["extra"]["note"] == "单元测试"


class TestGpt2Compatibility:
    def test_vocab_json_is_plain_id_map(self, saved):
        engine, dirs = saved
        payload = json.loads((dirs.param_dir / VOCAB_FILE).read_text("utf-8"))
        assert all(isinstance(k, str) and isinstance(v, int) for k, v in payload.items())
        assert len(payload) == engine.vocab_size
        # ids 从 0 连续排到 vocab_size-1，没有空洞也没有重复
        assert sorted(payload.values()) == list(range(engine.vocab_size))

    def test_space_is_encoded_as_gpt2_leadee(self, saved):
        """词表文件里代表空格的必须是 'Ġ'，否则与 GPT-2 工具链不兼容。"""
        engine, dirs = saved
        payload = json.loads((dirs.param_dir / VOCAB_FILE).read_text("utf-8"))
        assert payload[token_to_display(b" ")] == 32
        assert " " not in payload  # 绝不能出现真空格 key

    def test_merges_txt_header_and_rank_order(self, saved):
        engine, dirs = saved
        lines = (dirs.param_dir / MERGES_FILE).read_text("utf-8").splitlines()
        assert lines[0].startswith("#version:")
        rules = lines[1:]
        assert len(rules) == len(engine.merges)
        word_id_map = {tok: i for i, tok in engine.id_word_map.items()}
        for rank, line in enumerate(rules):
            a, b = line.split(" ")
            assert (word_id_map[display_to_token(a)], word_id_map[display_to_token(b)]) == engine.merges[rank]

    def test_merges_can_be_decoded_back(self, saved):
        engine, dirs = saved
        raw = (dirs.param_dir / MERGES_FILE).read_bytes()
        word_id_map = {tok: i for i, tok in engine.id_word_map.items()}
        assert decode_merges(raw, word_id_map) == engine.merges

    def test_malformed_merge_line_rejected(self):
        with pytest.raises(CheckpointError):
            decode_merges(b"a b c", {b"a": 0, b"b": 1, b"c": 2})

    def test_merge_referencing_unknown_token_rejected(self):
        with pytest.raises(CheckpointError):
            decode_merges(b"x y", {b"a": 0, b"b": 1})

    def test_duplicate_token_literal_rejected(self, saved):
        engine, dirs = saved
        broken = dict(engine.id_word_map)
        broken[999] = broken[0]  # 人为制造两个 id 指向同一 token
        with pytest.raises(CheckpointError):
            encode_vocab(broken)


class TestCorruptionDetection:
    def test_missing_meta_means_no_checkpoint(self, dirs):
        assert load_checkpoint(dirs, VocabSpec()) is None

    def test_truncated_artifact_rejected(self, saved):
        """模拟"写到一半断电"：长度变了必须被发现。"""
        engine, dirs = saved
        path = dirs.param_dir / WORDS_FILE
        data = path.read_bytes()
        path.write_bytes(data[:-6])
        with pytest.raises(CheckpointError, match="长度"):
            load_checkpoint(dirs, engine.spec)

    def test_same_size_tampering_rejected(self, saved):
        """长度没变但内容变了：只有哈希能抓到，所以哈希校验不能省。"""
        engine, dirs = saved
        path = dirs.param_dir / COUNTER_FILE
        data = path.read_bytes()
        path.write_bytes(data[:-1] + bytes([(data[-1] + 1) % 256]))
        with pytest.raises(CheckpointError, match="sha256"):
            load_checkpoint(dirs, engine.spec)

    def test_verify_hashes_off_accepts_tampered_but_same_size_content(self, saved):
        """关掉哈希校验后，"长度对但内容错"的文件会被放行。

        这条测试是故意把风险演示出来的：长度检查（一直生效）只能抓到截断，
        抓不到原地改写。所以 verify_hashes=False 只该用于"同一进程刚存完就取"，
        跨进程/跨机器续训一律保持默认开启。
        """
        engine, dirs = saved
        path = dirs.param_dir / VOCAB_FILE
        data = bytearray(path.read_bytes())
        # 原地改一个数字字节，保证 JSON 依然合法、长度一字不差
        index = next(i for i, b in enumerate(data) if 0x30 <= b <= 0x38)
        data[index] = data[index] + 1
        path.write_bytes(bytes(data))

        with pytest.raises(CheckpointError, match="sha256"):
            load_checkpoint(dirs, engine.spec)
        loaded = load_checkpoint(dirs, engine.spec, verify_hashes=False)
        assert loaded.id_word_map != engine.id_word_map

    def test_spec_mismatch_rejected(self, saved):
        """改了 max_vocabulary_size 就等于改了 counter 的 key 空间，绝不能混用。"""
        engine, dirs = saved
        with pytest.raises(CheckpointError, match="配置"):
            load_checkpoint(dirs, VocabSpec(max_vocabulary_size=50_000))

    def test_format_version_mismatch_rejected(self, saved):
        engine, dirs = saved
        meta = json.loads((dirs.param_dir / META_FILE).read_text("utf-8"))
        meta["format_version"] = 999
        (dirs.param_dir / META_FILE).write_text(json.dumps(meta), "utf-8")
        with pytest.raises(CheckpointError, match="版本"):
            load_checkpoint(dirs, engine.spec)

    def test_pretoken_version_stamp_is_recorded(self, saved):
        """预切分版本必须进 meta：换正则等于换词表定义。"""
        engine, dirs = saved
        meta = json.loads((dirs.param_dir / META_FILE).read_text("utf-8"))
        assert meta["pretoken_version"] == PRETOKEN_VERSION

    def test_pretoken_version_mismatch_rejected(self, saved):
        engine, dirs = saved
        meta = json.loads((dirs.param_dir / META_FILE).read_text("utf-8"))
        meta["pretoken_version"] = "some_other_pattern"
        (dirs.param_dir / META_FILE).write_text(json.dumps(meta), "utf-8")
        with pytest.raises(CheckpointError, match="预切分版本"):
            load_checkpoint(dirs, engine.spec)

    def test_missing_declared_file_rejected(self, saved):
        engine, dirs = saved
        (dirs.param_dir / MERGES_FILE).unlink()
        with pytest.raises(CheckpointError, match="缺失"):
            load_checkpoint(dirs, engine.spec)

    def test_merge_count_disagreement_rejected(self, saved):
        """meta 说 8 条规则、merges.txt 里却只有 7 行 —— 典型的"两份产物不同进度"。"""
        engine, dirs = saved
        path = dirs.param_dir / MERGES_FILE
        lines = path.read_text("utf-8").splitlines()
        path.write_text("\n".join(lines[:-1]) + "\n", "utf-8")
        # 内容与长度都变了，长度检查先命中
        with pytest.raises(CheckpointError):
            load_checkpoint(dirs, engine.spec)

    def test_partial_tmp_does_not_affect_valid_checkpoint(self, saved):
        """上一次保存中途被杀留下的 .tmp 必须完全无害：正式产物仍是上一代且可读。"""
        engine, dirs = saved
        (dirs.param_dir / (WORDS_FILE + ".tmp")).write_bytes(b"\x00garbage")
        loaded = load_checkpoint(dirs, engine.spec)
        assert loaded.word_freq == engine.word_freq


class TestEncodingEdgeCases:
    def test_empty_merges_file(self, dirs):
        engine = MergeEngine(VocabSpec())
        save_checkpoint(engine, dirs, generation=0)
        lines = (dirs.param_dir / MERGES_FILE).read_text("utf-8").splitlines()
        assert len(lines) == 1  # 只有版本头
        loaded = load_checkpoint(dirs, engine.spec)
        assert loaded.merges == []
        assert loaded.word_freq == {}

    def test_words_npz_handles_single_element_and_long_words(self, dirs):
        engine = MergeEngine(VocabSpec(), min_frequency=1)
        engine.add_word((65,), 3)
        engine.add_word(tuple(b"x" * 300), 1)
        save_checkpoint(engine, dirs, generation=0)
        loaded = load_checkpoint(dirs, engine.spec)
        assert loaded.word_freq == {(65,): 3, (tuple(b"x" * 300)): 1}

    def test_counter_keys_above_int32_survive(self, dirs):
        """flatten 用 60000 做基数，大 id 的 pair 会超过 int32，int64 落盘不能截断。"""
        engine = MergeEngine(VocabSpec(), min_frequency=1)
        big_left = 59_998
        engine.add_word((big_left, big_left, 257), 2)
        engine.build_index()
        key = engine.spec.flatten(big_left, big_left)
        assert key > 2**31
        assert key in engine.counter
        save_checkpoint(engine, dirs, generation=0)
        loaded = load_checkpoint(dirs, engine.spec)
        assert loaded.counter[key] == 2
        assert loaded.word_freq[(big_left, big_left, 257)] == 2

    def test_describe_summarizes(self, saved):
        engine, dirs = saved
        text = describe(dirs)
        assert "词表=" in text and "规则=" in text

    def test_describe_without_checkpoint(self, dirs):
        assert describe(dirs) == "尚无 checkpoint"
