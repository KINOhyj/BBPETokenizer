"""端到端测试：准备 -> 训练 -> 落盘 -> 重新加载 -> 推理，全程离线。

这一层盯的是"编排"是否正确：模块各自单测通过，不代表串起来就对。
尤其要验证断点续训（骨架最担心的进度不一致）和训练/推理一致性。
"""

from __future__ import annotations

import sys

import pytest

import BBPE
from bbpe.config import META_FILE, VOCAB_FILE
from bbpe.devdata import synthetic_items
from bbpe.param_store import CheckpointError


@pytest.fixture
def small_items():
    """一份固定 seed 的小假语料，保证任何一次运行拿到完全相同的数据。"""
    return synthetic_items(1500, seed=42)


class TestFreshTokenizer:
    def test_base_state(self, tmp_path):
        tokenizer = BBPE.BBPETokenizer(root_dir=tmp_path)
        assert tokenizer.vocab_size == 257
        assert tokenizer.base_vocabulary_size == 257
        assert tokenizer.max_vocabulary_size == 60_000
        assert tokenizer.end_of_text_id == 256
        assert tokenizer.merges == []
        assert tokenizer.counter == {}
        # 骨架里那三处判反的 None 检查修好之后，无 checkpoint 时词表必须仍是完整基础词表
        assert tokenizer.word_id_map[b"A"] == 65
        assert tokenizer.id_word_map[256] == b"<|EndOfText|>"

    def test_paths_follow_root_dir(self, tmp_path):
        tokenizer = BBPE.BBPETokenizer(root_dir=tmp_path)
        assert tokenizer.DATA_ROOT_DIR == tmp_path / "data"
        assert tokenizer.CORPUS_DIR == tmp_path / "data" / "data"
        assert tokenizer.PARAM_DIR == tmp_path / "data" / "params"

    def test_untrained_tokenize_is_byte_level(self, tmp_path):
        tokenizer = BBPE.BBPETokenizer(root_dir=tmp_path)
        ids = tokenizer.tokenize("Hi!", add_eot=False)
        assert ids == [72, 105, 33]

    def test_eot_appended_by_default(self, tmp_path):
        tokenizer = BBPE.BBPETokenizer(root_dir=tmp_path)
        assert tokenizer.tokenize("Hi")[-1] == 256

    def test_rejects_inconsistent_vocab_spec(self):
        with pytest.raises(ValueError):
            BBPE.BBPETokenizer(max_vocabulary_size=70_000)


class TestTrainingRun:
    def test_end_to_end(self, tmp_path, small_items):
        tokenizer = BBPE.BBPETokenizer(root_dir=tmp_path)
        config = BBPE.TrainConfig(
            corpus_items=small_items,
            buffer_bytes=256 * 1024,
            save_every_merges=50,
            progress_every_merges=0,
        )
        report = tokenizer.train(config)

        assert report.merges > 50
        assert report.vocab_size == tokenizer.vocab_size == 257 + report.merges
        assert report.prepare is not None and report.prepare.kept_records > 0
        assert report.stopped_because
        # 词表真的学到了语言现象，而不是随机字节对
        trained = b"".join(tokenizer.id_word_map[i] for i in range(257, tokenizer.vocab_size))
        assert b"the" in trained or b"\xe6\x9c\xba" in trained

        for name in (VOCAB_FILE, "merges.txt", "counter.npz", "words.npz", META_FILE):
            assert (tmp_path / "data" / "params" / name).exists(), name

    def test_round_trip_after_training(self, tmp_path, small_items):
        tokenizer = BBPE.BBPETokenizer(root_dir=tmp_path)
        tokenizer.train(
            BBPE.TrainConfig(
                corpus_items=small_items, buffer_bytes=256 * 1024,
                save_every_merges=50, progress_every_merges=0,
            )
        )
        text = "机器学习的训练数据经过分词后得到 token 序列。The low lowest newest."
        assert tokenizer.decode(tokenizer.tokenize(text)) == text

    def test_no_network_when_corpus_injected(self, tmp_path, small_items, monkeypatch):
        """注入语料的路径绝不能碰到 datasets —— 这是"离线可跑通"的硬保证。"""
        monkeypatch.setitem(sys.modules, "datasets", None)  # 一旦被 import 就 ImportError
        tokenizer = BBPE.BBPETokenizer(root_dir=tmp_path)
        report = tokenizer.train(
            BBPE.TrainConfig(
                corpus_items=small_items, buffer_bytes=64 * 1024,
                save_every_merges=20, progress_every_merges=0,
            )
        )
        assert report.merges > 0

    def test_deterministic(self, tmp_path, small_items):
        """同一份语料两次训练必须得到逐条相同的规则，否则实验不可复现。"""
        runs = []
        for name in ("a", "b"):
            root = tmp_path / name
            root.mkdir()
            tokenizer = BBPE.BBPETokenizer(root_dir=root)
            tokenizer.train(
                BBPE.TrainConfig(
                    corpus_items=small_items, buffer_bytes=128 * 1024,
                    save_every_merges=1000, progress_every_merges=0, max_merges=80,
                )
            )
            runs.append(tokenizer.merges)
        assert runs[0] == runs[1]

    def test_high_frequency_threshold_stops_early(self, tmp_path, small_items):
        tokenizer = BBPE.BBPETokenizer(root_dir=tmp_path, min_frequency=10_000_000)
        report = tokenizer.train(
            BBPE.TrainConfig(corpus_items=small_items, progress_every_merges=0)
        )
        assert report.merges == 0
        assert "min_frequency" in report.stopped_because


class TestResume:
    @staticmethod
    def _train_some(tmp_path, items, max_merges):
        tokenizer = BBPE.BBPETokenizer(root_dir=tmp_path, resume=False)
        report = tokenizer.train(
            BBPE.TrainConfig(
                corpus_items=items, buffer_bytes=128 * 1024,
                save_every_merges=max_merges, progress_every_merges=0, max_merges=max_merges,
            )
        )
        return tokenizer, report

    def test_checkpoint_is_reloaded_as_a_whole(self, tmp_path, small_items):
        _, first = self._train_some(tmp_path, small_items, 40)
        again = BBPE.BBPETokenizer(root_dir=tmp_path)
        assert again.vocab_size == 257 + first.merges
        assert again.generation == first.generations[-1]
        # 三份产物都到位才算"恢复"，counter 也不能空着
        assert again.merges and again.counter and again.engine.word_freq

    def test_resume_continues_without_repreparing(self, tmp_path, small_items):
        tokenizer, first = self._train_some(tmp_path, small_items, 40)
        saved_merges = list(tokenizer.merges)

        resumed = BBPE.BBPETokenizer(root_dir=tmp_path, resume=True)
        report = resumed.train(
            BBPE.TrainConfig(
                corpus_items=None, buffer_bytes=128 * 1024,
                save_every_merges=30, progress_every_merges=0, max_merges=30,
            )
        )
        assert report.prepare is None, "续训不该重新跑一遍语料准备"
        assert report.merges == 30
        assert resumed.merges[:40] == saved_merges, "已有规则必须原样保留，只在后面追加"

    def test_resume_false_starts_from_scratch(self, tmp_path, small_items):
        self._train_some(tmp_path, small_items, 40)
        fresh = BBPE.BBPETokenizer(root_dir=tmp_path, resume=False)
        assert fresh.vocab_size == 257

    def test_incremental_training_equals_one_long_run(self, tmp_path, small_items):
        """分两次各 20 步 == 一次 40 步：增量的正确性最终要由这条来背书。"""
        tokenizer, _ = self._train_some(tmp_path, small_items, 40)
        single = list(tokenizer.merges)

        step_root = tmp_path / "stepwise"
        step_root.mkdir()
        acc = BBPE.BBPETokenizer(root_dir=step_root, resume=False)
        for _ in range(2):
            acc.train(
                BBPE.TrainConfig(
                    corpus_items=small_items, buffer_bytes=128 * 1024,
                    save_every_merges=20, progress_every_merges=0, max_merges=20,
                    resume=True,
                )
            )
            acc = BBPE.BBPETokenizer(root_dir=step_root, resume=True)
        assert acc.merges == single


class TestCorruptedCheckpoint:
    def test_construction_refuses_truncated_vocab(self, tmp_path, small_items):
        tokenizer = BBPE.BBPETokenizer(root_dir=tmp_path, resume=False)
        tokenizer.train(
            BBPE.TrainConfig(
                corpus_items=small_items, buffer_bytes=128 * 1024,
                save_every_merges=20, progress_every_merges=0, max_merges=20,
            )
        )
        path = tmp_path / "data" / "params" / VOCAB_FILE
        path.write_bytes(path.read_bytes()[:-4])

        with pytest.raises(CheckpointError):
            BBPE.BBPETokenizer(root_dir=tmp_path, resume=True)

    def test_spec_change_is_rejected(self, tmp_path, small_items):
        tokenizer = BBPE.BBPETokenizer(root_dir=tmp_path, resume=False)
        tokenizer.train(
            BBPE.TrainConfig(
                corpus_items=small_items, buffer_bytes=128 * 1024,
                save_every_merges=20, progress_every_merges=0, max_merges=20,
            )
        )
        with pytest.raises(CheckpointError, match="配置"):
            BBPE.BBPETokenizer(root_dir=tmp_path, max_vocabulary_size=40_000, resume=True)


class TestSkeletonApiCompat:
    """骨架承诺过的方法名与属性必须继续可用。"""

    def test_load_helpers_return_live_state(self, tmp_path, small_items):
        tokenizer = BBPE.BBPETokenizer(root_dir=tmp_path, resume=False)
        assert tokenizer._load_vocab() is None
        assert tokenizer._load_counter() is None
        assert tokenizer._load_merges() is None

        tokenizer.train(
            BBPE.TrainConfig(
                corpus_items=small_items, buffer_bytes=128 * 1024,
                save_every_merges=20, progress_every_merges=0, max_merges=20,
            )
        )
        # 保存之后缓存仍是"无 checkpoint"的旧值，换新实例才会真正读盘
        reloaded = BBPE.BBPETokenizer(root_dir=tmp_path)
        checkpoint = reloaded._load_vocab()
        assert checkpoint is not None and checkpoint.id_word_map
        assert isinstance(reloaded._load_counter(), dict)
        assert isinstance(reloaded._load_merges(), list)
        assert len(reloaded._load_merges()) == len(reloaded.merges)

    def test_counter_key_flattening_follows_skeleton_formula(self, tmp_path, small_items):
        tokenizer = BBPE.BBPETokenizer(root_dir=tmp_path, resume=False)
        tokenizer.train(
            BBPE.TrainConfig(
                corpus_items=small_items, buffer_bytes=128 * 1024,
                save_every_merges=20, progress_every_merges=0, max_merges=20,
            )
        )
        assert tokenizer.counter
        for key in tokenizer.counter:
            a, b = tokenizer.spec.unflatten(key)
            assert a * tokenizer.max_vocabulary_size + b == key

    def test_get_dataset_scale_still_present(self):
        """骨架自带的规模探测方法不能因为重构而消失（它属于用户明确保留的 API）。"""
        assert callable(BBPE.BBPETokenizer.get_dataset_scale)
        import inspect

        signature = inspect.signature(BBPE.BBPETokenizer.get_dataset_scale)
        assert list(signature.parameters) == ["repo_id", "config", "split"]


class TestResourceEstimate:
    def test_uses_measured_bytes_for_injected_corpus(self, tmp_path, small_items):
        tokenizer = BBPE.BBPETokenizer(root_dir=tmp_path, resume=False)
        config = BBPE.TrainConfig(corpus_items=small_items)
        estimate = BBPE.estimate_resources(config, tokenizer.spec)
        expected = sum(len(BBPE.text_to_bytes(t)) for _, t in small_items)
        assert estimate.corpus_bytes == expected
        assert estimate.unique_pieces >= 1000
        assert estimate.ram_bytes > 0

    def test_does_not_consume_a_generator(self, tmp_path):
        """预估阶段如果把生成器跑完，真正训练就会拿到空语料 —— 而且是静默的。"""
        tokenizer = BBPE.BBPETokenizer(root_dir=tmp_path, resume=False)

        def generator():
            yield ("zh", "中文文本")

        config = BBPE.TrainConfig(corpus_items=generator())
        BBPE.estimate_resources(config, tokenizer.spec)
        assert next(iter(config.corpus_items)) == ("zh", "中文文本")

    def test_render_mentions_every_resource(self, tmp_path, small_items):
        tokenizer = BBPE.BBPETokenizer(root_dir=tmp_path, resume=False)
        text = BBPE.estimate_resources(BBPE.TrainConfig(corpus_items=small_items), tokenizer.spec).render()
        for token in ("采集语料", "联网下载量", "内存峰值", "落盘体积", "预计耗时"):
            assert token in text

    def test_target_bytes_drive_the_estimate(self, tmp_path):
        tokenizer = BBPE.BBPETokenizer(root_dir=tmp_path, resume=False)
        small = BBPE.estimate_resources(
            BBPE.TrainConfig(target_bytes_per_lang={"zh": 10 * 1024 * 1024}), tokenizer.spec
        )
        big = BBPE.estimate_resources(
            BBPE.TrainConfig(target_bytes_per_lang={"zh": 100 * 1024 * 1024}), tokenizer.spec
        )
        assert big.ram_bytes > small.ram_bytes and big.seconds > small.seconds

    def test_estimate_matches_measured_scaling(self, tmp_path):
        """这两个常数来自本机 RSS 实测（2.1MB->129MB / 8.4MB->374MB），
        改动 estimate 时如果偏离实测斜率就该重新标定，而不是改测试。"""
        tokenizer = BBPE.BBPETokenizer(root_dir=tmp_path, resume=False)
        estimate = BBPE.estimate_resources(
            BBPE.TrainConfig(target_bytes_per_lang={"zh": 4 * 1024 * 1024}), tokenizer.spec
        )
        assert estimate.unique_pieces == int(4 * 1024 * 1024 * BBPE.PIECES_PER_CORPUS_BYTE)
        assert abs(estimate.ram_bytes - (BBPE.BASE_RSS_BYTES + estimate.unique_pieces * BBPE.RSS_PER_PIECE)) < 2

    def test_piece_cap_bounds_ram_not_just_records(self, tmp_path):
        """设了上限之后，内存预估按上限走 —— 这正是"再大的语料也不会 OOM"的依据。"""
        tokenizer = BBPE.BBPETokenizer(root_dir=tmp_path, resume=False)
        capped = BBPE.estimate_resources(
            BBPE.TrainConfig(
                target_bytes_per_lang={"zh": 500 * 1024 * 1024},
                max_unique_pieces=200_000,
            ),
            tokenizer.spec,
        )
        assert capped.capped_by_pieces
        assert capped.unique_pieces == 200_000
        assert capped.ram_bytes < 1024**3, "20 万片段上限应该稳稳落在 1GB 以内"


class TestCli:
    def test_smoke_subcommand_runs_and_returns_zero(self, tmp_path, capsys):
        args = BBPE.build_parser().parse_args(["--root", str(tmp_path), "smoke", "--docs", "800", "--save-every", "25"])
        assert args.func(args) == 0
        out = capsys.readouterr().out
        assert "合并" in out and "重启加载后分词一致" in out

    def test_estimate_subcommand_does_not_train(self, tmp_path, capsys):
        args = BBPE.build_parser().parse_args(
            ["--root", str(tmp_path), "estimate", "--target-mb-per-lang", "10"]
        )
        assert args.func(args) == 0
        assert not (tmp_path / "data" / "params" / META_FILE).exists()
        assert "内存峰值" in capsys.readouterr().out

    def test_tokenize_subcommand(self, tmp_path, small_items, capsys):
        tokenizer = BBPE.BBPETokenizer(root_dir=tmp_path, resume=False)
        tokenizer.train(
            BBPE.TrainConfig(
                corpus_items=small_items, buffer_bytes=128 * 1024,
                save_every_merges=30, progress_every_merges=0, max_merges=30,
            )
        )
        args = BBPE.build_parser().parse_args(["--root", str(tmp_path), "tokenize", "--text", "the lowest low"])
        assert args.func(args) == 0
        assert "词表" in capsys.readouterr().out

    def test_inspect_subcommand(self, tmp_path, small_items, capsys):
        tokenizer = BBPE.BBPETokenizer(root_dir=tmp_path, resume=False)
        tokenizer.train(
            BBPE.TrainConfig(
                corpus_items=small_items, buffer_bytes=128 * 1024,
                save_every_merges=20, progress_every_merges=0, max_merges=20,
            )
        )
        args = BBPE.build_parser().parse_args(["--root", str(tmp_path), "inspect"])
        assert args.func(args) == 0
        assert "generation=" in capsys.readouterr().out
