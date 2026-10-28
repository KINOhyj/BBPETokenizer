"""测试共享装置：全部离线，绝不联网、绝不碰仓库里的 data/。"""

from __future__ import annotations

import pytest

from bbpe.config import ParamDirs, VocabSpec
from bbpe.corpus_stream import CorpusSource
from bbpe.merge_engine import MergeEngine
from bbpe.pretokenize import segment, text_to_bytes


@pytest.fixture
def spec() -> VocabSpec:
    """默认词表规格（与生产一致：256 字节 + EoT，上限 60000）。"""
    return VocabSpec()


@pytest.fixture
def tiny_spec() -> VocabSpec:
    """小上限的词表规格，用来专门触发"词表满"这条终止路径。"""
    return VocabSpec(max_vocabulary_size=300)


@pytest.fixture
def dirs(tmp_path) -> ParamDirs:
    """把参数目录指向 pytest 的临时目录，保证测试之间互不污染。"""
    return ParamDirs(tmp_path)


@pytest.fixture
def engine(spec) -> MergeEngine:
    return MergeEngine(spec, min_frequency=2)


def word_freq_of(text: str) -> dict[tuple[int, ...], int]:
    """把一段文本变成"唯一片段 -> 频次"表，等价于 _prepare 的最终产物。

    测试里到处需要这一步，所以提到 conftest：它把"数据准备"和"被测逻辑"分开，
    改预切分实现时只用改一处。
    """
    freq: dict[tuple[int, ...], int] = {}
    for piece in segment(text):
        word = tuple(text_to_bytes(piece))
        freq[word] = freq.get(word, 0) + 1
    return freq


@pytest.fixture
def fake_records():
    """假的 HF 流式记录：形状与 datasets 返回的 dict 一致，用于注入 loader。"""
    return [
        {"text": "机器学习的任务是研究算法。"},
        {"text": "The main task is to study algorithms."},
        {"content": "用 content 字段的数据源也要能取到正文。"},
        {"text": ""},
        {"other": "没有正文字段的记录应当被跳过而不是报错。"},
    ]
