"""BBPE 训练/推理流水线的业务模块集合。

分层原则：上层(BBPE.py)只做编排与 IO 提示，下面每个模块都只依赖 config，
且都是纯逻辑（不联网、不读文件），因此可以被单独测试。

    config        : id 空间与文件布局的唯一真值
    pretokenize   : 文本 <-> 基础字节 id（训练与推理共用同一切分逻辑）
    text_clean    : 语料清洗
    corpus_stream : 流式抽样 + 字节缓冲 + 词频累计
    merge_engine  : 精确增量 BPE 合并
    param_store   : 原子落盘与一致性校验
    devdata       : 离线假语料（冒烟测试与单测共用）
"""

from .config import ParamDirs, VocabSpec
from .corpus_stream import (
    DEFAULT_SOURCES,
    ByteBudgetBuffer,
    CorpusSource,
    PrepareStats,
    Sampler,
    accumulate_corpus,
    build_sources,
    hf_text_iterator,
    keep_probability,
)
from .merge_engine import MergeEngine, MergeResult, merge_occurrences
from .param_store import CheckpointError, LoadedCheckpoint, describe, load_checkpoint, save_checkpoint
from .pretokenize import (
    PAT,
    PAT_STR,
    PRETOKEN_VERSION,
    bytes_to_ids,
    bytes_to_unicode,
    display_to_token,
    ids_to_bytes,
    ids_to_text,
    segment,
    text_to_bytes,
    text_to_ids,
    token_to_display,
)
from .text_clean import CleanConfig, clean_text, dedup_window

__all__ = [
    "ParamDirs",
    "VocabSpec",
    "DEFAULT_SOURCES",
    "ByteBudgetBuffer",
    "CorpusSource",
    "PrepareStats",
    "Sampler",
    "accumulate_corpus",
    "build_sources",
    "hf_text_iterator",
    "keep_probability",
    "MergeEngine",
    "MergeResult",
    "merge_occurrences",
    "CheckpointError",
    "LoadedCheckpoint",
    "describe",
    "load_checkpoint",
    "save_checkpoint",
    "PAT",
    "PAT_STR",
    "PRETOKEN_VERSION",
    "bytes_to_ids",
    "bytes_to_unicode",
    "display_to_token",
    "ids_to_bytes",
    "ids_to_text",
    "segment",
    "text_to_bytes",
    "text_to_ids",
    "token_to_display",
    "CleanConfig",
    "clean_text",
    "dedup_window",
]
