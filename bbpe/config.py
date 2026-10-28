"""词表 id 空间、常量与参数目录布局。

# 为什么单独一个模块
词表(counter/merges/vocab)这三份产物被 merge_engine、param_store、BBPE.py 三个
不同的业务模块读写。如果让每个模块各自"记住"文件名、flatten 基数、id 起始值，
改一个格式就要改四处 —— 这就是业务逻辑耦合。所有关于 id 空间和文件布局的知识
只在这里定义一次。

# flatten 是什么、为什么必须是 int64
pair (id1, id2) 用一个整数当 dict key：flat = id1 * V + id2，V = max_vocabulary_size。
因为所有 id < V，这个编码是双射（可逆），所以 counter 的 key 空间无冲突。
但 V=60000 时 flat 最大约 3.6e9，已经超过 int32 上限(2.1e9)，
所以落盘必须用 int64；而"单个 token 的 id"最大 60000，落在 uint16 内，
words.npz 用 uint16 存 id 序列可以让续训文件小一半。两个 dtype 的选择互相独立。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 参数文件布局
# ---------------------------------------------------------------------------
# 格式版本号。任何"旧文件用新代码读会读出错误结果"的改动都要 +1。
# 加载时版本不匹配是**直接拒绝**而不是兼容读取：词表静默错位比训练失败危险得多，
# 因为错位的词表还能"正常"推理，只是输出全是垃圾 token。
PARAM_FORMAT_VERSION = 2

VOCAB_FILE = "vocab.json"      # GPT-2 兼容：{"Ġth": 257, ...}，人可读
MERGES_FILE = "merges.txt"     # GPT-2 兼容：每行 "Ġt he"，行号即 merge rank
COUNTER_FILE = "counter.npz"   # 展平 pair -> 频率（百万级，必须二进制）
WORDS_FILE = "words.npz"       # 唯一词频表：增量合并与断点续训的真值
META_FILE = "meta.json"        # 提交点：generation + 每个产物的 sha256

# 词表 id 落盘类型：uint16 要求 id 严格 < 65536
ID_DTYPE = np.dtype("<u2")
# pair 展平 key 与频率落盘类型：int64（flat 可达 3.6e9，频率是语料级累加值）
KEY_DTYPE = np.dtype("<i8")
FREQ_DTYPE = np.dtype("<i8")

# 一个字节固定对应一个基础 id，所以基础词表里"字节区"永远是 0..255。
BYTE_COUNT = 256


@dataclass(frozen=True)
class VocabSpec:
    """词表 id 空间的唯一权威定义。

    字段含义（与 BBPE.py 骨架的属性一一对应）：
        base_vocabulary_size: 基础词表容量 = 256 个字节 + 1 个终止符 = 257。
        max_vocabulary_size : 训练允许的最大词表容量，同时也是 flatten 基数 V。
        end_of_text_id      : 唯一特殊字符 <|EndOfText|> 的 id。

    派生量：
        first_merge_id: 合并产生的新 token 从 257 开始编号（0..255 字节、256 终止符）。
    """

    base_vocabulary_size: int = 257
    max_vocabulary_size: int = 60_000
    end_of_text_id: int = 256

    def __post_init__(self) -> None:
        # 这些约束不是"防御性校验"，而是 id 空间自洽的前提。
        # 一旦违反，counter 的 key 就会碰撞，训练出来的词表是错的但不会报错，
        # 所以必须在构造期就炸掉。
        if self.base_vocabulary_size != BYTE_COUNT + 1:
            raise ValueError(
                f"字节级 BPE 的基础词表必须是 256 字节 + 1 终止符 = 257，"
                f"当前 base_vocabulary_size={self.base_vocabulary_size}"
            )
        if self.end_of_text_id != BYTE_COUNT:
            raise ValueError(
                f"end_of_text_id 必须正好排在字节区之后(=256)，当前 {self.end_of_text_id}"
            )
        if self.max_vocabulary_size <= self.base_vocabulary_size:
            raise ValueError("max_vocabulary_size 必须大于 base_vocabulary_size，否则无法合并")
        if self.max_vocabulary_size > np.iinfo(ID_DTYPE).max + 1:
            # words.npz 用 uint16 存 id；超了就得换 dtype，进而换文件格式。
            raise ValueError(
                f"max_vocabulary_size={self.max_vocabulary_size} 超过 uint16 能表示的 "
                f"{np.iinfo(ID_DTYPE).max + 1}，需要同时更换落盘 dtype"
            )

    # -- 派生量 --------------------------------------------------------------
    @property
    def first_merge_id(self) -> int:
        """第一个"合并 token"的 id。字节(0..255) 和终止符(256) 之后的下一个空位。"""
        return self.base_vocabulary_size

    @property
    def flatten_base(self) -> int:
        """flatten 基数 V。用 max_vocabulary_size 保证 id 增长到上限时编码仍单射。"""
        return self.max_vocabulary_size

    # -- pair 编解码 ---------------------------------------------------------
    def flatten(self, id1: int, id2: int) -> int:
        """(id1, id2) -> 单一整数 key。"""
        return id1 * self.flatten_base + id2

    def unflatten(self, key: int) -> tuple[int, int]:
        """flatten 的逆运算。"""
        return divmod(key, self.flatten_base)

    # -- 基础词表 ------------------------------------------------------------
    def byte_vocab(self) -> dict[bytes, int]:
        """256 个字节 -> id 的基础词表。骨架里 `i.to_bytes(1, "big")` 的显式化。"""
        return {i.to_bytes(1, "big"): i for i in range(BYTE_COUNT)}


@dataclass
class ParamDirs:
    """参数与语料的目录布局。

    CORPUS_DIR 目前只用于放"清洗后仍然要留痕的东西"（比如导出给用户肉眼检查的
    抽样片段）。按骨架注释，原始语料全程不落盘，所以这个目录在训练主流程里
    只创建不使用，属于预留的调试出口。

    不做成 frozen：__post_init__ 要把 str 归一成 Path，frozen 会挡住这一步。
    """

    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.data_dir = self.root_dir / "data"
        self.corpus_dir = self.data_dir / "data"
        self.param_dir = self.data_dir / "params"

    def ensure(self) -> "ParamDirs":
        """幂等建目录。训练启动前调用一次即可，Windows 下 mkdir(parents=True) 安全。"""
        for d in (self.data_dir, self.corpus_dir, self.param_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self
