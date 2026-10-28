"""语料流式管道：抽样 -> 字节上限缓冲 -> 清洗 -> 预切分 -> 字节化 -> 累计词频。

# 这一层唯一的目标：把"无限大的网络数据"变成"有限内存里的词频表"
CCI3-HQ / finewiki 都是 TB 级数据集，任何"先下载再处理"的方案都会先把机器打满。
所以这里全程 streaming，内存里只留两样东西：
    - 一个字节上限的缓冲区（瞬时，可预算）
    - 一张去重后的词频表（累积，上界是"唯一片段数"而不是"语料长度"）

# 为什么要"按语言各开一个缓冲区"
中英文的字节密度差 3 倍（汉字 utf-8 占 3 字节），清洗规则的有效长度也差很多。
混在一个缓冲区里，"缓冲区满"这个事件对两种语言的含义不同，
抽样比例也容易被先到的语言吃掉。分开之后每个语言的吞吐是独立可控的。

# 抽样概率怎么定（以及一个容易被忽略的坑）
保留概率 keep_prob = 目标字节数 / 数据集真实字节数（由 get_dataset_scale 提供）。
但**按概率抽样仍然要走完整个数据集**：streaming 是一条条读、一条条掷骰子，
即使只保留 1/1000，网络侧的读取量一点没少。所以这里额外加一个"攒够目标字节就停"
的硬闸 stop_after_bytes —— 真正决定这次训练消耗多少资源的其实是它，
不是概率。做资源评估时千万别只算 keep_prob。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator

from .config import VocabSpec
from .merge_engine import MergeEngine
from .pretokenize import segment, text_to_bytes
from .text_clean import CleanConfig, clean_text

# ---------------------------------------------------------------------------
# 数据源定义
# ---------------------------------------------------------------------------
# 已知数据集的记录字段不统一（finewiki 有 title/summary/text，CCI3 一般是 text），
# 所以按顺序探测这几个 key，取第一个非空字符串。
_TEXT_KEY_CANDIDATES = ("text", "content", "raw_content", "question", "answer")


@dataclass(frozen=True)
class CorpusSource:
    """一个 HF 数据集（或配置子集）作为一个语料来源。"""

    name: str
    repo_id: str
    lang: str                      # 仅用于分缓冲区与进度显示，不参与算法
    config: str | None = None      # HF 的 subset/name，finewiki 的语言码就写在这里
    split: str = "train"
    text_key: str | None = None    # 显式指定字段；None 时按候选顺序探测

    def extract_text(self, record) -> str:
        """从一条记录里取出正文。

        取不到就返回空串而不是抛异常：流式管道里字段缺失是常态（不同子集 schema 不同），
        中断几小时的任务去处理一条脏记录是不可接受的。
        """
        if not isinstance(record, dict):
            return record if isinstance(record, str) else ""
        if self.text_key:
            value = record.get(self.text_key)
            return value if isinstance(value, str) else ""
        for key in _TEXT_KEY_CANDIDATES:
            value = record.get(key)
            if isinstance(value, str) and value:
                return value
        return ""


DEFAULT_SOURCES: tuple[CorpusSource, ...] = (
    CorpusSource(name="cci3_hq_zh", repo_id="BAAI/CCI3-HQ", lang="zh"),
    CorpusSource(name="finewiki_en", repo_id="HuggingFaceFW/finewiki", lang="en", config="en"),
)


# ---------------------------------------------------------------------------
# 抽样
# ---------------------------------------------------------------------------
def keep_probability(dataset_bytes: int | None, target_bytes: int) -> float:
    """按"目标字节 / 数据集真实字节"给出保留概率。

    拿不到数据集规模（离线、dataset card 没填、viewer 未处理）时返回 1.0，
    即**不抽样**：宁可少读几条就早停，也不要在未知规模下赌一个小概率，
    那会因为概率算错而把语料抽得只剩千分之一却毫不知情。
    """
    if not dataset_bytes or dataset_bytes <= 0:
        return 1.0
    return max(0.0, min(1.0, target_bytes / dataset_bytes))


class Sampler:
    """按语言预算做 Bernoulli 抽样，并负责"攒够就停"。

    为什么不是"随机取 N 条"：数据集的条数与字节数不成正比（一条 wiki 记录 100 字，
    一条论坛记录 5000 字）。BPE 需要的是**字节量**达标，所以预算与停止条件都按字节算。
    """

    def __init__(
        self,
        probabilities: dict[str, float],
        stop_after_bytes: dict[str, int] | None = None,
        *,
        seed: int = 20261028,
    ) -> None:
        self._rng = random.Random(seed)
        self.probabilities = probabilities
        self.stop_after_bytes = stop_after_bytes or {}
        self._collected: dict[str, int] = {}

    def keep(self, lang: str, nbytes: int) -> bool:
        """判定一条记录是否保留；同时负责早停闸门的开合。"""
        limit = self.stop_after_bytes.get(lang, 0)
        if limit and self._collected.get(lang, 0) >= limit:
            return False
        prob = self.probabilities.get(lang, 1.0)
        # 概率为 1 时不必掷骰子，省掉热路径上的一次随机数生成。
        if prob < 1.0 and self._rng.random() >= prob:
            return False
        self._collected[lang] = self._collected.get(lang, 0) + nbytes
        return True

    def exhausted(self, lang: str) -> bool:
        limit = self.stop_after_bytes.get(lang, 0)
        return bool(limit) and self._collected.get(lang, 0) >= limit

    @property
    def collected(self) -> dict[str, int]:
        return dict(self._collected)


# ---------------------------------------------------------------------------
# 字节上限缓冲区
# ---------------------------------------------------------------------------
class ByteBudgetBuffer:
    """按 utf-8 字节数限定的文本缓冲区。

    为什么精确计字节而不是按条数或字符数：内存是按字节占用的，而中英混合语料里
    字符数与字节数差 3 倍。用字符数当上限会让中文缓冲区的实际内存是英文的 1/3，
    "缓冲区上限"这个内存保证就变成了一个近似值。
    """

    __slots__ = ("capacity_bytes", "_items", "_used")

    def __init__(self, capacity_bytes: int) -> None:
        if capacity_bytes <= 0:
            raise ValueError("缓冲区容量必须为正数")
        self.capacity_bytes = capacity_bytes
        self._items: list[str] = []
        self._used = 0

    def __len__(self) -> int:
        return len(self._items)

    @property
    def used_bytes(self) -> int:
        return self._used

    def push(self, text: str) -> list[str] | None:
        """写入一条文本；缓冲区满时返回本轮要处理的批次并清空自己，否则返回 None。

        返回 list 而不是拼接后的大 string：拼接会让同一份文本在内存里同时存在两份，
        而清洗本来就是逐条做的，保持 list 可以避免这次拷贝。
        """
        self._items.append(text)
        self._used += len(text_to_bytes(text))
        if self._used >= self.capacity_bytes:
            flushed, self._items, self._used = self._items, [], 0
            return flushed
        return None

    def drain(self) -> list[str]:
        """取走剩余内容（数据流结束时必须调用，否则最后一批会被丢掉）。"""
        flushed, self._items, self._used = self._items, [], 0
        return flushed


# ---------------------------------------------------------------------------
# 累计词频（骨架 _prepare 的第 3/4/5 步）
# ---------------------------------------------------------------------------
@dataclass
class PrepareStats:
    """一次 _prepare 的量化产出，用于判断"这次语料到底喂进去多少"。"""

    raw_records: int = 0
    kept_records: int = 0
    kept_bytes: int = 0
    cleaned_bytes: int = 0
    pieces_seen: int = 0
    flushes: int = 0
    # 是否因为触到 max_unique_pieces 而提前停止采集。必须显式报告：
    # 否则"抽样 50MB"实际只用了 12MB，而用户以为用满了，词表质量归因会全错。
    stopped_by_piece_cap: bool = False
    per_lang_bytes: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        tail = "，[触发唯一片段上限而提前停止]" if self.stopped_by_piece_cap else ""
        return (
            f"记录 {self.raw_records} -> 保留 {self.kept_records} 条/"
            f"{self.kept_bytes / 1e6:.2f}MB，清洗后 {self.cleaned_bytes / 1e6:.2f}MB，"
            f"预切分片段 {self.pieces_seen}，缓冲刷写 {self.flushes} 次，"
            f"分语言 { {k: round(v / 1e6, 2) for k, v in self.per_lang_bytes.items()} }MB{tail}"
        )


def process_batch(
    batch: list[str],
    engine: MergeEngine,
    stats: PrepareStats,
    *,
    clean_config: CleanConfig,
) -> None:
    """缓冲区的处理流水线：清洗 -> 预切分 -> 字节化 -> 累加词频。

    单独成函数是为了让"缓冲区满之后做什么"这件事可以被单独测试，
    不必真的构造一个装满的缓冲区。
    """
    add_word = engine.add_word
    for text in batch:
        cleaned = clean_text(text, clean_config)
        if not cleaned:
            continue
        stats.cleaned_bytes += len(text_to_bytes(cleaned))
        for piece in segment(cleaned):
            # 字节化：每个字节即一个基础 id（0..255），见 pretokenize 模块说明。
            add_word(tuple(text_to_bytes(piece)), 1)
            stats.pieces_seen += 1


def accumulate_corpus(
    items: Iterable[tuple[str, str]],
    engine: MergeEngine,
    *,
    sampler: Sampler | None = None,
    buffer_bytes: int = 8 * 1024 * 1024,
    clean_config: CleanConfig | None = None,
    max_unique_pieces: int | None = None,
    on_flush: Callable[[PrepareStats], None] | None = None,
) -> PrepareStats:
    """主循环：消费 (lang, text) 流，按语言分别缓冲，满了就跑流水线。

    参数:
        items        : (语言标记, 原始文本) 可迭代对象。数据来源被这一层彻底抽象掉，
                       所以测试可以喂一个普通 list，不必联网。
        engine       : 词频写进它的 word_freq（合并统计由 build_index 统一导出）。
        sampler      : 抽样与早停；None 表示全量保留。
        buffer_bytes : 每个语言缓冲区的字节上限，决定瞬时内存峰值。
        max_unique_pieces : 词频表规模硬上限。这是**内存的真正闸门** —— 实测
                       RSS ≈ 40MB + 2KB × 唯一片段数，也就是说撑爆内存的不是语料字节数，
                       而是"去重后还剩多少唯一片段"。中文语句几乎不重复，所以语料翻几倍、
                       唯一片段就跟着翻几倍；只靠"缓冲区上限"完全挡不住累积态的膨胀。
                       None 表示不设限（小语料/冒烟用）。
        on_flush     : 每次缓冲刷写后的回调，用于打印进度。

    返回:
        PrepareStats。达到 max_unique_pieces 时提前结束，已缓冲的内容照常处理完再返回，
        不会丢数据 —— 丢数据会让"抽样了 50MB"这句话变成假的。
    """
    clean_config = clean_config or CleanConfig()
    stats = PrepareStats()
    buffers: dict[str, ByteBudgetBuffer] = {}

    for lang, text in items:
        if max_unique_pieces is not None and len(engine.word_freq) >= max_unique_pieces:
            stats.stopped_by_piece_cap = True
            break
        stats.raw_records += 1
        if not text:
            continue
        nbytes = len(text_to_bytes(text))
        if sampler is not None:
            if sampler.exhausted(lang):
                continue
            if not sampler.keep(lang, nbytes):
                continue
        stats.kept_records += 1
        stats.kept_bytes += nbytes
        stats.per_lang_bytes[lang] = stats.per_lang_bytes.get(lang, 0) + nbytes

        buffer = buffers.get(lang)
        if buffer is None:
            buffer = buffers[lang] = ByteBudgetBuffer(buffer_bytes)
        batch = buffer.push(text)
        if batch is not None:
            stats.flushes += 1
            process_batch(batch, engine, stats, clean_config=clean_config)
            if on_flush is not None:
                on_flush(stats)

    # 收尾：每个语言缓冲区里未满的部分必须处理掉，否则最后一次刷写之前的数据全丢。
    for lang, buffer in buffers.items():
        leftover = buffer.drain()
        if leftover:
            stats.flushes += 1
            process_batch(leftover, engine, stats, clean_config=clean_config)
    return stats


# ---------------------------------------------------------------------------
# HuggingFace 数据源适配（延迟导入）
# ---------------------------------------------------------------------------
def hf_text_iterator(
    source: CorpusSource,
    *,
    loader: Callable[..., Iterable] | None = None,
) -> Iterator[tuple[str, str]]:
    """把一个 CorpusSource 变成 (lang, text) 惰性流。

    `loader` 是依赖注入点：默认真实的 datasets.load_dataset(streaming=True)，
    测试传入一个返回 list[dict] 的假函数即可，整条流水线不需要网络、不需要装 datasets。
    loader 的签名固定为 (repo_id, config, split, streaming) -> 可迭代的记录流。
    """
    if loader is None:
        # 延迟导入：推理和冒烟测试路径绝不该因为没装 datasets 而无法 import 本模块。
        from datasets import load_dataset

        def loader(repo_id, config, split, streaming):
            return load_dataset(repo_id, name=config, split=split, streaming=streaming)

    stream = loader(source.repo_id, source.config, source.split, True)
    for record in stream:
        yield source.lang, source.extract_text(record)


def iter_sources(
    sources: Iterable[CorpusSource],
    *,
    loader: Callable[..., Iterable] | None = None,
) -> Iterator[tuple[str, str]]:
    """把多个 CorpusSource 首尾相接成一条 (lang, text) 流。

    刻意用"依次排空"而不是"轮询交织"：Sampler 会在某语言攒够目标字节后开始拒绝，
    依次排空已经能保证两种语言都能拿到自己的预算，而轮询要多维护 N 个迭代器状态，
    在流式网络源上还可能因为交替读取而放大连接数。
    """
    for source in sources:
        yield from hf_text_iterator(source, loader=loader)


def build_sources(
    sources: Iterable[CorpusSource],
    *,
    scale_fn: Callable[[str, str | None, str], int | None] | None,
    target_bytes_per_lang: dict[str, int],
) -> dict[str, float]:
    """按各语言目标量与数据集体量算出保留概率。

    scale_fn 同样可注入：它对应骨架里的 get_dataset_scale（要联网）。
    离线时传 None -> 概率全为 1.0，靠 stop_after_bytes 控制实际用量。
    """
    probabilities: dict[str, float] = {}
    for source in sources:
        scale = None
        if scale_fn is not None:
            scale = scale_fn(source.repo_id, source.config, source.split)
        target = target_bytes_per_lang.get(source.lang, 0)
        probabilities[source.lang] = keep_probability(scale, target)
    return probabilities
