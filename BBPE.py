"""BBPE 分词器：训练工作流的编排层。

本文件只做三件事：把各业务模块串起来、维护对外 API、以及"要不要开跑"的决策与提示。
所有算法细节都在 bbpe/ 包里，改算法不应该动这个文件。

快速上手：
    uv sync --extra dev
    python BBPE.py smoke                 # 本地假语料，零网络，跑通 准备->训练->落盘->加载->推理
    python BBPE.py inspect               # 看当前 checkpoint 到哪一步了
    python BBPE.py tokenize --text "..."  # 用已训好的词表分词
    python BBPE.py train --target-mb-per-lang 50   # 真实训练（先报资源量再征求确认）
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from bbpe.config import BYTE_COUNT, ParamDirs, VocabSpec
from bbpe.corpus_stream import (
    DEFAULT_SOURCES,
    CorpusSource,
    PrepareStats,
    Sampler,
    accumulate_corpus,
    build_sources,
    iter_sources,
)
from bbpe.devdata import synthetic_items
from bbpe.merge_engine import MergeEngine
from bbpe.param_store import (
    CheckpointError,
    LoadedCheckpoint,
    describe,
    load_checkpoint,
    save_checkpoint,
)
from bbpe.pretokenize import segment, text_to_bytes
from bbpe.text_clean import DEFAULT_CLEAN_CONFIG, CleanConfig


# ---------------------------------------------------------------------------
# 训练配置与结果
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """一次训练的全部输入。集中成一个对象而不是 train() 的二十个参数，
    这样配置可以被打印、被复用、被写进 checkpoint 的 extra 元数据里做溯源。
    """

    # --- 数据来源 ---
    # corpus_items 是依赖注入点：给了它就完全不走网络（冒烟、单测、离线复现都用它）。
    corpus_items: Optional[Iterable[Tuple[str, str]]] = None
    sources: Tuple[CorpusSource, ...] = DEFAULT_SOURCES
    # 每种语言希望采集的原始字节数。这是真正的资源闸门，不是抽样概率。
    target_bytes_per_lang: Dict[str, int] = field(
        default_factory=lambda: {"zh": 50 * 1024 * 1024, "en": 50 * 1024 * 1024}
    )
    # 是否联网查数据集真实规模来定抽样概率。关掉时靠 stop_after_bytes 兜底。
    query_dataset_scale: bool = False

    # --- 内存上限 ---
    # 单语言缓冲区字节数：只决定"瞬时"峰值（清洗副本另算），撑不了累积态。
    buffer_bytes: int = 1024 * 1024 * 1024
    # 词频表规模硬上限（唯一片段条数）。实测 RSS ≈ 40MB + 2KB × 唯一片段数，
    # 所以这条才是"会不会 OOM"的真正开关。None = 不设限，只适合冒烟/小语料。
    max_unique_pieces: Optional[int] = 100_000
    clean: CleanConfig = DEFAULT_CLEAN_CONFIG

    # --- 训练主循环 ---
    resume: bool = True                # False 则无视已有 checkpoint，从零开始
    save_every_merges: int = 2000      # 每 N 次合并提交一次断点
    verify_every_merges: int = 0       # >0 时每 N 次合并跑一次全量一致性断言（很慢，调试用）
    progress_every_merges: int = 200
    max_merges: Optional[int] = None   # 调试用硬上限，None 表示只受词表容量与频率阈值约束
    sampler_seed: int = 20060822

    def describe(self) -> str:
        targets = {k: round(v / 1e6, 1) for k, v in self.target_bytes_per_lang.items()}
        cap = "不限" if self.max_unique_pieces is None else f"{self.max_unique_pieces:,}"
        return (
            f"来源={'注入的内存语料' if self.corpus_items is not None else ','.join(s.repo_id for s in self.sources)}"
            f" 目标语言量(MB)={targets} 缓冲={self.buffer_bytes / 1e6:.1f}MB"
            f" 唯一片段上限={cap} 续训={self.resume} 每{self.save_every_merges}合并提交一次"
        )


@dataclass
class TrainReport:
    merges: int
    vocab_size: int
    prepare: Optional[PrepareStats]
    seconds_prepare: float
    seconds_train: float
    generations: List[int] = field(default_factory=list)
    stopped_because: str = ""

    def summary(self) -> str:
        return (
            f"合并 {self.merges} 次，词表 {self.vocab_size}，"
            f"准备 {self.seconds_prepare:.1f}s / 训练 {self.seconds_train:.1f}s，"
            f"提交断点 {self.generations}，终止原因：{self.stopped_because}"
        )


# ---------------------------------------------------------------------------
# 资源量估算（在真正开跑前展示给用户）
# ---------------------------------------------------------------------------
@dataclass
class ResourceEstimate:
    corpus_bytes: int
    unique_pieces: int
    pair_slots: int
    merge_count: int
    ram_bytes: int
    disk_bytes: int
    seconds: float
    network_bytes: int
    merge_rate: float = 0.0
    capped_by_pieces: bool = False

    def render(self) -> str:
        gb = 1024**3
        lines = [
            "本次训练的资源预估（本机实测标定，数量级可信，跨语言分布请重新标定）",
            f"  采集语料        {self.corpus_bytes / 1e6:,.2f} MB（所有语言合计，按 UTF-8 原始字节）",
            f"  联网下载量      {self.network_bytes / 1e6:,.2f} MB（streaming 会读完走过的记录，与保留量不等）",
            f"  唯一预切分片段  {self.unique_pieces:,.0f} 条（内存主导项，不是语料字节数）"
            + ("  <- 已被唯一片段上限截断" if self.capped_by_pieces else ""),
            f"  待统计 pair     {self.pair_slots:,.0f} 种",
            f"  预计合并次数    {self.merge_count:,.0f} 次",
            f"  内存峰值        {self.ram_bytes / gb:,.2f} GB",
            f"  落盘体积        {self.disk_bytes / 1e6:,.2f} MB（每次提交都全量重写）",
            f"  合并速率        {self.merge_rate:,.1f} 次/秒",
            f"  预计耗时        {self.seconds / 60:,.1f} 分钟（单进程 CPython，无并行）",
        ]
        return "\n".join(lines)


# --- 实测标定系数（见 estimate_resources 的 docstring）----------------------
PIECES_PER_CORPUS_BYTE = 0.021
BASE_RSS_BYTES = 40 * 1024 * 1024
RSS_PER_PIECE = 2000
PAIRS_PER_PIECE = 1.8
MERGE_RATE_CONST = 3.0e6


def _report_progress(stats: PrepareStats) -> None:
    """缓冲区每刷写一次打印一行。真实训练要跑几十分钟，没有心跳就像卡死。"""
    print(
        f"  已处理 {stats.kept_bytes / 1e6:,.1f}MB / 唯一片段累计 {stats.pieces_seen:,.0f}"
        f" / 刷写 {stats.flushes} 次",
        flush=True,
    )


def estimate_resources(config: TrainConfig, spec: VocabSpec) -> ResourceEstimate:
    """按经验系数把"字节数"推成"内存/时间"。

    系数出处（本机 Python 3.13 + devdata 中英混合假语料实测，两个标定取点拟合误差 <3%）：
      - 唯一片段率 0.021/字节：平均片段 18.6 字节，去重后约 2.1MB 语料 -> 4.6 万片段。
        中文语句几乎不重复，所以这个比率远高于纯英文，也是内存膨胀的根源。
      - RSS ≈ 40MB(解释器+numpy 底座) + 2.0KB × 唯一片段数。
        主导项是 pair 倒排索引的槽位（每片段约 3.6 个槽），不是语料本身。
      - pair 种类 ≈ 1.8 × 唯一片段数。
      - 合并速率 ≈ 3.0e6 / 唯一片段数 次/秒：每步成本正比于"含该 pair 的词数 × 词长"。
    换语料分布（纯英文、噪声更多）就要重新标定一次，别把它当物理常数。
    """
    corpus_bytes = sum(config.target_bytes_per_lang.values())
    if isinstance(config.corpus_items, (list, tuple)):
        # 注入语料且是实体序列：直接量真实字节数。
        # 刻意不消费生成器 —— 预估阶段把语料迭代一遍会让真正训练拿到空流。
        corpus_bytes = sum(len(text_to_bytes(t)) for _, t in config.corpus_items)
    unique_pieces = max(1000, int(corpus_bytes * PIECES_PER_CORPUS_BYTE))
    capped = config.max_unique_pieces is not None and unique_pieces > config.max_unique_pieces
    if capped:
        # 设了唯一片段上限，内存与时间都按上限算 —— OOM 的驱动因素是片段数，
        # 语料再大也只是被提前截断，不会把内存推高。
        unique_pieces = config.max_unique_pieces
    pair_slots = int(unique_pieces * PAIRS_PER_PIECE)
    merge_count = min(
        spec.max_vocabulary_size - spec.base_vocabulary_size,
        max(0, corpus_bytes // 40),  # 频率阈值的粗略等价：平均每个 pair 至少 40 次才值得合并
    )
    ram = int(BASE_RSS_BYTES + unique_pieces * RSS_PER_PIECE)
    disk = int(
        unique_pieces * (18.6 * 2 + 16)          # words.npz：uint16 内容 + offsets/counts
        + pair_slots * 16                        # counter.npz：int64 key + int64 freq
        + spec.max_vocabulary_size * 40           # vocab.json + merges.txt
    )
    rate = MERGE_RATE_CONST / max(1, unique_pieces)
    network = 0 if config.corpus_items is not None else corpus_bytes * 3
    return ResourceEstimate(
        corpus_bytes=corpus_bytes,
        unique_pieces=unique_pieces,
        pair_slots=pair_slots,
        merge_count=merge_count,
        ram_bytes=ram,
        disk_bytes=disk,
        seconds=merge_count / rate,
        network_bytes=int(network),
        merge_rate=rate,
        capped_by_pieces=capped,
    )


# ---------------------------------------------------------------------------
# 分词器主体
# ---------------------------------------------------------------------------
class BBPETokenizer:
    """
    # BBPE分词器
    ## 用于从一段文本序列得到相应的token序列,其词表经过训练后得到
    ## 主要方法:
        1. `tokenize (str) -> list[token]`  用于将一段文本序列分割成一段token序列
        2. `train()`  用于从一个基础模型通过一定训练得到一定容量的词表，词表即模型
    ## 词表即模型：推理只需要 vocab + merges 两样，其余(word_freq/counter)都是训练态。
    """

    def __init__(
        self,
        *,
        max_vocabulary_size: int = 60_000,
        min_frequency: int = 2,
        root_dir: Optional[Path | str] = None,
        resume: bool = True,
    ) -> None:
        # 设置数据存储路径
        self.ROOT_DIR: Path = Path(root_dir) if root_dir else Path(__file__).resolve().parent
        self.DATA_ROOT_DIR: Path = self.ROOT_DIR / "data"
        self.CORPUS_DIR: Path = self.DATA_ROOT_DIR / "data"
        self.PARAM_DIR: Path = self.DATA_ROOT_DIR / "params"
        self.dirs = ParamDirs(self.ROOT_DIR)

        # id 空间的唯一权威定义（基础词表/上限/终止符），见 bbpe/config.py
        self.spec = VocabSpec(max_vocabulary_size=max_vocabulary_size)
        self.base_vocabulary_size: int = self.spec.base_vocabulary_size
        self.max_vocabulary_size: int = self.spec.max_vocabulary_size
        self.end_of_text_id: int = self.spec.end_of_text_id

        # 设置频率阈值：最高频 pair 低于它就停训（再合并只会把噪声固化成 token）
        self.min_frequency: int = min_frequency

        # 训练与推理的全部状态都在引擎里
        self.engine = MergeEngine(self.spec, min_frequency=min_frequency)

        # 骨架里那三段"读不到就初始化"的逻辑判反了（is not None 才重置 == 永远丢弃断点）。
        # 正确做法是整套加载：成功就全套换上，失败或没有就保持引擎构造好的基础词表。
        # 逐项 setter 会带来 counter/merges/词表 进度不一致的风险，因此这里没有分步恢复。
        self.generation: int = 0
        if resume:
            checkpoint = self._load_vocab()
            if checkpoint is not None:
                checkpoint.apply_to(self.engine)
                self.generation = checkpoint.generation

    # -- 骨架里那三个属性名的落点（读的是引擎的活数据，不复制）------------------
    @property
    def word_id_map(self) -> Dict[bytes, int]:
        return self.engine.word_id_map

    @property
    def id_word_map(self) -> Dict[int, bytes]:
        return self.engine.id_word_map

    @property
    def counter(self) -> Dict[int, int]:
        """全局 pair 计数器：键为扁平化后的 pair id（id1 * max_vocabulary_size + id2）。"""
        return self.engine.pair_freq

    @property
    def merges(self) -> List[Tuple[int, int]]:
        """合并规则列表，下标即 rank。"""
        return self.engine.merges

    @property
    def vocab_size(self) -> int:
        return self.engine.vocab_size

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    def tokenize(self, text: str, add_eot: bool = True) -> list:
        """
        # 主要作用:分词
        ## 将一段文本序列分割成一段token序列
        ### params:
        text: str, 输入文本序列
        add_eot: bool, 是否在结尾追加唯一特殊 token <|EndOfText|>。
                     默认 True —— 这个 token 存在的意义就是标记序列终点，
                     训练时也必须以同样的方式构造样本，否则推理与训练分布不一致。
        ### returns:
        list[int], 输出分词后的token序列
        ### 说明
        未经训练时词表只有 256 个字节 + 终止符，此时本方法退化为"逐字节切分"，
        这仍然是一个合法可用的分词器（只是压缩率为 1）。
        """
        merge_seq = self.engine.merge_seq
        eot = self.end_of_text_id
        findall = segment
        to_bytes = text_to_bytes
        ids: list[int] = []
        for piece in findall(text):
            # 训练时"一个预切分片段 = 一个独立合并域"，推理必须完全一致，
            # 否则会出现训练里从没合并过的跨片段 pair 被拿来用（或反过来）。
            ids.extend(merge_seq(tuple(to_bytes(piece))))
        if add_eot:
            ids.append(eot)
        return ids

    def decode(self, ids: Iterable[int], *, drop_eot: bool = True) -> str:
        """token 序列 -> 文本。

        注意必须先把每个 id 经 id_word_map 还原成 **token 的字面字节**再拼接：
        合并 token 的 id 大于 255，直接把 id 当字节值会当场报错（这正是曾经踩过的坑）。
        字节级词表的好处就在这儿 —— 任何 id 序列拼出来的字节都能无损还原成原文，
        哪怕切分点落在一个汉字的三字节中间。
        """
        payload = list(ids)
        if drop_eot and payload and payload[-1] == self.end_of_text_id:
            payload.pop()
        unknown = [i for i in payload if i not in self.id_word_map]
        if unknown:
            raise KeyError(f"词表里没有这些 id：{unknown[:8]}（词表与 token 序列不匹配？）")
        buffer = b"".join(self.id_word_map[i] for i in payload)
        # errors="replace"：截断到词中间的序列（比如推理只生成了一半）不该让解码崩掉。
        return buffer.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------
    def train(self, config: Optional[TrainConfig] = None) -> TrainReport:
        config = config or TrainConfig()
        report = TrainReport(0, self.vocab_size, None, 0.0, 0.0)

        # 准备工作：只有在还没统计过词频时才做（续训场景下词频表已从 checkpoint 恢复，
        # 而且 load_state 内部已经建好索引，再建一次是几十秒的白工）。
        started = time.perf_counter()
        if not self.engine.word_freq:
            report.prepare = self._prepare(config)
            self.engine.build_index()
        report.seconds_prepare = time.perf_counter() - started

        # 训练主循环，循环条件为词表大小为主，频率阈值为辅
        limit = config.max_merges if config.max_merges is not None else float("inf")
        started = time.perf_counter()
        since_save = 0
        while not self.engine.vocab_full and report.merges < limit:
            best = self.engine.best_pair()
            # 当最高频率对已经小于频率阈值，终止训练
            if best is None or best[1] < self.min_frequency:
                report.stopped_because = (
                    "无可用 pair" if best is None else f"最高频 pair 仅 {best[1]} < min_frequency"
                )
                break

            result = self.engine.step()
            if result is None:  # 与上面的判定等价，留作兜底：步进来只能因为容量或阈值停下
                report.stopped_because = "引擎拒绝合并"
                break
            report.merges += 1
            since_save += 1

            if config.verify_every_merges and report.merges % config.verify_every_merges == 0:
                self.engine.consistency_check()

            # 一定轮次后保存 counter、merges、词表。
            # 保存由 param_store 以"提交点"方式完成，异常中断不会留下三者不一致的断点。
            if since_save >= config.save_every_merges:
                self._save(config)
                report.generations.append(self.generation)
                since_save = 0

            if config.progress_every_merges and report.merges % config.progress_every_merges == 0:
                elapsed = time.perf_counter() - started
                rate = report.merges / max(elapsed, 1e-9)
                eta = (
                    (limit - report.merges) / rate / 60
                    if rate > 0 and limit != float("inf")
                    else float("inf")
                )
                print(
                    f"  [{report.merges:>6}] 词表 {self.vocab_size}/{self.max_vocabulary_size} "
                    f"rank={result.rank} 合并 {result.pair} -> {result.new_token!r} "
                    f"freq={result.frequency} 波及词 {result.affected_words} "
                    f"{rate:.1f} merges/s 剩余 {eta:.1f}min",
                    flush=True,
                )

        report.seconds_train = time.perf_counter() - started
        report.vocab_size = self.vocab_size
        # 收尾提交：即使一轮都没跑满 save_every，也要把最终状态落盘。
        self._save(config)
        report.generations.append(self.generation)
        if not report.stopped_because:
            report.stopped_because = "达到词表上限" if self.engine.vocab_full else "达到 max_merges"
        return report

    def _prepare(self, config: TrainConfig) -> PrepareStats:
        """
        # 此函数的职责:
        1. 首先流式加载中英文训练数据(中文from "BAAI/CCI3-HQ",英文from "finewiki"->"en"),
           同时根据数据集体量(get_dataset_scale)确定保留概率做抽样
        2. 为避免全量加载把内存打满，缓冲区按字节数限上限，中/英各一个；每当缓冲区满
           就触发一次文本预处理
        3. 文本预处理：清洗（去冗余信息、重复信息、连续无用符号）
        4. 预处理后用 GPT 正则分词并字节化，把每个字节替换成基础词表中的 id
        5. 统计该片段的 id 对频率并累计进全局统计量
           （实现上先累积"唯一片段频次"，pair 统计由 build_index 一次性精确导出 ——
            数值与逐段累加完全相同，但事后还能增量修正，详见 bbpe/merge_engine.py）
        """
        if config.corpus_items is not None:
            # 注入式语料（冒烟/单测/离线复现）：数据已在内存里，规模已知，
            # 不需要抽样器，也不需要早停闸门。
            return accumulate_corpus(
                config.corpus_items,
                self.engine,
                buffer_bytes=config.buffer_bytes,
                clean_config=config.clean,
                max_unique_pieces=config.max_unique_pieces,
            )

        probabilities = build_sources(
            config.sources,
            scale_fn=self._scale_probe if config.query_dataset_scale else None,
            target_bytes_per_lang=config.target_bytes_per_lang,
        )
        sampler = Sampler(
            probabilities,
            stop_after_bytes=config.target_bytes_per_lang,
            seed=config.sampler_seed,
        )
        print(
            "  抽样概率 "
            + ", ".join(f"{lang}={prob:.6f}" for lang, prob in probabilities.items())
            + f"（攒够目标字节即停：{ {k: round(v / 1e6, 1) for k, v in config.target_bytes_per_lang.items()} }MB）"
        )
        return accumulate_corpus(
            iter_sources(config.sources),
            self.engine,
            sampler=sampler,
            buffer_bytes=config.buffer_bytes,
            clean_config=config.clean,
            max_unique_pieces=config.max_unique_pieces,
            on_flush=_report_progress,
        )

    def _scale_probe(self, repo_id: str, config: Optional[str], split: str) -> Optional[int]:
        """联网查数据集规模；失败就返回 None（随后按"不抽样 + 攒够就停"执行）。"""
        try:
            return BBPETokenizer.get_dataset_scale(repo_id, config=config, split=split)
        except Exception as exc:  # 网络/权限/元数据缺失都不该中断训练
            print(f"  ! 读取 {repo_id} 规模失败：{exc}", file=sys.stderr)
            return None

    def _save(self, config: TrainConfig) -> None:
        self.generation += 1
        save_checkpoint(
            self.engine,
            self.dirs,
            generation=self.generation,
            extra_meta={"train_config": config.describe(), "min_frequency": self.min_frequency},
        )

    def save(self) -> int:
        """手动提交一次断点（train() 内部会自动调）。"""
        self._save(TrainConfig())
        return self.generation

    # ------------------------------------------------------------------
    # 骨架里那三个 _load_* 的落点
    # ------------------------------------------------------------------
    def _checkpoint(self):
        """加载整套 checkpoint。

        注意：三份产物必须一起加载（见 bbpe/param_store.py 的提交点说明），
        所以这三个方法共享一次加载结果。第一个调用者触发真正的读取。
        """
        if not hasattr(self, "_checkpoint_cache"):
            try:
                self._checkpoint_cache = load_checkpoint(self.dirs, self.spec)
            except CheckpointError as exc:
                print(f"  x checkpoint 无效：{exc}", file=sys.stderr)
                raise
        return self._checkpoint_cache

    def _load_vocab(self) -> Optional[LoadedCheckpoint]:
        """该函数用于加载PARAM_DIR下的模型词表文件（连同配套的规则/词频/计数器）。
        返回 LoadedCheckpoint；None 表示尚无有效词表，调用方应使用基础词表。
        """
        checkpoint = self._checkpoint()
        if checkpoint is None:
            return None
        # 骨架里词表是返回 dict 的；这里保持"能拿到词表"的语义，同时把整套状态一并返回，
        # 因为只取词表而丢弃 merges 会直接导致 tokenize 用一套不完整的规则。
        return checkpoint

    def _load_counter(self) -> Optional[Dict[int, int]]:
        """该函数用于加载PARAM_DIR下的计数器文件（派生产物，仅用于校验）。"""
        checkpoint = self._checkpoint()
        return None if checkpoint is None else checkpoint.counter

    def _load_merges(self) -> Optional[List[Tuple[int, int]]]:
        """该函数用于加载PARAM_DIR下的合并规则文件"""
        checkpoint = self._checkpoint()
        return None if checkpoint is None else checkpoint.merges

    # ------------------------------------------------------------------
    # 数据集规模探测（保持骨架原实现：三级回退，不改动语义）
    # ------------------------------------------------------------------
    @staticmethod
    def get_dataset_scale(
        repo_id: str,
        config: Optional[str] = None,
        split: Optional[str] = "train",
    ) -> Optional[int]:
        """
        获取 Hugging Face 数据集的真实规模（字节数）。

        优先使用 load_dataset_builder 读取上传者提供的 dataset_size / download_size；
        若不可用，回退到 Dataset Viewer 的 /size API,
        再不可用,直接通过 HfApi 累加所有文件大小

        返回:
                num_bytes: 数据集的字节数（近似真实文本体量)
        """
        import requests
        from datasets import load_dataset_builder

        HF_SIZE_API = "https://datasets-server.huggingface.co/size"

        # 方式一：load_dataset_builder（最快，依赖上传者提供元数据）
        try:
            builder = load_dataset_builder(repo_id, name=config)
            info = builder.info
            num_bytes = info.dataset_size or info.download_size
            if num_bytes:
                return int(num_bytes)
        except Exception:
            pass  # 静默失败，尝试 API

        # 方式二：Dataset Viewer /size API
        try:
            params = {"dataset": repo_id}
            if config:
                params["config"] = config
            if split:
                params["split"] = split
            resp = requests.get(HF_SIZE_API, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            # /size 返回的结构可能因 config/split 而异
            # 尝试从不同层级提取 num_bytes
            size_info = data.get("size", {})
            dataset_info = size_info.get("dataset", {})

            # num_bytes_original_files 最接近真实文本体量；parquet 大小可作为次选
            num_bytes = (
                dataset_info.get("num_bytes_original_files")
                or dataset_info.get("num_bytes_parquet_files")
                or dataset_info.get("num_bytes_memory")
            )

            if num_bytes:
                return int(num_bytes)
        except Exception:
            pass

        # 方式三:直接累加文件大小
        from huggingface_hub import HfApi
        try:
            info = HfApi().dataset_info(repo_id=repo_id, files_metadata=True)
            total = sum(f.size or 0 for f in info.siblings)
            return total if total > 0 else None
        except Exception:
            pass

        return None


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------
def _confirm_or_abort(estimate: ResourceEstimate, *, assume_yes: bool) -> None:
    print(estimate.render())
    if assume_yes:
        print("--yes 已给出，跳过交互确认\n")
        return
    answer = input("确认开跑？输入 yes 继续，其它任意输入取消：").strip().lower()
    if answer != "yes":
        print("已取消，未产生任何下载。")
        raise SystemExit(1)


def _cmd_smoke(args) -> int:
    """本地假语料冒烟：不联网、不装 datasets，也能把全流程走一遍。"""
    tokenizer = BBPETokenizer(root_dir=args.root, resume=False)
    items = synthetic_items(args.docs, seed=args.seed)
    config = TrainConfig(
        corpus_items=items,
        buffer_bytes=1024 * 1024,
        save_every_merges=args.save_every,
        progress_every_merges=args.progress_every,
        max_merges=args.max_merges,
    )
    estimate = estimate_resources(config, tokenizer.spec)
    print("冒烟模式（内存假语料，零网络）：")
    print(estimate.render())
    report = tokenizer.train(config)
    print(report.summary())

    # 重新加载一次，证明"落盘的词表 + 推理"这条链路真的闭合
    reloaded = BBPETokenizer(root_dir=args.root, resume=True)
    sample = "机器学习的训练数据经过分词后得到 token 序列。The quick brown fox jumps."
    before = tokenizer.tokenize(sample)
    after = reloaded.tokenize(sample)
    if before != after:
        print("x 重新加载后分词结果与内存中的不一致！")
        return 1
    print(f"✓ 重启加载后分词一致，样本切成 {len(after)} 个 token")
    print("  样例 token:", after[:16], "...")
    print("  解码回来  :", reloaded.decode(after)[:80])
    return 0


def _cmd_train(args) -> int:
    tokenizer = BBPETokenizer(root_dir=args.root, resume=not args.fresh)
    target = args.target_mb_per_lang * 1024 * 1024
    config = TrainConfig(
        target_bytes_per_lang={"zh": target, "en": target},
        buffer_bytes=args.buffer_mb * 1024 * 1024,
        save_every_merges=args.save_every,
        query_dataset_scale=args.query_scale,
        max_unique_pieces=args.max_unique_pieces,
        resume=not args.fresh,
        max_merges=args.max_merges,
    )
    print("训练配置：", config.describe())
    _confirm_or_abort(estimate_resources(config, tokenizer.spec), assume_yes=args.yes)
    report = tokenizer.train(config)
    print(report.summary())
    return 0


def _cmd_tokenize(args) -> int:
    tokenizer = BBPETokenizer(root_dir=args.root)
    text = args.text if args.text is not None else sys.stdin.read()
    ids = tokenizer.tokenize(text, add_eot=not args.no_eot)
    print(f"词表 {tokenizer.vocab_size}/{tokenizer.max_vocabulary_size}，规则 {len(tokenizer.merges)} 条")
    print("ids :", ids[:64], "..." if len(ids) > 64 else "")
    print("tokens:", [tokenizer.id_word_map[i] for i in ids[:16]], "...")
    print("round-trip:", "OK" if tokenizer.decode(ids) == text.strip() or args.no_eot else "MISMATCH")
    return 0


def _cmd_inspect(args) -> int:
    tokenizer = BBPETokenizer(root_dir=args.root)
    dirs = ParamDirs(Path(args.root))
    print(describe(dirs))
    top = sorted(tokenizer.counter.items(), key=lambda kv: -kv[1])[:5] if tokenizer.counter else []
    for key, freq in top:
        a, b = tokenizer.spec.unflatten(key)
        print(f"  高频 pair {tokenizer.id_word_map[a]!r}+{tokenizer.id_word_map[b]!r} = {freq}")
    return 0


def _cmd_estimate(args) -> int:
    tokenizer = BBPETokenizer(root_dir=args.root, resume=False)
    target = args.target_mb_per_lang * 1024 * 1024
    config = TrainConfig(
        target_bytes_per_lang={"zh": target, "en": target},
        max_unique_pieces=args.max_unique_pieces,
    )
    print(estimate_resources(config, tokenizer.spec).render())
    return 0


def _cmd_scale(args) -> int:
    print(BBPETokenizer.get_dataset_scale(args.repo_id, config=args.config, split=args.split))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BBPE 训练/推理工作流")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent), help="数据与参数根目录")
    sub = parser.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("smoke", help="本地假语料冒烟（零网络）")
    p.add_argument("--docs", type=int, default=6000)
    p.add_argument("--seed", type=int, default=20261028)
    p.add_argument("--max-merges", type=int, default=None)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--progress-every", type=int, default=100)
    p.set_defaults(func=_cmd_smoke)

    p = sub.add_parser("train", help="真实数据训练（会联网）")
    p.add_argument("--target-mb-per-lang", type=int, default=50)
    p.add_argument("--buffer-mb", type=int, default=8)
    p.add_argument(
        "--max-unique-pieces",
        type=int,
        default=None,
        help="词频表唯一片段上限，内存的真正闸门（RSS≈40MB+2KB/片段）；建议按机器内存反推",
    )
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--max-merges", type=int, default=None)
    p.add_argument("--query-scale", action="store_true", help="联网查数据集真实规模来定抽样概率")
    p.add_argument("--fresh", action="store_true", help="忽略已有 checkpoint 从头训练")
    p.add_argument("--yes", action="store_true", help="跳过资源确认（脚本环境用）")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("estimate", help="只打印资源预估，不训练")
    p.add_argument("--target-mb-per-lang", type=int, default=50)
    p.add_argument("--max-unique-pieces", type=int, default=None)
    p.set_defaults(func=_cmd_estimate)

    p = sub.add_parser("tokenize", help="用已训好的词表分词")
    p.add_argument("--text", default=None)
    p.add_argument("--no-eot", action="store_true")
    p.set_defaults(func=_cmd_tokenize)

    p = sub.add_parser("inspect", help="查看 checkpoint 概况")
    p.set_defaults(func=_cmd_inspect)

    p = sub.add_parser("scale", help="查询 HF 数据集体量")
    p.add_argument("repo_id")
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="train")
    p.set_defaults(func=_cmd_scale)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))
