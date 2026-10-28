"""参数产物的原子存取与一致性校验。

# 为什么"怎么存"比"存什么"更要命
骨架注释里点名的风险是：程序异常中断导致 counter / merges / 词表 三者进度不一致。
这三个文件（外加词频表）逻辑上是一个整体，但物理上是四次独立写入。
只要有任何一次写入被打断，磁盘上就会留下"59000 条规则 + 58000 行词表"这种组合。
最危险的是它**加载得动、跑得出结果**，只是结果是错的。

对策是"提交点"式落盘，顺序固定：
    1) 每个产物先写成 `<name>.tmp`，边写边算 sha256，写完 fsync；
    2) 全部 tmp 成功之后，用 os.replace 逐个原子替换正式文件（NTFS/ext4 同分区 rename 是原子的）；
    3) **最后**才写 meta.json（含每个文件的 sha256、generation、词表配置）。
meta.json 是唯一的"提交记录"。任何一步被打断：
    - tmp 阶段中断  -> 正式文件与 meta 仍是上一代的完整组合，加载正常；
    - replace 阶段中断 -> meta 还是上一代，与新写的文件哈希不符，加载被拒绝；
    - meta 阶段中断 -> meta 缺失或 JSON 截断，加载被拒绝。
所以"读到一半的训练状态"永远不会被当成有效断点，宁可不续训也不能续错。

# counter 的地位：派生缓存，不是真值
counter.npz 按骨架要求存下来（便于离线检查和断点验证），但加载时它**只参与校验**，
counter 一律由 words.npz 的词频表重新导出。理由见 merge_engine 模块 docstring：
pair->词 的倒排索引无法从 counter 还原，counter 单独存在不足以支撑续训。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import (
    COUNTER_FILE,
    ID_DTYPE,
    KEY_DTYPE,
    META_FILE,
    MERGES_FILE,
    PARAM_FORMAT_VERSION,
    VOCAB_FILE,
    WORDS_FILE,
    FREQ_DTYPE,
    ParamDirs,
    VocabSpec,
)
from .merge_engine import MergeEngine, Word
from .pretokenize import PRETOKEN_VERSION, display_to_token, token_to_display

# GPT-2 的 merges.txt 首行是版本头，保留同样格式以免被下游工具当成第一条规则。
MERGES_HEADER = "#version: 1.0-BBPE"


class CheckpointError(RuntimeError):
    """checkpoint 不完整、被截断或配置不匹配时抛出。属于"必须人工介入"的错误。"""


@dataclass
class LoadedCheckpoint:
    """一次成功加载得到的全部状态。"""

    id_word_map: dict[int, bytes]
    merges: list[tuple[int, int]]
    word_freq: dict[Word, int]
    counter: dict[int, int]
    generation: int
    meta: dict = field(default_factory=dict)

    def apply_to(self, engine: MergeEngine) -> None:
        """整体灌入引擎。counter 传 None，让引擎自己从词频表导出并校验。"""
        engine.load_state(
            id_word_map=self.id_word_map,
            merges=self.merges,
            word_freq=self.word_freq,
            counter_to_verify=None,
        )


# ---------------------------------------------------------------------------
# 底层写入：tmp + fsync + hash
# ---------------------------------------------------------------------------
def _write_and_hash(target: Path, payload: bytes | Iterable[bytes]) -> tuple[str, int]:
    """把 payload 落到 `target.tmp`，返回 (sha256, 字节数)。

    fsync 是必须的：os.replace 只保证 rename 原子，不保证 rename 之前数据已经离开
    操作系统页缓存。断电时常见"文件存在、大小正确、内容一半是零"，哈希能抓到它。
    """
    tmp = target.with_name(target.name + ".tmp")
    hasher = hashlib.sha256()
    total = 0
    with open(tmp, "wb") as fh:
        if isinstance(payload, (bytes, bytearray)):
            chunks = [payload]
        else:
            chunks = payload
        for chunk in chunks:
            fh.write(chunk)
            hasher.update(chunk)
            total += len(chunk)
        fh.flush()
        os.fsync(fh.fileno())
    return hasher.hexdigest(), total


def _hash_file(path: Path, *, chunk: int = 1 << 20) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def _replace_atomically(tmp: Path, target: Path) -> None:
    os.replace(tmp, target)


# ---------------------------------------------------------------------------
# 各产物的序列化
# ---------------------------------------------------------------------------
def encode_vocab(id_word_map: dict[int, bytes]) -> bytes:
    """词表 -> GPT-2 兼容的 vocab.json。

    key 用字节->可打印字符的置换（空格显示成 "Ġ"），这样 JSON 仍是可读文本，
    而且能被 transformers 的 GPT2Tokenizer 直接加载。value 就是真实 id。
    """
    # id_word_map 是 id -> bytes，所以解包顺序是 (i, tok)；写反了会把 int 送进
    # token_to_display，报错信息还很难看懂，这里刻意加一层断言把方向钉死。
    printable = {token_to_display(tok): i for i, tok in id_word_map.items()}
    if len(printable) != len(id_word_map):
        # token_to_display 是双射，理论上不可能撞；真撞了说明 id_word_map 本身有重复 token。
        raise CheckpointError("词表里存在重复的 token 字面形式，无法落盘")
    # 按 id 排序输出：diff 两个 checkpoint 时能一眼看出新增在哪一段。
    ordered = dict(sorted(printable.items(), key=lambda kv: kv[1]))
    return json.dumps(ordered, ensure_ascii=False, sort_keys=False, indent=None).encode("utf-8")


def decode_vocab(raw: bytes) -> dict[int, bytes]:
    payload = json.loads(raw.decode("utf-8"))
    return {int(i): display_to_token(name) for name, i in payload.items()}


def encode_merges(id_word_map: dict[int, bytes], merges: list[tuple[int, int]]) -> bytes:
    """合并规则 -> GPT-2 兼容的 merges.txt，第 n 行的规则 rank 即 n-（表头占 1 行）。

    存的是 token 的**字面形式**而不是 id 对：字面形式自带语义（"Ġ the" 一眼就懂），
    id 又能通过 vocab.json 反查，因此不必再存一份 id 版 merges，少一个可能错位的文件。
    """
    lines = [MERGES_HEADER]
    for left, right in merges:
        try:
            a = token_to_display(id_word_map[left])
            b = token_to_display(id_word_map[right])
        except KeyError as exc:  # pragma: no cover
            raise CheckpointError(f"合并规则引用了不存在的 id：{exc}") from exc
        lines.append(f"{a} {b}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def decode_merges(raw: bytes, word_id_map: dict[bytes, int]) -> list[tuple[int, int]]:
    lines = raw.decode("utf-8").splitlines()
    if lines and lines[0].startswith("#version:"):
        lines = lines[1:]
    merges: list[tuple[int, int]] = []
    for line in lines:
        if not line:
            continue
        # 分隔符只有一个空格，且 token 内部绝不含空格（置换后 0x20 显示为 "Ġ"），
        # 所以 split(" ") 是安全的；用 split() 反而会把 token 里的空白误切。
        parts = line.split(" ")
        if len(parts) != 2:
            raise CheckpointError(f"merges.txt 行格式非法：{line!r}")
        a, b = (display_to_token(p) for p in parts)
        if a not in word_id_map or b not in word_id_map:
            raise CheckpointError(f"merges.txt 引用了词表里没有的 token：{line!r}")
        merges.append((word_id_map[a], word_id_map[b]))
    return merges


def encode_words(word_freq: dict[Word, int]) -> bytes:
    """词频表 -> npz（ids 连续存放 + offsets 分段 + counts）。

    为什么不用 JSON：续训要靠这张表精确定位"哪些词含某个 pair"，是必须完整保存的
    真值。几百万条词频写成 JSON 会大 4-6 倍、慢 10 倍以上，而且浮点/整型语义模糊。
    用定长整型数组还能直接 mmap/流式读，内存峰值可控。
    """
    import io

    words = list(word_freq.items())
    lengths = np.fromiter((len(w) for w, _ in words), dtype=np.int64, count=len(words))
    offsets = np.zeros(len(words) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    total = int(offsets[-1])
    ids = np.empty(total, dtype=ID_DTYPE)
    cursor = 0
    for word, _ in words:
        max_id = max(word) if word else 0
        if max_id >= (1 << ID_DTYPE.itemsize * 8):
            raise CheckpointError(
                f"token id {max_id} 超出 {ID_DTYPE.name} 表示范围，需要更换落盘 dtype"
            )
        ids[cursor : cursor + len(word)] = word
        cursor += len(word)
    counts = np.fromiter((c for _, c in words), dtype=FREQ_DTYPE, count=len(words))
    buffer = io.BytesIO()
    # 不压缩：这三段都是整型流，zlib 只能省约 40% 体积，代价是每次保存多花几秒，
    # 而保存是"每 N 次合并"就发生一次的热点操作。
    np.savez(buffer, ids=ids, offsets=offsets, counts=counts)
    return buffer.getvalue()


def decode_words(raw: bytes) -> dict[Word, int]:
    import io

    with np.load(io.BytesIO(raw)) as payload:
        ids = payload["ids"]
        offsets = payload["offsets"]
        counts = payload["counts"]
    # 用 stdlib array 而不是 numpy 逐行取值：array 的切片 + tuple() 是 C 层循环，
    # 对几百万个词解包能快 3-5 倍；numpy 每个元素过一遍 Python 对象协议是不可接受的。
    view = array("H")
    view.frombytes(np.ascontiguousarray(ids, dtype="<u2").tobytes())
    out: dict[Word, int] = {}
    starts = offsets.tolist()
    freqs = counts.tolist()
    for index, freq in enumerate(freqs):
        out[tuple(view[starts[index] : starts[index + 1]])] = freq
    return out


def encode_counter(counter: dict[int, int]) -> bytes:
    import io

    items = list(counter.items())
    keys = np.fromiter((k for k, _ in items), dtype=KEY_DTYPE, count=len(items))
    freqs = np.fromiter((v for _, v in items), dtype=FREQ_DTYPE, count=len(items))
    buffer = io.BytesIO()
    np.savez(buffer, keys=keys, freqs=freqs)
    return buffer.getvalue()


def decode_counter(raw: bytes) -> dict[int, int]:
    import io

    with np.load(io.BytesIO(raw)) as payload:
        return dict(zip(payload["keys"].tolist(), payload["freqs"].tolist()))


# ---------------------------------------------------------------------------
# 对外主接口
# ---------------------------------------------------------------------------
def save_checkpoint(
    engine: MergeEngine,
    dirs: ParamDirs,
    *,
    generation: int,
    extra_meta: dict | None = None,
) -> Path:
    """把词表/规则/counter/词频表四份产物作为一个整体提交到 PARAM_DIR。"""
    dirs.ensure()
    param_dir = dirs.param_dir
    spec = engine.spec

    artifacts = {
        VOCAB_FILE: encode_vocab(engine.id_word_map),
        MERGES_FILE: encode_merges(engine.id_word_map, engine.merges),
        COUNTER_FILE: encode_counter(engine.pair_freq),
        WORDS_FILE: encode_words(engine.word_freq),
    }

    # 第 1 步：全部写成 .tmp，任一个失败就中止，正式文件一个字节都没被动过。
    digests: dict[str, dict[str, int | str]] = {}
    for name, payload in artifacts.items():
        target = param_dir / name
        sha, size = _write_and_hash(target, payload)
        digests[name] = {"sha256": sha, "bytes": size}

    # 第 2 步：原子替换。此刻若崩溃，meta 仍是上一代 -> 加载时哈希不符 -> 拒绝续训。
    for name in artifacts:
        _replace_atomically(param_dir / (name + ".tmp"), param_dir / name)

    # 第 3 步：写提交点。
    meta = {
        "format_version": PARAM_FORMAT_VERSION,
        "generation": generation,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "vocab": {
            "base_vocabulary_size": spec.base_vocabulary_size,
            "max_vocabulary_size": spec.max_vocabulary_size,
            "end_of_text_id": spec.end_of_text_id,
            "flatten_base": spec.flatten_base,
        },
        # 预切分版本戳单独存一项：它不属于 id 空间，但同样决定词表定义。
        # 换正则等于换词表，旧词表配新切分会"不报错但输出全是垃圾 token"。
        "pretoken_version": PRETOKEN_VERSION,
        "counts": {
            "vocab_size": engine.vocab_size,
            "merges": len(engine.merges),
            "unique_words": len(engine.word_freq),
            "pairs": len(engine.pair_freq),
            "next_id": engine.next_id,
        },
        "files": digests,
    }
    if extra_meta:
        meta["extra"] = extra_meta
    # meta 自己不参与哈希（没有"谁来哈希哈希者"的答案）。它的安全性来自两点：
    # JSON 截断会在解析期就抛错；而它只在所有产物 replace 成功之后才被写下。
    _write_and_hash(
        param_dir / META_FILE,
        json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    _replace_atomically(param_dir / (META_FILE + ".tmp"), param_dir / META_FILE)
    return param_dir / META_FILE


def checkpoint_exists(dirs: ParamDirs) -> bool:
    return dirs.param_dir is not None and (dirs.param_dir / META_FILE).exists()


def load_checkpoint(
    dirs: ParamDirs,
    spec: VocabSpec,
    *,
    verify_hashes: bool = True,
) -> LoadedCheckpoint | None:
    """加载一个完整且自洽的 checkpoint；没有任何有效产物时返回 None。

    verify_hashes=False 只应在"确定介质可信"时使用（例如刚保存完就在同一进程续训），
    默认开启，因为这是唯一能发现"写了一半"的手段。
    """
    param_dir = dirs.param_dir
    meta_path = param_dir / META_FILE
    if not meta_path.exists():
        return None

    meta = json.loads(meta_path.read_text("utf-8"))
    if int(meta.get("format_version", -1)) != PARAM_FORMAT_VERSION:
        raise CheckpointError(
            f"参数格式版本 {meta.get('format_version')} 与当前代码期望的 "
            f"{PARAM_FORMAT_VERSION} 不匹配，请显式迁移或重训"
        )
    vocab_meta = meta.get("vocab", {})
    checked_fields = ("base_vocabulary_size", "max_vocabulary_size", "end_of_text_id")
    mismatched = {
        k: (v, getattr(spec, k))
        for k, v in vocab_meta.items()
        if k in checked_fields and getattr(spec, k) != v
    }
    if mismatched:
        raise CheckpointError(
            f"checkpoint 的词表配置与当前实例不同，不能混用：{mismatched}（左=文件, 右=内存）。"
            "注意 flatten_base 由 max_vocabulary_size 推出，改了上限就等于改了 counter 的 key 空间"
        )
    recorded_version = meta.get("pretoken_version")
    if recorded_version != PRETOKEN_VERSION:
        raise CheckpointError(
            f"checkpoint 由预切分版本 {recorded_version!r} 训练，当前代码是 {PRETOKEN_VERSION!r}。"
            "片段边界变了就意味着词表定义变了，必须重训，不能拿旧词表推理"
        )

    raw: dict[str, bytes] = {}
    for name, record in meta["files"].items():
        path = param_dir / name
        if not path.exists():
            raise CheckpointError(f"提交点记录的文件缺失：{path}")
        data = path.read_bytes()
        if len(data) != int(record["bytes"]):
            raise CheckpointError(
                f"{name} 长度 {len(data)} 与提交点记录 {record['bytes']} 不符，"
                "典型成因：上一次保存中途被强杀"
            )
        if verify_hashes:
            digest = hashlib.sha256(data).hexdigest()
            if digest != record["sha256"]:
                raise CheckpointError(
                    f"{name} 内容 sha256 与提交点不符，拒绝加载可能被截断/篡改的产物"
                )
        raw[name] = data

    id_word_map = decode_vocab(raw[VOCAB_FILE])
    word_id_map = {token: i for i, token in id_word_map.items()}
    if len(word_id_map) != len(id_word_map):
        raise CheckpointError("词表存在两个 id 指向同一 token 字面形式")
    merges = decode_merges(raw[MERGES_FILE], word_id_map)
    if len(merges) != int(meta["counts"]["merges"]):
        raise CheckpointError("merges.txt 行数与提交点记录不一致")

    checkpoint = LoadedCheckpoint(
        id_word_map=id_word_map,
        merges=merges,
        word_freq=decode_words(raw[WORDS_FILE]),
        counter=decode_counter(raw[COUNTER_FILE]),
        generation=int(meta.get("generation", 0)),
        meta=meta,
    )
    return checkpoint


def describe(dirs: ParamDirs) -> str:
    """打印当前 checkpoint 的人读摘要，供 CLI 的 inspect 模式使用。"""
    path = dirs.param_dir / META_FILE if dirs.param_dir else None
    if path is None or not path.exists():
        return "尚无 checkpoint"
    meta = json.loads(path.read_text("utf-8"))
    counts = meta.get("counts", {})
    size = sum(int(v["bytes"]) for v in meta.get("files", {}).values())
    return (
        f"generation={meta.get('generation')} 词表={counts.get('vocab_size')} "
        f"规则={counts.get('merges')} 唯一词={counts.get('unique_words')} "
        f"pair={counts.get('pairs')} 体积={size / 1e6:.1f}MB 保存于={meta.get('saved_at')}"
    )
