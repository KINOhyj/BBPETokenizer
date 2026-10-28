"""GPT-2 风格的预切分（pre-tokenization）与字节级 id 映射。

# 这一层为什么必须独立存在
"文本 -> 基础 id 序列" 是全流程唯一的入口，训练和推理都必须走同一份实现。
如果两边各写一遍，任何一边改了正则，训出来的词表就和推理时的切分不一致，
表现为"能跑但 token 全是碎的"，且极难定位。所以这里是唯一的真值来源。

# 什么是预切分，为什么它不可省略
字节级 BPE 并不是在整个字节流上自由合并的：先用正则把文本切成若干"原子片段"
（GPT-2 的片段大致是 单词/数字串/标点串/空白串），然后**每个片段内部独立**做 BPE。
这条"不允许跨片段合并"的约束是 BPE 训练语义的一部分：
  - 没有它，词表会学到 "word 词边界 + 下一个词首字母" 这种无意义跨词 pair；
  - 有了它，训练只需要保留片段(词)的集合与频次，内存量从"语料长度"降到
    "唯一片段数" —— 这正是本项目敢于流式处理 GB 级语料的前提。
"""

from __future__ import annotations

try:
    import regex as _re  # 必须是第三方 regex：下面用到 \p{L} 这类 Unicode 属性类
except ImportError as exc:  # pragma: no cover - 环境缺依赖时给出可操作提示
    raise ImportError(
        "BBPE 的预切分依赖第三方 `regex` 模块（标准库 re 不支持 \\p{L} 属性类，"
        "会把中文/日文等所有非 ASCII 字母误判）。请先执行 `uv sync`。"
    ) from exc

# GPT-2 官方预切分模式（HuggingFace tokenizers 里也用它）。
# 四个分支的顺序是有意义的，regex 的 alternation 取最先匹配：
#   1) 英文缩写后缀  's 't 're 've 'm 'n 'll —— 必须优先，否则 ' 会被归到标点串
#   2) 可选前导空白 + 一串字母          —— 空白并进词首，是 GPT-2 用 "Ġ" 表示空格的原因
#   3) 可选前导空白 + 一串数字
#   4) 可选前导空白 + 一串既非空白也非字母数字（标点/符号）
#   5) \s+(?!\S)  —— 只有"后面还跟着非空白"的空白才并进下一个片段；纯尾随空白走分支 6
#   6) \s+        —— 兜底：连续空白自成片段（保证 round-trip 无损，不丢任何字符）
PAT_STR = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# 预切分版本戳。会被写进 checkpoint 的 meta 并在加载时比对。
# 它不是"功能开关"，而是防"静默错位"：换正则等于换词表定义，
# 旧词表 + 新切分 = 不报错但全是垃圾 token。
PRETOKEN_VERSION = "gpt2"

PAT = _re.compile(PAT_STR)


def segment(text: str) -> list[str]:
    """把文本切成预切分片段（BPE 的合并边界）。

    返回的片段拼接起来与原文完全一致（无损切分），这是 round-trip 测试的依据。

    **一条踩过的坑，别再改回去**：中文没有空格，看起来"应该"把每个汉字单独切成一片
    （本项目实测过 ` ?\\p{Han}` 这种分支）。但片段边界就是合并边界 —— 逐字切分后
    BPE 只能在**一个字内部**合并 3 个字节，永远无法把"机器"+"学习"合成一个 token。
    实测同一份 2MB 语料：唯一片段从 46,016 掉到 830，合并次数在 959 次就撞频率阈值，
    中文词表被锁死在 ~1200 条，压缩率彻底废掉。
    保持"整段汉字是一个片段"，BPE 才会自动走出 字节 -> 汉字 -> 词组 的层级。
    """
    return PAT.findall(text)


# ---------------------------------------------------------------------------
# 字节 <-> 基础 id
# ---------------------------------------------------------------------------
# 字节级 BPE 的核心约定：基础 id == 该字节的数值（0..255）。
# 也就是说 utf-8 编码之后不需要任何"查表转换"，byte 本身就是 token id。
# 保留下面的显式函数，是为了让骨架注释里"把每一个字节替换成基础词表中的 id"
# 这一步在代码里可见，而不是被 `list(b"...")` 这种隐式行为替代。


def text_to_bytes(text: str) -> bytes:
    """统一用 utf-8。字节级词表的意义就在于任何语言都能无损落到字节上。"""
    return text.encode("utf-8")


def bytes_to_ids(raw: bytes) -> list[int]:
    """每个字节 -> 基础词表 id（数值相同，此处只是显式表达映射关系）。"""
    return list(raw)


def ids_to_bytes(ids: list[int] | tuple[int, ...]) -> bytes:
    """bytes_to_ids 的逆运算。"""
    return bytes(bytearray(ids))


def text_to_ids(text: str) -> list[int]:
    """text -> utf-8 字节 id 序列（不切分，用于单测与调试）。"""
    return bytes_to_ids(text_to_bytes(text))


def ids_to_text(ids, *, errors: str = "replace") -> str:
    """id 序列 -> 文本。

    errors 默认 "replace"：字节级词表的中间产物常常是半个多字节字符（比如中文
    utf-8 三字节被切成 1+2），解码必然失败。解码失败不代表 tokenizer 坏了，
    只有整段序列才是合法单位，所以这里不做严格校验、也不抛异常。
    """
    return ids_to_bytes(list(ids)).decode("utf-8", errors=errors)


# ---------------------------------------------------------------------------
# 词表文件的可读编码（只在 IO 边界使用，内部一律用 bytes 当 key）
# ---------------------------------------------------------------------------
# GPT-2 的 vocab.json 用"字节 -> 可打印字符"的置换当 key，好处是文件人类可读、
# 且能被 transformers 直接加载。内部如果用这个字符串当 key，就会把
# "显示的表示"和"真实的字节"混在一起，一旦遇到非法 utf-8 就出 bug，
# 所以置换严格限制在存盘/读盘这一层。
def bytes_to_unicode() -> dict[int, str]:
    """返回 0..255 每个字节对应的可打印字符（**与 GPT-2 官方逐位一致**）。

    可打印区 33..126、161..172、174..255 直接映射到自身；剩下的 68 个字节
    （控制字符、空格、软连字符 173 等）映射到 256 起的连续码点，保证双射且都可打印。

    为什么强调"逐位一致"：这不是随手挑的一段码点，而是 GPT-2 vocab.json 的事实标准。
    只要有一个字节映射不同，导出的词表就无法被 transformers/HF tokenizers 直接加载，
    同一个字节还会在跨工具比对时显示成另一个字符。单测里锁死了几个关键映射
    （0x20 -> 'Ġ'、0xA0 -> 'Ġ' 邻近区）来防住这种漂移。
    """
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = list(bs.copy())
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_BYTE_TO_CHAR = bytes_to_unicode()
_CHAR_TO_BYTE = {v: k for k, v in _BYTE_TO_CHAR.items()}
assert len(_CHAR_TO_BYTE) == 256, "字节置换必须双射，否则词表存盘会有碰撞"


def token_to_display(token: bytes) -> str:
    """token 字节 -> 词表文件里的显示串（单射，可逆）。"""
    return "".join(_BYTE_TO_CHAR[b] for b in token)


def display_to_token(text: str) -> bytes:
    """token_to_display 的逆运算。"""
    return bytes(_CHAR_TO_BYTE[c] for c in text)
