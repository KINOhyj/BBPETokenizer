"""语料清洗：把"原始网页/wiki 文本"变成"值得拿去统计词频的文本"。

# 为什么这一层对词表质量是决定性的
BPE 词表是按 pair 频率自底向上长出来的。频率被什么占据，词表就被什么占满。
CCI3/finewiki 这类爬取语料里，噪声的重复度远高于正常语言：
  - 页眉页脚、"编辑"链接、免责声明会整段重复上百次；
  - 中文 wiki 里连续的标点、省略号、装饰符号 "——————" "………" 频次极高。
如果不清洗，几千个合并名额会被 "—— + ——" 这种 pair 吃掉，
真正有语言价值的 pair 因为排在后面而拿不到名额 —— 而且**这个过程不可逆**，
词表长度一满就停训了。所以清洗必须在统计之前，而不是推理之后。

# 设计约束
本模块只暴露纯函数：输入 str、输出 str，不读文件、不碰全局状态。
这样 _prepare 的缓冲区、测试的 mock 数据、将来的新数据源都能复用同一套规则。

# 顺序有讲究
格式字符/私用区删除 -> 特殊空白转普通空格 -> 换行统一 -> 装饰线与重复行折叠
-> 符号长串塌缩 -> 空白归一。
先清不可见字符再判重复，否则同一段落会因为混入零宽字符而被判定为"不重复"。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 字符级清理
# ---------------------------------------------------------------------------
# 零宽字符 / 双向控制符 / BOM / 软连字符：不参与语义，却会拆断 pair 统计
# （同一段文字插入一个零宽字符就变成两个不同的 pair），必须最先删掉。
# 这里一律写 \uXXXX 转义而不是字面字符 —— 源码里放不可见字符本身就是 bug 源，
# 没人能在 review 时看出方括号里到底写了几个字符。
FORMAT_CHAR_RE = re.compile(
    "["
    "\u00ad"               # soft hyphen
    "\u034f"               # combining grapheme joiner
    "\u061c"               # arabic letter mark
    "\u180e"               # mongolian vowel separator
    "\u200b-\u200f"        # ZWSP / ZWNJ / ZWJ / LRM / RLM
    "\u202a-\u202e"        # bidi embedding & override
    "\u2060-\u2064"        # word joiner / invisible operators
    "\u206a-\u206f"        # bidi isolates & nudges
    "\ufeff"               # BOM / ZWNBSP
    "\ufff9-\ufffb"        # interlinear annotation
    "]"
)

# 不换行空格与全角空格类：它们是"排版空白"而不是"语言字符"，
# 转成普通空格交给最后的空白归一步骤；直接删会把两个词粘成一个假词
# （中文语料里 \u3000 全角空格尤其常见）。
SPECIAL_SPACE_RE = re.compile("[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]")

# C0/C1 控制字符（保留 \t \n \r，它们在后续步骤会被归一）。
# 语料里出现 \x00-\x08 通常意味着上游编码错误，留着只会污染统计。
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# 替换符 U+FFFD：上游解码失败的明确标记，出现即删（对应字节序列在词表里没有意义）。
REPLACEMENT_RE = re.compile("\ufffd")

# 私用区字符：各机构自定义、跨语料完全不通用，保留会让词表学到无意义 token。
PRIVATE_USE_RE = re.compile("[\ue000-\uf8ff]")

# ---------------------------------------------------------------------------
# 结构级清理
# ---------------------------------------------------------------------------
# 换行统一：\r\n 和 \r 都归成 \n。不统一的话 "\n" 与 "\r\n" 会分裂成两套边界上下文。
CRLF_RE = re.compile(r"\r\n?")

# 连续重复的同一个"既非单词字符也非空白"的符号：出现 3 次以上塌缩到 3 次。
# 为什么保留 3 个而不是全部压成 1 个：中文里 "！！！" "？？？" "……" 本身带语气信息，
# 全压平会破坏语言现象；但 "!!!!!…!…" 这种刷屏是噪声，压掉能省下大量合并名额。
# 字符类 [^\w\s] 在 str 模式下是 Unicode 语义，中英文标点都能覆盖。
SYMBOL_RUN_RE = re.compile(r"([^\w\s])\1{2,}")

# 连续重复的"词"：如 "很好很好很好很好" -> "很好很好"。
# 只针对 1-8 字符的短片段且要求连续出现 4 次以上，避免误伤正常叠词和长句。
# 用惰性 \S{1,8}? 而不是贪婪：贪婪会把 "abab" 里的更长重复单元漏判。
WORD_RUN_RE = re.compile(r"(\S{1,8}?)\1{3,}")

# 整行重复（爬虫表格/页脚/版权声明的典型形态）：连续多行完全一致时只留一行。
CONSECUTIVE_DUP_LINE_RE = re.compile(r"(?m)^([^\n]{4,})$(?:\n\1$)+")

# 装饰性分隔线：整行只由 - = * _ ~ · — 或制表符构成且长度 >= 6。
DECORATOR_LINE_RE = re.compile(r"(?m)^\s*[\-=*_~·—─▏▕■●◆◇\s]{6,}\s*$")

# 空白归一：所有连续空白（含换行）压成一个空格。
# 为什么这么激进：训练语料的换行位置来自排版，不是语言现象；保留它只会让
# "\n" 与相邻字符的 pair 白白抢名额。预切分阶段空格仍会被并入片段（GPT-2 的 "Ġ"），
# 所以"词前有空格"这一信息不丢失，丢掉的只是"在第几个字符处换行"。
WHITESPACE_RE = re.compile(r"[ \t\r\n\f\v]+")


@dataclass(frozen=True)
class CleanConfig:
    """清洗阈值。集中成参数而不是写死在正则里，方便做消融实验。"""

    # 清洗后文本的最小长度（字符）。低于此值视为解析残留，返回空串由调用方跳过。
    min_text_chars: int = 1
    # 单条文本的最大长度（字符）。防止一条超长记录把缓冲区一次性撑爆，
    # 否则"缓冲区按字节上限 flush"的内存保证就失效了。
    max_text_chars: int = 200_000
    # 是否做行级去重与装饰线删除。关掉可用于对比清洗强度对词表的影响。
    drop_duplicate_lines: bool = True


DEFAULT_CLEAN_CONFIG = CleanConfig()


def clean_text(text: str, config: CleanConfig = DEFAULT_CLEAN_CONFIG) -> str:
    """清洗单条文本。

    参数:
        text  : 一条原始语料记录。
        config: 阈值配置。

    返回:
        清洗后的文本；内容被判为无价值时返回空串。
        这里返回空串而不是抛异常，是因为流式管道里一条脏数据不值得中断整个训练。
    """
    if not text:
        return ""

    # 超长截断：让调用方的内存占用可预期。
    if len(text) > config.max_text_chars:
        text = text[: config.max_text_chars]

    # 1) 字符级：先干掉不可见与非法字符，后面的重复判定才不会被它们绕过。
    text = FORMAT_CHAR_RE.sub("", text)
    text = CONTROL_RE.sub("", text)
    text = REPLACEMENT_RE.sub("", text)
    text = PRIVATE_USE_RE.sub("", text)
    text = SPECIAL_SPACE_RE.sub(" ", text)

    # 2) 结构级：换行统一 + 装饰线整行删除 + 连续重复行折叠。
    text = CRLF_RE.sub("\n", text)
    if config.drop_duplicate_lines:
        text = DECORATOR_LINE_RE.sub("\n", text)
        text = CONSECUTIVE_DUP_LINE_RE.sub(r"\1", text)

    # 3) 内容级：符号长串与词长串塌缩。
    text = SYMBOL_RUN_RE.sub(r"\1\1\1", text)
    text = WORD_RUN_RE.sub(r"\1\1", text)

    # 4) 空白归一并 strip。
    text = WHITESPACE_RE.sub(" ", text).strip()

    if len(text) < config.min_text_chars:
        return ""
    return text


def dedup_window(
    chunks: list[str],
    *,
    min_chars: int = 24,
) -> list[str]:
    """在缓冲区层面去掉完全重复的长文本片段。

    为什么需要一个 buffer 级函数：clean_text 是逐条调用的，看不到"上一条是什么"。
    爬虫语料里同一段页脚会出现在成千上万条记录中，只有把一批记录放在一起比较才抓得住。
    只比较 min_chars 以上的片段，避免把 "是的" "OK" 这种高频短语误删 ——
    它们恰恰是必须被统计的真实语言现象。
    """
    seen: set[str] = set()
    out: list[str] = []
    for chunk in chunks:
        if len(chunk) >= min_chars:
            if chunk in seen:
                continue
            seen.add(chunk)
        out.append(chunk)
    return out
