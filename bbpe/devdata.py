"""离线假语料：冒烟测试与单元测试共用的数据替身。

# 为什么要专门一个模块
"能不能跑通训练"这个问题不该依赖网络，但也不该依赖硬编码在测试文件里的字符串 ——
那样既没法控制规模，也没法复现性能问题。这里用**确定性生成器**（固定 seed）产出
带真实统计特征的文本：词频长尾、中英混合、重复页脚、符号刷屏，
这些都是清洗层和合并层必须对付的实际形态。

生成而不是抽取，还顺带保证了一个重要性质：**真值可知**。
假语料里哪些 pair 高频是可以手算的，单测据此核对合并结果是否符合预期。
"""

from __future__ import annotations

import bisect
import random

from .corpus_stream import CorpusSource  # noqa: F401  (re-export，方便测试与 CLI 直接导入)

# 高频英文词：让 a/the/of/to 占大头，BPE 早期合并必然发生在它们内部，
# 于是单测可以断言 "Ġthe" 一定出现在 rank 很小的位置。
EN_WORDS = [
    "the", "a", "of", "and", "to", "in", "is", "that", "it", "for",
    "as", "with", "was", "on", "be", "by", "this", "have", "from", "or",
]

# Zipf 形长尾的累积分布，模块级算一次即可（放在函数里会让生成语料慢一个数量级）。
_EN_WEIGHTS = [1.0 / (i + 2) for i in range(len(EN_WORDS))]
_EN_TOTAL = sum(_EN_WEIGHTS)
_EN_CUMULATIVE = []
_acc = 0.0
for _w in _EN_WEIGHTS:
    _acc += _w / _EN_TOTAL
    _EN_CUMULATIVE.append(_acc)

ZH_WORDS = ["机器", "学习", "训练", "词表", "分词", "中文", "语言", "模型", "数据", "算法", "文本", "字符"]

# 常用字打底 + 真实码位补齐成长尾。只放几十字的话，中文侧唯一片段会瞬间饱和，
# 训练在几百步内就因"最高频 pair < 阈值"停下，测不出规模效应，也无法和英文侧对比。
_COMMON_ZH = "的一是了我不在有这个人他们对上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能没小很好于起什之"
ZH_CHARS = _COMMON_ZH + "".join(chr(0x4E00 + i * 7) for i in range(700))

# Zipf 权重：真实语料里"的/一/是"的频率是生僻字的几百倍，直接均匀抽样会让词表学不出层次。
_ZH_WEIGHTS = [1.0 / (i + 1.35) for i in range(len(ZH_CHARS))]
_ZH_TOTAL = sum(_ZH_WEIGHTS)
_ZH_CUMULATIVE = []
_acc = 0.0
for _w in _ZH_WEIGHTS:
    _acc += _w / _ZH_TOTAL
    _ZH_CUMULATIVE.append(_acc)


def _pick_zh_char(rng: random.Random) -> str:
    """按 Zipf 分布抽一个汉字（bisect 而不是线性扫描：每生成一条记录要抽几十次）。"""
    return ZH_CHARS[bisect.bisect_left(_ZH_CUMULATIVE, rng.random())]

FOOTER = "本文讨论的内容来自网络，转载请注明出处。"


def _english_sentence(rng: random.Random) -> str:
    words = []
    for _ in range(rng.randint(6, 18)):
        pick = rng.random()
        index = len(EN_WORDS) - 1
        for i, bound in enumerate(_EN_CUMULATIVE):
            if pick <= bound:
                index = i
                break
        words.append(EN_WORDS[index])
    return " ".join(words) + "."


ZH_PUNCT = "，。、；：？！"


def _chinese_sentence(rng: random.Random) -> str:
    # 三成用"词"拼（让双字词的 pair 能被稳定统计到），七成用"字"随机拼（模拟真实汉字的
    # pair 分散性）。两者的比例直接决定中文侧词表能长多大，是冒烟测试的调节旋钮。
    clauses = []
    for _ in range(rng.randint(2, 4)):
        if rng.random() < 0.35:
            clauses.append("".join(rng.choice(ZH_WORDS) for _ in range(rng.randint(2, 5))))
        else:
            clauses.append("".join(_pick_zh_char(rng) for _ in range(rng.randint(3, 8))))
    # 标点是真实中文分句的主要边界。没有它，整段会被 GPT-2 正则切成一个几百字节的
    # 片段，预切分退化成"整条记录一个 token"，去重与合并成本都会失真 —— 基准就没法信。
    return "，".join(clauses) + rng.choice("。！？")


def make_noise(rng: random.Random) -> str:
    """生成一条典型"爬虫噪声"，用来验证清洗层真的把它压掉了。"""
    kind = rng.randrange(5)
    if kind == 0:
        return "!" * 14 + "…" * 8
    if kind == 1:
        return "----" * 20
    if kind == 2:
        return FOOTER + "\n" + FOOTER + "\n" + FOOTER
    if kind == 3:
        return "很好" * 12
    return "\x00\x0b" + "​‮" + "数据" * 3


def synthetic_items(
    doc_count: int = 4000,
    *,
    seed: int = 20261028,
    zh_ratio: float = 0.5,
    noise_ratio: float = 0.15,
    footer_ratio: float = 0.2,
) -> list[tuple[str, str]]:
    """产出 (lang, text) 列表，形状与 accumulate_corpus 的输入完全一致。

    同一个 seed 必然得到同一份语料 —— 这是冒烟测试可复现的基础，
    也是"两次训练得到同一个词表"这条断言能成立的前提。
    """
    rng = random.Random(seed)
    items: list[tuple[str, str]] = []
    for _ in range(doc_count):
        if rng.random() < noise_ratio:
            items.append(("noise", make_noise(rng)))
            continue
        if rng.random() < zh_ratio:
            text = "".join(_chinese_sentence(rng) for _ in range(rng.randint(2, 6)))
            lang = "zh"
        else:
            text = " ".join(_english_sentence(rng) for _ in range(rng.randint(2, 5)))
            lang = "en"
        if rng.random() < footer_ratio:
            text += "\n" + FOOTER + "\n" + FOOTER + "\n"
        items.append((lang, text))
    return items
