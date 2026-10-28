"""预切分与字节映射的单元测试。

这些断言看起来"太具体"（连 'Ġ' 都写死了），但这是有意的：
预切分结果和字节置换表是**词表的定义本身**，一旦变动，已训好的词表就全部失效。
把它们钉死在测试里，改动时才会被逼着做一次"我知道这会作废现有词表"的决定。
"""

from __future__ import annotations

import pytest

from bbpe.pretokenize import (
    PAT_STR,
    bytes_to_unicode,
    display_to_token,
    ids_to_bytes,
    ids_to_text,
    segment,
    text_to_bytes,
    text_to_ids,
    token_to_display,
)


class TestSegment:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Hello, world!", ["Hello", ",", " world", "!"]),
            ("it's a test", ["it", "'s", " a", " test"]),
            ("abc123!@# 42", ["abc", "123", "!@#", " 42"]),
            # 连续空白的切法最容易被写错：\s+(?!\S) 保证只有"后面还有内容"的空白
            # 才并进下一个片段，纯尾随空白自成片段。
            ("  spaced  out", [" ", " spaced", " ", " out"]),
            ("a\n\nb", ["a", "\n", "\n", "b"]),
            # 中文整体是一串 \p{L}，会成一个长片段；这正是字节级 BPE 需要的形态
            ("机器学习的训练", ["机器学习的训练"]),
            ("", []),
        ],
    )
    def test_known_splits(self, text, expected):
        assert segment(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "Hello, world! 你好，世界。",
            "  leading and trailing   \n\t tabs",
            "a1 b2 。。。!!!'''",
            "混合 mixed 文本 text 123 符号 @#% —— 破折号",
            "\x00控制字符\x0b也存在",
        ],
    )
    def test_lossless(self, text):
        """预切分必须无损：片段拼回去等于原文。

        有损的话推理会静默丢字符，而训练时统计到的 pair 与推理时能取到的 pair 不再同构。
        """
        assert "".join(segment(text)) == text

    def test_no_empty_piece(self):
        assert all(piece for piece in segment("a b  c\ndddd ！！！中文"))

    def test_pattern_is_gpt2(self):
        """锁死 GPT-2 官方模式串，防止有人"顺手优化"正则。"""
        assert PAT_STR == (
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )


class TestByteIds:
    def test_byte_id_identity(self):
        """字节级 BPE 的核心约定：基础 id 就是字节值本身，不需要任何查表转换。"""
        for text in ["A", "\x00", "你", "😀", "~"]:
            assert text_to_ids(text) == list(text.encode("utf-8"))
        assert text_to_ids("A") == [65]

    def test_all_byte_ids_in_base_range(self):
        assert all(0 <= i <= 255 for i in text_to_ids("任何文本 ✓ 😀"))

    def test_ids_bytes_roundtrip(self):
        ids = text_to_ids("中英 mixed 混 合")
        assert ids_to_bytes(ids) == "中英 mixed 混 合".encode("utf-8")
        assert ids_to_text(ids) == "中英 mixed 混 合"

    def test_decode_partial_multibyte_does_not_raise(self):
        """半个汉字必须能解码（推理中间态就是这个形状），不能抛异常。"""
        ids = text_to_ids("你")[1:]  # 只剩 2/3 个字节
        assert isinstance(ids_to_text(ids), str)


class TestDisplayCodec:
    def test_mapping_is_bijection(self):
        table = bytes_to_unicode()
        assert len(table) == 256
        assert len(set(table.values())) == 256
        assert set(table) == set(range(256))

    def test_gpt2_landmark_mappings(self):
        """几个必须逐位一致的锚点：错了就意味着 vocab.json 与 GPT-2 不兼容。"""
        table = bytes_to_unicode()
        assert table[33] == "!"          # 可打印区起点
        assert table[126] == "~"         # 可打印区终点
        assert table[161] == "¡"
        assert table[172] == "¬"
        assert table[174] == "®"
        assert table[255] == "ÿ"
        assert table[32] == "Ġ"          # 空格（最常见的 token 前缀）
        assert table[0] == "Ā"
        assert table[173] == "Ń"         # 软连字符不在可打印区，易被误包含

    def test_display_roundtrip_over_all_bytes(self):
        token = bytes(range(256))
        display = token_to_display(token)
        assert display_to_token(display) == token

    def test_display_contains_no_space(self):
        """词表文件名里绝不能出现真空格，否则 merges.txt 的 "A B" 分隔符会歧义。"""
        assert " " not in token_to_display(bytes(range(256)))

    def test_display_is_printable(self):
        """置换的目的就是让 vocab.json 可以肉眼读：控制字节必须映射到可打印字符。"""
        display = token_to_display(b"\x00\x1f abc\x7f")
        assert display.isprintable()
