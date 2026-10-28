"""清洗层单元测试。

重点不是"正则有没有写错"，而是**清洗强度是否可控**：
太弱 -> 噪声吃掉词表名额；太强 -> 把真实语言现象（叠词、语气标点）当成噪声删掉。
所以每个断言都同时给出"必须删掉"和"必须保留"的两面。

测试文本里的特殊字符一律写转义（\\uFEFF 而不是贴一个真实的 BOM）：
在源文件里放不可见字符，没人能在 review 时确认它到底是什么。
"""

from __future__ import annotations

from bbpe.devdata import FOOTER
from bbpe.text_clean import CleanConfig, clean_text, dedup_window


class TestCharacterLevel:
    def test_removes_bom_and_zero_width(self):
        assert clean_text("你好\ufeff世界") == "你好世界"
        assert clean_text("a\u200bb\u202ec") == "abc"
        assert clean_text("软\u00ad连字符") == "软连字符"  # 软连字符

    def test_special_space_becomes_normal_space(self):
        """全角/不换行空格转普通空格而不是删除：删了会把两个词粘成一个假词。"""
        assert clean_text("训练\u3000数据") == "训练 数据"  # 全角空格
        assert clean_text("a\u00a0b") == "a b"  # NBSP

    def test_removes_control_chars(self):
        assert clean_text("a\x00\x0b\x1fb") == "ab"
        assert clean_text("保留\x09制表符") == "保留 制表符"

    def test_removes_replacement_char(self):
        assert clean_text("解码\ufffd失败") == "解码失败"

    def test_removes_private_use_area(self):
        assert clean_text("a\ue000b") == "ab"

    def test_empty_and_blank(self):
        assert clean_text("") == ""
        assert clean_text("   ") == ""


class TestSymbolRuns:
    def test_collapses_long_symbol_run_to_three(self):
        assert clean_text("!" * 20) == "!!!"
        # 行内的长破折号会被塌缩；但"整行都是破折号"属于装饰线，由装饰线规则优先整行删除
        assert clean_text("文字 ———————— 文字") == "文字 ——— 文字"

    def test_full_line_of_dashes_is_removed_as_decorator(self):
        """记录规则优先级：整行符号 = 装饰线（删除），行内符号 = 重复符号（塌缩）。"""
        assert clean_text("第一段\n——————————\n第二段") == "第一段 第二段"

    def test_keeps_short_natural_runs(self):
        """中文语气标点与省略号是语言现象，不能压平。"""
        assert clean_text("！！！") == "！！！"
        assert clean_text("……") == "……"

    def test_does_not_merge_different_symbols(self):
        assert clean_text("!?!?!?") == "!?!?!?"   # 交替出现不构成"同一符号的连续重复"


class TestDuplicateContent:
    def test_consecutive_duplicate_lines_collapse(self):
        assert clean_text("正文一段。\n正文一段。\n正文一段。") == "正文一段。"

    def test_footer_repetition_is_removed(self):
        assert clean_text(FOOTER + "\n" + FOOTER + "\n" + FOOTER) == FOOTER

    def test_full_decorator_line_removed(self):
        assert clean_text("第一段\n----------\n第二段") == "第一段 第二段"

    def test_inline_dashes_only_collapsed_not_removed(self):
        """行内的长破折号只会被塌缩成 3 个，不会被当装饰线删除。"""
        assert clean_text("标题 ---------- 分栏") == "标题 --- 分栏"

    def test_word_run_collapsed(self):
        assert clean_text("很好很好很好很好") == "很好很好"

    def test_normal_repetition_under_threshold_kept(self):
        """只出现两次的短语必须原样保留，否则会把正常语言当成噪声。"""
        assert clean_text("数据数据") == "数据数据"


class TestWhitespaceAndLimits:
    def test_all_whitespace_collapses_to_single_space(self):
        assert clean_text("a  b\t\tc\nd") == "a b c d"

    def test_truncates_overlong_text(self):
        """用全不相同的字符构造长文本：重复串会先被去重规则改写，测不出截断。"""
        config = CleanConfig(max_text_chars=100)
        text = "".join(chr(0x4E00 + i) for i in range(500))
        assert clean_text(text, config) == text[:100]

    def test_min_length_gate(self):
        assert clean_text("x", CleanConfig(min_text_chars=2)) == ""

    def test_duplicate_line_removal_can_be_disabled(self):
        config = CleanConfig(drop_duplicate_lines=False)
        assert clean_text("重复\n重复", config) == "重复 重复"


class TestDedupWindow:
    # 默认阈值是 24 个字符，所以测试里的"长片段"必须真的超过它。
    LONG_A = "同一段较长的文本内容，用来模拟爬虫反复抓到的页脚声明。"
    LONG_B = "另一段同样很长的文本内容，用来确认顺序与去重互不干扰。"

    def test_drops_repeat_of_long_chunk(self):
        chunks = [self.LONG_A, "短", self.LONG_A, self.LONG_B]
        assert dedup_window(chunks) == [self.LONG_A, "短", self.LONG_B]

    def test_keeps_short_chunks_even_if_repeated(self):
        """高频短语恰恰是要统计的语言现象，去重窗口只处理长片段。"""
        assert dedup_window(["是的", "是的", "是的"]) == ["是的", "是的", "是的"]

    def test_threshold_is_configurable(self):
        assert dedup_window(["xxxxx", "xxxxx"], min_chars=3) == ["xxxxx"]
        assert dedup_window(["xxx", "xxx"], min_chars=5) == ["xxx", "xxx"]

    def test_order_preserved(self):
        chunks = ["a" * 30, "b" * 30, "a" * 30, "c" * 30]
        assert dedup_window(chunks) == ["a" * 30, "b" * 30, "c" * 30]
