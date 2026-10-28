"""BPE 合并引擎：精确的增量式 pair 统计与合并。

# 这个模块解决的核心问题
骨架注释要求"增量更新 counter（只增加合并后 token 对，只减少被合并 token 对）"。
要做这个增量，光有 counter 本身是不够的 —— 必须知道**哪些 token 序列里含有待合并
的 pair**。反例（两份 bigram 谱完全相同的语料）：

    语料 A：词 "ab" 与词 "xa"      bigram = {(a,b):1, (x,a):1}
    语料 B：词 "xab"               bigram = {(x,a):1, (a,b):1}

合并 (a,b)->c 之后，A 得到 "c","xa" -> pair (x,c) 频率 0；B 得到 "xc" -> (x,c) 频率 1。
counter 完全一致，正确答案却不同。`flat = id1 * V + id2` 只是把 pair 编码成一个
整数 key（双射，解决"怎么存"），它不能凭空补出三元上下文（解决不了"频率从哪来"）。

所以本引擎在内存里保留三份互相支撑的结构：
    word_freq  : 唯一 surface form(当前 id 序列) -> 出现次数。真值来源，体积上界是
                 "去重后的唯一片段数"，与语料总长度无关，所以 GB 级语料也放得下。
    pair_freq  : flatten(pair) -> 频率。就是骨架里的 counter，由 word_freq 精确导出，
                 之后每步只做增量修正。
    pair_words : flatten(pair) -> 含该 pair 的词集合。增量修正的定位索引。
三者构成闭环，任何一份都能由另外两份校验，这就是"防进度不一致"的结构性保证。

# 堆为什么是"惰性"的
每次合并只修正受影响的词，pair_freq 里其余项不变。被改小的项不主动从堆里删
（删任意位置的堆项要 O(n)），取最优时若发现堆顶与当前值不符，就把它**归一化成当前值**
再放回堆（不是丢弃 —— 见 best_pair 里记录的那次真实 bug）。
代价是堆里会有重复条目，所以再加一道膨胀闸门：超过 pair 总数 4 倍就整堆重建，O(P)。
好处是 best_pair 均摊 O(log n)，而"每轮 max(pair_freq)"是 O(P) —— 60000 轮下来差三个数量级。
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

from .config import VocabSpec

# 一个"词"就是当前的 id 序列（元组可哈希，能直接当 dict/set 的 key）。
Word = tuple[int, ...]


@dataclass(frozen=True)
class MergeResult:
    """一次合并的完整回执，供上层打印进度和落盘。"""

    pair: tuple[int, int]
    new_id: int
    new_token: bytes
    frequency: int
    rank: int
    affected_words: int


class MergeEngine:
    """可独立测试的 BPE 合并状态机。不读文件、不联网、不知道数据集存在。"""

    def __init__(self, spec: VocabSpec, min_frequency: int = 2) -> None:
        self.spec = spec
        self.min_frequency = min_frequency
        # 堆膨胀倍数上限，见 _inc_pair 里的说明。
        self._heap_growth_factor = 4

        # --- 训练状态 ---
        self.word_freq: dict[Word, int] = {}
        self.pair_freq: dict[int, int] = {}
        self.pair_words: dict[int, set[Word]] = {}
        self._heap: list[tuple[int, int]] = []

        # --- 模型（训练产物，推理只需要这两样 + 基础词表）---
        self.merges: list[tuple[int, int]] = []
        self.merge_rank: dict[int, int] = {}
        self.id_word_map: dict[int, bytes] = {}
        self.word_id_map: dict[bytes, int] = {}

        # 合并 token 从 first_merge_id(=257) 起连续编号。
        # 由此得到一条贯穿全流程的不变式：new_id == first_merge_id + rank。
        self.next_id = spec.first_merge_id

        self._init_base_vocab()

    # ------------------------------------------------------------------
    # 基础词表
    # ------------------------------------------------------------------
    def _init_base_vocab(self) -> None:
        """装填 256 个字节 token 与唯一的特殊 token <|EndOfText|>。

        字节 token 的 id 恒等于字节值，因此字节化不需要查表；但这里仍然把映射
        建出来，因为它是 id_word_map 的起点，合并时的 new_token = 左 token + 右 token
        要靠这张表取字节的原始形态。
        """
        for token, i in self.spec.byte_vocab().items():
            self.id_word_map[i] = token
            self.word_id_map[token] = i
        eot = b"<|EndOfText|>"
        self.id_word_map[self.spec.end_of_text_id] = eot
        self.word_id_map[eot] = self.spec.end_of_text_id

    # ------------------------------------------------------------------
    # 准备阶段（_prepare）的写入接口
    # ------------------------------------------------------------------
    def add_word(self, word: Word, count: int = 1) -> None:
        """累加一个预切分片段的频次。

        只累加词频、不碰 pair_freq —— 配对统计留到 build_index() 一次做完。
        原因：边流式边统计 pair 会得到同样的数值，但**无法**在事后修正，
        而合并阶段一定要修正；既然一定要一套 pair->词 的索引，不如统一在
        末尾一次建好，避免两套代码路径产生偏差。
        """
        if count <= 0:
            return
        self.word_freq[word] = self.word_freq.get(word, 0) + count

    def add_words(self, words) -> None:
        """批量入口，接收 (Word, count) 可迭代对象。"""
        for word, count in words:
            self.add_word(word, count)

    # ------------------------------------------------------------------
    # 索引构建
    # ------------------------------------------------------------------
    def build_index(self) -> None:
        """由 word_freq 精确导出 pair_freq / pair_words / 堆。"""
        self.pair_freq.clear()
        self.pair_words.clear()
        pair_freq = self.pair_freq
        pair_words = self.pair_words
        flatten = self.spec.flatten

        for word, count in self.word_freq.items():
            # 逐个相邻位置统计，**允许重复**：词 (x,x,x) 含两次 (x,x)，
            # 后续增量修正也按同样的重复次数加减，两边严格抵消。
            for i in range(len(word) - 1):
                key = flatten(word[i], word[i + 1])
                pair_freq[key] = pair_freq.get(key, 0) + count
                bucket = pair_words.get(key)
                if bucket is None:
                    pair_words[key] = {word}
                else:
                    bucket.add(word)

        # 堆里存 (-频率, pair key)；Python 只有小顶堆，取负即等价于最大堆。
        self._heap = [(-f, k) for k, f in pair_freq.items()]
        heapq.heapify(self._heap)

    def recompute_pair_freq(self) -> dict[int, int]:
        """独立地全量重算 pair 频率。

        存在的意义不是生产使用，而是**验证增量逻辑没写错**：
        单测和自检里比较它与 self.pair_freq，任何一步合并后两者都必须完全相等。
        增量算法的错误通常是静默的（词表质量变差但不报错），这条不变式是唯一
        廉价的探测器。
        """
        fresh: dict[int, int] = {}
        flatten = self.spec.flatten
        for word, count in self.word_freq.items():
            for i in range(len(word) - 1):
                key = flatten(word[i], word[i + 1])
                fresh[key] = fresh.get(key, 0) + count
        return fresh

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    @property
    def counter(self) -> dict[int, int]:
        """骨架里 self.counter 的落点：flatten(pair) -> 频率。"""
        return self.pair_freq

    @property
    def vocab_size(self) -> int:
        return len(self.id_word_map)

    @property
    def vocab_full(self) -> bool:
        return self.vocab_size >= self.spec.max_vocabulary_size

    def best_pair(self) -> tuple[int, int] | None:
        """返回当前最高频的 (pair key, 频率)，堆里没有有效项则返回 None。

        只"看一眼"而不消费，所以上层因为频率低于阈值停下时堆状态不被破坏，
        下次续训仍从同一位置继续。

        为什么陈旧项不能简单地 pop 丢弃（这里曾是一个真 bug，被等价性测试抓到）：
        一个 pair 的频率会先降后升。假设 key 的有效条目是 (-10)，被减到 7（不推送），
        再被加到 12（推送 (-12)），随后又减到 9 —— 此刻堆里只有 (-10) 和 (-12)，
        没有一条等于当前值 9。若把陈旧项直接 pop 丢掉，而这个 key 恰好是全局最大，
        它就**永远选不出来了**，训练会在错误的 pair 上合并且完全不报错。
        所以正确做法是"归一化"：把陈旧项替换成该 key 的当前值再放回堆里，
        从而维持一条硬不变式 —— counter 里的每个 key 在堆中至少有一条有效记录。
        """
        heap = self._heap
        pair_freq = self.pair_freq
        while heap:
            neg_freq, key = heap[0]
            current = pair_freq.get(key)
            if current is None:
                # 该 pair 已经彻底消失，条目没有价值，丢掉。
                heapq.heappop(heap)
                continue
            if current == -neg_freq:
                return key, current
            # 条目陈旧但 key 还活着：换成当前值重新入堆。
            # 因为降过，堆里记的值偏高，这条目一定会浮到顶部被处理，不会漏。
            heapq.heappop(heap)
            heapq.heappush(heap, (-current, key))
        return None

    # ------------------------------------------------------------------
    # 合并
    # ------------------------------------------------------------------
    def _flatten(self, id1: int, id2: int) -> int:
        return self.spec.flatten(id1, id2)

    def _rebuild_heap(self) -> None:
        """整堆重建：把陈旧项一次性清掉。

        惰性策略下堆只增不减，长跑会变成纯粹的内存泄漏；超过倍数上限就重建。
        重建是 O(P)，但只有堆膨胀到 pair 总数的数倍后才发生，均摊是常数级。
        """
        self._heap = [(-f, k) for k, f in self.pair_freq.items()]
        heapq.heapify(self._heap)

    def step(self) -> MergeResult | None:
        """执行一次合并：取最高频 pair，合成新 token，增量修正统计。

        返回 None 表示"无法继续"（已达词表上限 / 无有效 pair / 最高频低于阈值），
        用返回值而不是异常表达终止，方便主循环写成 `while (r = engine.step())`。

        实现要点（都是被 profile 逼出来的，见 _bench 的记录）：
        1. **按 key 聚合再落账**。朴素写法对"每个词的每个 pair  occurrence"立刻改
           dict 并 heappush，一次合并要推几万次堆；而其中绝大多数是同一个 key 被
           不同词反复贡献。改成先累加 freq_delta，本步结束后每个 key 只结算一次、
           只推一次堆。合并速率提升一个数量级。
        2. **flatten 内联**。`self.spec.flatten(a, b)` 在热循环里是 8800 万次方法调用
           加一次 property 查找；把基数 V 取到局部变量后写成 `prev * V + cur`。
        3. **只在净增时推堆**。频率下降的陈旧项由 best_pair 归一化（见那里），
           不需要推送；只有净增的 key 才可能越过当前堆顶，才必须推送。
        """
        if self.vocab_full:
            return None
        best = self.best_pair()
        if best is None:
            return None
        key, freq = best
        if freq < self.min_frequency:
            return None

        left, right = self.spec.unflatten(key)
        new_id = self.next_id
        # 新 token 的字面形式 = 左右 token 字面拼接。这是字节级词表的定义：
        # 词表里每一项都是"一串字节"，而不是"一串符号引用"，所以能直接导出 GPT-2 格式。
        new_token = self.id_word_map[left] + self.id_word_map[right]

        # --- 先登记模型产物，再改统计：中途异常时宁可少一条规则，
        #     也不能出现"有 token 却没有 rank"的半状态。
        heapq.heappop(self._heap)
        self.merges.append((left, right))
        self.merge_rank[key] = len(self.merges) - 1
        self.id_word_map[new_id] = new_token
        self.word_id_map[new_token] = new_id
        self.next_id += 1

        affected = self.pair_words.pop(key, ())
        # 注意：不能提前 del self.pair_freq[key]。pair_words[key] 恰好是所有贡献该频率的
        # 词，聚合后 (a,b) 的净变化正好是 -freq，会被自然结算成 0；提前删除会让后面 KeyError。
        V = self.spec.flatten_base
        pair_freq = self.pair_freq
        pair_words = self.pair_words
        word_freq = self.word_freq
        merge_occ = merge_occurrences

        freq_delta: dict[int, int] = {}
        rewritten: dict[Word, int] = {}

        for word in affected:
            count = word_freq.pop(word)
            new_word = merge_occ(word, left, right, new_id)

            # 旧词的相邻 pair 全部减掉（含重数：(x,x,x) 里 (x,x) 有两次）
            previous = word[0]
            for current in word[1:]:
                old_key = previous * V + current
                previous = current
                bucket = pair_words.get(old_key)
                if bucket is not None:
                    bucket.discard(word)
                freq_delta[old_key] = freq_delta.get(old_key, 0) - count

            # 新词的相邻 pair 全部加上。必须等旧 pair 处理完：同一个 key 若既是旧 pair
            # 又是新 pair（离合并点很远的相邻关系不变），先减后加正好抵消。
            if len(new_word) > 1:
                previous = new_word[0]
                for current in new_word[1:]:
                    new_key = previous * V + current
                    previous = current
                    bucket = pair_words.get(new_key)
                    if bucket is None:
                        pair_words[new_key] = {new_word}
                    else:
                        bucket.add(new_word)
                    freq_delta[new_key] = freq_delta.get(new_key, 0) + count

            rewritten[new_word] = rewritten.get(new_word, 0) + count

        word_freq.update(rewritten)

        # --- 每个 key 只结算一次 ---
        heap = self._heap
        push = heapq.heappush
        for pair_key, delta in freq_delta.items():
            value = pair_freq.get(pair_key, 0) + delta
            if value > 0:
                pair_freq[pair_key] = value
                if delta > 0:
                    push(heap, (-value, pair_key))
            else:
                # 频率归零 <=> 再也没有词含该 pair，连倒排桶一起回收
                pair_freq.pop(pair_key, None)
                pair_words.pop(pair_key, None)

        # 惰性堆的膨胀闸门：陈旧项只增不减，长跑会变成纯内存泄漏。
        # 重建是 O(P)，但只在堆超出 pair 总数若干倍后才发生，均摊是常数级。
        if len(heap) > self._heap_growth_factor * len(pair_freq):
            self._rebuild_heap()

        return MergeResult(
            pair=(left, right),
            new_id=new_id,
            new_token=new_token,
            frequency=freq,
            rank=len(self.merges) - 1,
            affected_words=len(affected),
        )

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    def merge_seq(self, ids: Word) -> Word:
        """对一个预切分片段应用全部合并规则（按 rank 从小到大）。

        与训练语义严格一致的两点：
          1. 每一轮只应用"当前 rank 最小"的那一条规则；
          2. 一次应用会把该 pair 在词内出现的**所有**位置同时合并（从左到右、不重叠）。
        只要两边语义一致，训练出的词表和推理切分就必然可复现 —— 这类"训练/推理
        不一致"是 BPE 实现里最常见的静默 bug。
        """
        if not self.merge_rank or len(ids) < 2:
            return ids
        current = ids
        while len(current) >= 2:
            best_key: int | None = None
            best_rank = 0
            for i in range(len(current) - 1):
                key = self._flatten(current[i], current[i + 1])
                rank = self.merge_rank.get(key)
                if rank is not None and (best_key is None or rank < best_rank):
                    best_key, best_rank = key, rank
            if best_key is None:
                break
            left, right = self.spec.unflatten(best_key)
            # 不变式：合并 token 的 id = first_merge_id + rank（编号与规则同步递增）。
            new_id = self.spec.first_merge_id + best_rank
            current = merge_occurrences(current, left, right, new_id)
        return current

    # ------------------------------------------------------------------
    # 断点续训的导入导出
    # ------------------------------------------------------------------
    def load_state(
        self,
        *,
        id_word_map: dict[int, bytes],
        merges: list[tuple[int, int]],
        word_freq: dict[Word, int],
        counter_to_verify: dict[int, int] | None = None,
    ) -> None:
        """从 param_store 读到的产物整体替换当前状态。

        刻意要求"一次性全部传入"而不是逐项 setter：逐项设置就是骨架担心的
        "counter/merges/词表进度不一致"的来源。要么全套换上，要么保持原样。

        counter 一律由 word_freq **重新导出**，落盘的 counter.npz 只用来做一致性校验。
        原因是 pair->词 的索引无法从 counter 还原（见模块 docstring 的反例），
        既然必须重算，再信任文件里的 counter 就是引入第二个真值来源。
        """
        self.id_word_map = dict(id_word_map)
        self.word_id_map = {token: i for i, token in self.id_word_map.items()}
        self.merges = list(merges)
        self.merge_rank = {self._flatten(a, b): r for r, (a, b) in enumerate(self.merges)}
        self.word_freq = dict(word_freq)
        # 合并 token 连续编号：续训时下一个新 id 直接由规则条数推出，
        # 不依赖文件里存的历史 next_id，少一个可能被写坏的字段。
        self.next_id = self.spec.first_merge_id + len(self.merges)
        self.build_index()
        if counter_to_verify is not None and counter_to_verify != self.pair_freq:
            raise AssertionError(
                "落盘 counter 与由词频表重算的结果不一致："
                "checkpoint 生成时三份产物就不是同一进度，拒绝续训"
            )

    def consistency_check(self) -> None:
        """断言三份结构互相自洽。续训加载后与长跑期间周期性调用。"""
        rebuilt = self.recompute_pair_freq()
        if rebuilt != self.pair_freq:
            missing = len(set(rebuilt) - set(self.pair_freq))
            extra = len(set(self.pair_freq) - set(rebuilt))
            raise AssertionError(
                f"增量 counter 与全量重算不一致（缺失 {missing} 项 / 多余 {extra} 项）"
            )
        # 每个有频率的 pair 必须有非空的词集合，否则下一轮合并会漏改词。
        for key in self.pair_freq:
            bucket = self.pair_words.get(key)
            if not bucket:
                raise AssertionError(f"pair {self.spec.unflatten(key)} 有频率但没有词索引")
        if self.vocab_size != len(self.merges) + self.spec.base_vocabulary_size:
            raise AssertionError("词表规模与合并规则条数不匹配")


def merge_occurrences(sequence: Word, left: int, right: int, new_id: int) -> Word:
    """把 sequence 里所有 (left, right) 相邻出现替换成 new_id（从左到右，不重叠）。

    "不重叠"必须和 GPT-2 的 replace 行为一致，否则 (x,x,x) 在训练里合成 1 个新 token、
    在推理里合成 2 个，词表直接不可用。
    """
    out: list[int] = []
    i = 0
    n = len(sequence)
    while i < n:
        if i + 1 < n and sequence[i] == left and sequence[i + 1] == right:
            out.append(new_id)
            i += 2
        else:
            out.append(sequence[i])
            i += 1
    return tuple(out)
