BBPE.py                    编排层 + CLI（smoke/train/estimate/tokenize/inspect/scale）  
bbpe/config.py             id 空间与文件布局的唯一真值（flatten 基数、uint16 边界校验）  
bbpe/pretokenize.py        GPT-2 正则 + 字节映射 + vocab.json 的字节置换（与 GPT-2 逐位一致）  
bbpe/text_clean.py         清洗（格式字符/重复行/符号长串/空白归一），阈值可配  
bbpe/corpus_stream.py      抽样 → 字节上限缓冲 → 清洗 → 切分 → 字节化 → 词频  
bbpe/merge_engine.py       精确增量 BPE  
bbpe/param_store.py        提交点式原子落盘 + sha256 校验  
bbpe/devdata.py            确定性假语料（Zipf 长尾 + 中英 + 爬虫噪声）  
tests/                     5 个文件，181 项，零网络  
pyproject.toml + uv.lock   uv 管理，datasets/huggingface_hub 为 train extra