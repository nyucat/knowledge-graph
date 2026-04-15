# 维基百科知识图谱课程作业（实体识别重点）

本项目实现了一个可运行的最小流程：

1. 读取本地 `datafile` 目录下全部 `txt` 文本
2. 输入 0 到多个网页链接并抓取正文文本
3. 做实体识别（`CRF` 为重点，`CogIE` 为扩展）
4. 使用细粒度实体分类（mention+上下文）输出层级标签
5. 输出实体结果到 `JSON` 和 `CSV`

说明：当前实体候选由 `CRF/CogIE` 与 `jieba` 分词联合产生，不使用阈值过滤。

---

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

若要使用 CogIE（可选）：

```bash
pip install git+https://github.com/jinzhuoran/CogIE.git
```

---

## 2. CRF 实体识别（重点）

CRF 使用字符级 BIO 标注训练，训练样例在：

- `data/sample_ner_train.jsonl`

运行完整流程（默认 CRF，输出覆盖到同一个 `outputs` 目录）：

```bash
python src/run_pipeline.py --ner-backend crf
```

直接通过参数给链接（可多个）：

```bash
python src/run_pipeline.py --urls https://zh.wikipedia.org/wiki/图灵 https://zh.wikipedia.org/wiki/人工智能 --ner-backend crf
```

说明：

- 程序启动会先检查 `datafile` 是否有 `txt` 文档；
- 若没有，会提示：`无本地数据`；
- 随后可输入 0 到多个链接（可直接回车跳过）；
- 程序会把所有链接抓取到的正文写入 `webfile/web.txt`；
- 然后对 `web.txt` 中文本做实体抽取，并与本地 `txt` 的抽取结果合并输出。

输出文件：

- `outputs/entities.json`：实体抽取结果（JSON）
- `outputs/entities.csv`：实体抽取结果（CSV）
- `outputs/crf_model.bin`：训练好的 CRF 模型

---

## 3. CogIE 实体识别（扩展）

切换实体识别后端为 CogIE：

```bash
python src/run_pipeline.py --urls https://en.wikipedia.org/wiki/Artificial_intelligence --ner-backend cogie
```

说明：

- `CogIE` 不同版本可能存在接口差异；
- 当前适配器位于 `src/cogie_adapter.py`，使用 `TokenizeToolkit + NerToolkit`。

---

## 4. 项目结构

- `src/text_sources.py`：本地 txt + 网页链接文本加载
- `src/crf_ner.py`：CRF 训练与预测（实体识别核心）
- `src/fine_grained_typing.py`：细粒度实体分类与层级标签
- `src/cogie_adapter.py`：CogIE NER 适配器
- `src/run_pipeline.py`：一键流程脚本

---

