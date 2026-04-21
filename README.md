# knowledge_graph

本项目用于从图灵文本中进行：
- 实体抽取（NER）
- 实体消歧
- 关系抽取
- 知识图谱三元组构建
- Neo4j Cypher 导出

## 环境

建议使用项目内虚拟环境：

```bash
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 主要输入输出

- 输入文本：`datafile/test.txt`
- 消歧实体：`outputs/turing_entities_disambiguated.json`
- 关系输出：`outputs/turing_relations_only.json`
- 三元组输出：`outputs/turing_kg_triples.json`
- Neo4j 导入脚本：`outputs/CY.txt`

## 运行流程

### 1) 图灵关系抽取（LSTM-CRF + 配置规则）

```bash
.\.venv\Scripts\python.exe src/lstm_crf_relation_extract_turing.py
```

说明：
- 关系规则配置在 `data/relation_rules_turing.json`
- 该脚本会输出 `turing_relations_only.json` 和 `turing_kg_triples.json`

### 2) 生成 Neo4j Cypher

```bash
.\.venv\Scripts\python.exe src/generate_neo4j_cypher.py --output outputs/CY.txt
```

生成格式：
- `CREATE` 批量建节点
- `MATCH` 绑定节点变量
- `CREATE` 批量建关系

## 可调配置

关系规则位于：
- `data/relation_rules_turing.json`

可调项包括：
- 触发词 `triggers`
- 头尾实体类型约束 `type_in`
- 配对策略 `pairing`
- 最大配对数 `max_pairs`
- 正则模板规则（用于补充高覆盖关系）

## 备注

- `turing_lexicon.json` 与 `turing_disambiguation.json` 按需求可保持不改动。
- `.gitignore` 已忽略 `*.pt` 和 `models/`。
