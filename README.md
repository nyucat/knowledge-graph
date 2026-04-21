# 缁村熀鐧剧鐭ヨ瘑鍥捐氨璇剧▼浣滀笟锛堝疄浣撹瘑鍒噸鐐癸級

鏈」鐩疄鐜颁簡涓€涓彲杩愯鐨勬渶灏忔祦绋嬶細

1. 璇诲彇鏈湴 `datafile` 鐩綍涓嬪叏閮?`txt` 鏂囨湰
2. 杈撳叆 0 鍒板涓綉椤甸摼鎺ュ苟鎶撳彇姝ｆ枃鏂囨湰
3. 鍋氬疄浣撹瘑鍒紙`CRF` 涓洪噸鐐癸紝`CogIE` 涓烘墿灞曪級
4. 浣跨敤缁嗙矑搴﹀疄浣撳垎绫伙紙mention+涓婁笅鏂囷級杈撳嚭灞傜骇鏍囩
5. 杈撳嚭瀹炰綋缁撴灉鍒?`JSON` 鍜?`CSV`

璇存槑锛氬綋鍓嶅疄浣撳€欓€夌敱 `CRF/CogIE` 涓?`jieba` 鍒嗚瘝鑱斿悎浜х敓锛屼笉浣跨敤闃堝€艰繃婊ゃ€?

---

## 1. 瀹夎渚濊禆

```bash
pip install -r requirements.txt
```

鑻ヨ浣跨敤 CogIE锛堝彲閫夛級锛?

```bash
pip install git+https://github.com/jinzhuoran/CogIE.git
```

---

## 2. CRF 瀹炰綋璇嗗埆锛堥噸鐐癸級

CRF 浣跨敤瀛楃绾?BIO 鏍囨敞璁粌锛岃缁冩牱渚嬪湪锛?

- `data/sample_ner_train.jsonl`

杩愯瀹屾暣娴佺▼锛堥粯璁?CRF锛岃緭鍑鸿鐩栧埌鍚屼竴涓?`outputs` 鐩綍锛夛細

```bash
python src/run_pipeline.py --ner-backend crf
```

鐩存帴閫氳繃鍙傛暟缁欓摼鎺ワ紙鍙涓級锛?

```bash
python src/run_pipeline.py --urls https://zh.wikipedia.org/wiki/鍥剧伒 https://zh.wikipedia.org/wiki/浜哄伐鏅鸿兘 --ner-backend crf
```

璇存槑锛?

- 绋嬪簭鍚姩浼氬厛妫€鏌?`datafile` 鏄惁鏈?`txt` 鏂囨。锛?
- 鑻ユ病鏈夛紝浼氭彁绀猴細`鏃犳湰鍦版暟鎹甡锛?
- 闅忓悗鍙緭鍏?0 鍒板涓摼鎺ワ紙鍙洿鎺ュ洖杞﹁烦杩囷級锛?
- 绋嬪簭浼氭妸鎵€鏈夐摼鎺ユ姄鍙栧埌鐨勬鏂囧啓鍏?`webfile/web.txt`锛?
- 鐒跺悗瀵?`web.txt` 涓枃鏈仛瀹炰綋鎶藉彇锛屽苟涓庢湰鍦?`txt` 鐨勬娊鍙栫粨鏋滃悎骞惰緭鍑恒€?

杈撳嚭鏂囦欢锛?

- `outputs/entities.json`锛氬疄浣撴娊鍙栫粨鏋滐紙JSON锛?
- `outputs/entities.csv`锛氬疄浣撴娊鍙栫粨鏋滐紙CSV锛?
- `outputs/crf_model.bin`锛氳缁冨ソ鐨?CRF 妯″瀷

---

## 3. CogIE 瀹炰綋璇嗗埆锛堟墿灞曪級

鍒囨崲瀹炰綋璇嗗埆鍚庣涓?CogIE锛?

```bash
python src/run_pipeline.py --urls https://en.wikipedia.org/wiki/Artificial_intelligence --ner-backend cogie
```

璇存槑锛?

- `CogIE` 涓嶅悓鐗堟湰鍙兘瀛樺湪鎺ュ彛宸紓锛?
- 褰撳墠閫傞厤鍣ㄤ綅浜?`src/cogie_adapter.py`锛屼娇鐢?`TokenizeToolkit + NerToolkit`銆?

---

## 4. 椤圭洰缁撴瀯

- `src/text_sources.py`锛氭湰鍦?txt + 缃戦〉閾炬帴鏂囨湰鍔犺浇
- `src/crf_ner.py`锛欳RF 璁粌涓庨娴嬶紙瀹炰綋璇嗗埆鏍稿績锛?
- `src/fine_grained_typing.py`锛氱粏绮掑害瀹炰綋鍒嗙被涓庡眰绾ф爣绛?
- `src/cogie_adapter.py`锛欳ogIE NER 閫傞厤鍣?
- `src/run_pipeline.py`锛氫竴閿祦绋嬭剼鏈?

---


## Bert-BiLSTM-CRF 瀹炰綋鎶藉彇涓庢秷姝э紙鏂板锛?
瀹夎鏂板渚濊禆锛?
```bash
pip install -r requirements.txt
```

榛樿宸茬粡鏀逛负 `bert_bilstm_crf` 鍚庣锛岀洿鎺ヨ繍琛岋細

```bash
python src/run_pipeline.py --data-dir datafile --ner-backend bert_bilstm_crf --disambiguation-backend bert_bilstm_crf
```

鍙拡瀵?`datafile/test.txt` 鎶藉彇锛?
```bash
python src/run_pipeline.py --data-dir datafile --ner-backend bert_bilstm_crf --disambiguation-backend bert_bilstm_crf --disable-disambiguation
```

璇存槑锛?- `--bert-model-name` 鍙浛鎹负鏈湴宸蹭笅杞芥ā鍨嬬洰褰曪紙绂荤嚎鐜鎺ㄨ崘锛夈€?- `--reuse-crf-model` 浼氫紭鍏堝姞杞?`outputs/` 涓嬪凡鏈?BERT/CRF 妯″瀷銆?- 杈撳嚭浠嶅湪 `outputs/entities.json` 涓?`outputs/entities.csv`銆?

### 镜像下载 BERT 模型（清华优先）

```bash
.\\.venv\\Scripts\\python.exe src/download_bert_model.py --repo-id bert-base-chinese --output-dir models/bert-base-chinese
```

运行主流程时指定本地模型目录：

```bash
.\\.venv\\Scripts\\python.exe src/run_pipeline.py --data-dir datafile --ner-backend bert_bilstm_crf --disambiguation-backend bert_bilstm_crf --bert-model-name models/bert-base-chinese
```

## Turing 定向训练与抽取

```bash
.\\.venv\\Scripts\\python.exe src/train_and_extract_turing.py
```

输出文件：
- `outputs/turing_bert_bilstm_crf.pt`（训练后的模型）
- `outputs/turing_entities.json`（分类结果 + 按原文顺序结果）

词典文件：
- `data/turing_lexicon.json`

说明：
- 采用 `BERT-BiLSTM-CRF + 词典强召回`，优先保证你给的目标实体可被检出。
- 如需进一步贴合你给的完整超长清单，可继续扩充 `data/turing_lexicon.json`。
