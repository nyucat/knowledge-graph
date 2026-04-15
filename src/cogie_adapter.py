from __future__ import annotations

from typing import Dict, List


class CogIEEntityRecognizer:
    """
    CogIE 适配器（最小可用版）。
    说明：
    - 不同版本的 CogIE 在模型名称与参数上可能存在差异。
    - 当前实现采用 Toolkit 风格 API：TokenizeToolkit + NerToolkit。
    """

    def __init__(self, language: str = "english", corpus: str = "trex") -> None:
        self.language = language
        self.corpus = corpus
        self._tokenize_tool = None
        self._ner_tool = None
        self._init_toolkits()

    def _init_toolkits(self) -> None:
        try:
            import cogie
        except ImportError as exc:
            raise RuntimeError(
                "未安装 CogIE，请先执行: pip install git+https://github.com/jinzhuoran/CogIE.git"
            ) from exc

        if not hasattr(cogie, "TokenizeToolkit") or not hasattr(cogie, "NerToolkit"):
            raise RuntimeError("当前 CogIE 版本缺少 TokenizeToolkit/NerToolkit，请检查版本。")

        self._tokenize_tool = cogie.TokenizeToolkit(
            task="ws",
            language=self.language,
            corpus=None,
        )
        self._ner_tool = cogie.NerToolkit(
            task="ner",
            language=self.language,
            corpus=self.corpus,
        )

    def predict(self, text: str) -> List[Dict[str, str]]:
        if not text.strip():
            return []
        words = self._tokenize_tool.run(text)
        ner_result = self._ner_tool.run(words)

        entities = []
        for item in ner_result:
            if isinstance(item, dict):
                entity = {
                    "text": str(item.get("entity", item.get("text", ""))),
                    "label": str(item.get("type", item.get("label", "UNKNOWN"))),
                }
            else:
                entity = {"text": str(item), "label": "UNKNOWN"}
            if entity["text"]:
                entities.append(entity)
        return entities
