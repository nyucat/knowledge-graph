from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class TypingResult:
    fine_label: str
    parent_label: str
    score: float


class FineGrainedEntityTyper:
    """
    轻量级细粒度实体分类器（无额外深度学习依赖）。

    设计思想对应“mention + 左右上下文 + 拼接决策”：
    - mention 通道：实体字符串本身的关键词证据
    - context 通道：实体左右上下文中的提示词
    - decision 通道：将三类分数拼接后选最大类别
    """

    def __init__(self) -> None:
        self.label_parent: Dict[str, str] = {
            "PER.Politician": "PER",
            "PER.Scientist": "PER",
            "PER.Artist": "PER",
            "PER.Athlete": "PER",
            "ORG.Company": "ORG",
            "ORG.University": "ORG",
            "ORG.Government": "ORG",
            "ORG.ResearchInstitute": "ORG",
            "ORG.Media": "ORG",
            "LOC.Country": "LOC",
            "LOC.City": "LOC",
            "LOC.Region": "LOC",
            "EVENT.SportsEvent": "EVENT",
            "EVENT.Conference": "EVENT",
            "EVENT.HistoricalEvent": "EVENT",
            "WORK.Book": "WORK",
            "WORK.MovieOrTV": "WORK",
            "WORK.Song": "WORK",
            "PROD.SoftwareOrModel": "PROD",
            "PROD.HardwareOrDevice": "PROD",
            "THEORY.LawOrTheorem": "THEORY",
            "THEORY.Hypothesis": "THEORY",
            "SUBJECT.ComputerScience": "SUBJECT",
            "SUBJECT.Math": "SUBJECT",
            "SUBJECT.Physics": "SUBJECT",
            "SUBJECT.Chemistry": "SUBJECT",
            "SUBJECT.Biology": "SUBJECT",
            "SUBJECT.History": "SUBJECT",
            "SUBJECT.Philosophy": "SUBJECT",
            "MONEY.Currency": "MONEY",
            "PERCENT.Ratio": "PERCENT",
            "DATE.CalendarDate": "DATE",
            "TIME.ClockTime": "TIME",
            "FIELD.ResearchField": "FIELD",
            "TECH.Technology": "TECH",
        }

        self.mention_keywords: Dict[str, List[str]] = {
            "ORG.University": ["大学", "学院", "university", "college"],
            "ORG.Company": ["公司", "集团", "有限公司", "corp", "inc", "ltd"],
            "ORG.ResearchInstitute": ["研究院", "研究所", "科学院", "实验室"],
            "ORG.Government": ["政府", "部", "委员会", "法院", "议会"],
            "ORG.Media": ["电视台", "报社", "新闻网", "杂志"],
            "LOC.Country": ["中国", "美国", "英国", "法国", "日本", "德国"],
            "LOC.City": ["市", "北京", "上海", "深圳", "广州", "杭州"],
            "EVENT.SportsEvent": ["奥运会", "世界杯", "锦标赛", "联赛"],
            "EVENT.Conference": ["会议", "峰会", "论坛", "大会"],
            "WORK.Book": ["书", "小说", "文集", "诗集"],
            "WORK.MovieOrTV": ["电影", "电视剧", "纪录片"],
            "WORK.Song": ["歌曲", "专辑"],
            "PROD.SoftwareOrModel": ["模型", "系统", "软件", "gpt", "app"],
            "PROD.HardwareOrDevice": ["手机", "芯片", "显卡", "设备"],
            "THEORY.LawOrTheorem": ["定律", "定理", "原理"],
            "THEORY.Hypothesis": ["假说", "假设"],
            "SUBJECT.ComputerScience": ["计算机科学", "计算机", "人工智能", "机器学习"],
            "SUBJECT.Math": ["数学", "代数", "几何", "微积分"],
            "SUBJECT.Physics": ["物理", "力学", "电磁学", "量子"],
            "SUBJECT.Chemistry": ["化学", "有机", "无机"],
            "SUBJECT.Biology": ["生物", "遗传", "细胞"],
            "SUBJECT.History": ["历史", "史学"],
            "SUBJECT.Philosophy": ["哲学", "伦理学"],
            "MONEY.Currency": ["元", "美元", "欧元", "人民币", "¥", "$"],
            "PERCENT.Ratio": ["%", "％", "百分之"],
            "DATE.CalendarDate": ["年", "月", "日", "号"],
            "TIME.ClockTime": ["点", "时", "分", "秒", "上午", "下午"],
            "FIELD.ResearchField": ["领域", "方向", "分支", "学科"],
            "TECH.Technology": ["技术", "算法", "框架", "工程"],
            "PER.Politician": ["总统", "总理", "议员", "书记", "主席"],
            "PER.Scientist": ["教授", "学者", "科学家", "研究员"],
            "PER.Artist": ["歌手", "演员", "导演", "作家"],
            "PER.Athlete": ["运动员", "球员", "冠军"],
        }

        self.context_keywords: Dict[str, List[str]] = {
            "PER.Politician": ["当选", "执政", "访问", "政府"],
            "PER.Scientist": ["提出", "研究", "论文", "实验"],
            "PER.Artist": ["出演", "发行", "创作", "表演"],
            "PER.Athlete": ["比赛", "夺冠", "赛季", "奥运"],
            "ORG.Company": ["发布", "融资", "营收", "总部"],
            "ORG.University": ["录取", "学院", "专业", "本科"],
            "ORG.ResearchInstitute": ["课题", "实验室", "研究成果"],
            "EVENT.SportsEvent": ["开幕", "闭幕", "参赛", "奖牌"],
            "SUBJECT.ComputerScience": ["计算", "编程", "模型", "神经网络"],
            "MONEY.Currency": ["价格", "成本", "预算", "支付"],
            "PERCENT.Ratio": ["增长", "下降", "占比", "比例"],
            "DATE.CalendarDate": ["发布于", "日期", "截止", "时间为"],
            "TIME.ClockTime": ["开始于", "发生在", "定于"],
        }

    def _score_keywords(self, text: str, keyword_map: Dict[str, List[str]]) -> Dict[str, float]:
        lowered = text.lower()
        scores = {label: 0.0 for label in self.label_parent}
        for label, words in keyword_map.items():
            for w in words:
                if w.lower() in lowered:
                    scores[label] += 1.0
        return scores

    def _coarse_prior(self, coarse_label: str) -> Dict[str, float]:
        prior = {label: 0.0 for label in self.label_parent}
        for label, parent in self.label_parent.items():
            if parent == coarse_label:
                prior[label] = 1.0
        return prior

    def predict(
        self,
        entity_text: str,
        page_text: str,
        start: int,
        end: int,
        coarse_label: str,
    ) -> TypingResult:
        left = page_text[max(0, start - 30) : start] if start >= 0 else ""
        right = page_text[end : min(len(page_text), end + 30)] if end >= 0 else ""
        context = f"{left} {right}".strip()

        mention_scores = self._score_keywords(entity_text, self.mention_keywords)
        context_scores = self._score_keywords(context, self.context_keywords)
        prior_scores = self._coarse_prior(coarse_label)

        best_label = ""
        best_score = -1e9
        for label in self.label_parent:
            score = (
                0.60 * mention_scores[label]
                + 0.30 * context_scores[label]
                + 0.10 * prior_scores[label]
            )
            if score > best_score:
                best_score = score
                best_label = label

        if best_score <= 0:
            # 若无明显细粒度信号，则退化为“父类.通用”
            generic = {
                "PER": "PER.Scientist",
                "ORG": "ORG.Company",
                "LOC": "LOC.City",
                "TIME": "TIME.ClockTime",
                "DATE": "DATE.CalendarDate",
                "MONEY": "MONEY.Currency",
                "PERCENT": "PERCENT.Ratio",
                "SUBJECT": "SUBJECT.ComputerScience",
                "EVENT": "EVENT.Conference",
                "WORK": "WORK.Book",
                "PROD": "PROD.SoftwareOrModel",
                "THEORY": "THEORY.LawOrTheorem",
                "FIELD": "FIELD.ResearchField",
                "TECH": "TECH.Technology",
            }
            best_label = generic.get(coarse_label, "ORG.Company")
            best_score = 0.0

        return TypingResult(
            fine_label=best_label,
            parent_label=self.label_parent[best_label],
            score=best_score,
        )
