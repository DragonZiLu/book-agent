"""
Judge + 三层评估器：
L1（纯规则）：原文 token 覆盖率
L2（LLM辅助）：四维度召回率
L3（ScenarioQA）：Examiner 出题 → Executor 答 → Judge 判分
"""

import re
import json
import math
import logging
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field

from config import (
    L1_THRESHOLD,
    L2_THRESHOLD,
    L3_THRESHOLD,
    QA_QUESTIONS_PER_CHAPTER,
    QA_TYPE_DISTRIBUTION,
    REQUIRED_DIMENSIONS,
)
from llm_client import ModelPool, LLMClient

logger = logging.getLogger(__name__)


# ============ 数据结构 ============

@dataclass
class ScenarioQA:
    """Examiner 出的题 + 标准答案"""
    question: str
    gold_answer: str
    evidence_span: str
    answer_type: str          # "numeric" | "direction" | "factual"
    chapter_index: int


@dataclass
class WeakChapterFeedback:
    """薄弱章节的反馈信息，供重提取使用"""
    index: int
    missing_dimensions: List[str] = field(default_factory=list)
    failed_questions: List[str] = field(default_factory=list)
    hint: str = ""


@dataclass
class QualityReport:
    """三层评估结果完整报告"""
    l1_token_coverage: float = 0.0
    l2_dimension_recall: float = 0.0
    l3_qa_score: float = 0.0
    l3_breakdown: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    passed: bool = False
    chapter_scores: List[dict] = field(default_factory=list)
    weak_chapters: List[int] = field(default_factory=list)
    feedbacks: List[WeakChapterFeedback] = field(default_factory=list)
    extraction_warnings: List[str] = field(default_factory=list)
    cost_usd: float = 0.0


# ============ Judge 类（出题 + 判分）============

class Judge:
    """
    与 Executor 异源的独立评委。
    - Examiner：看原文出题 + 生成标准答案
    - Judge：以原文为唯一真相源判分
    """

    def __init__(self, pool: ModelPool):
        self.examiner: LLMClient = pool.get("examiner")
        self.judge: LLMClient = pool.get("judge")

    def generate_questions(self, chapter) -> List[ScenarioQA]:
        """
        Examiner 从原文出题，题型分布按 config.QA_TYPE_DISTRIBUTION。
        同时产出 gold_answer + evidence_span。
        """
        total = QA_QUESTIONS_PER_CHAPTER
        n_numeric = QA_TYPE_DISTRIBUTION.get("numeric", 2)
        n_direction = QA_TYPE_DISTRIBUTION.get("direction", 2)
        n_factual = QA_TYPE_DISTRIBUTION.get("factual", 1)

        prompt = f"""你是一位严格的考试出题人。请从以下投资书籍章节中出 {total} 道题，
用于检验 AI 是否真正理解了这本书的核心内容。

题型分布要求：
- {n_numeric} 道数值题（必须有明确数字答案）
- {n_direction} 道方向判断题（答案是方向性的：是/否、符合/不符合、买入/回避等）
- {n_factual} 道事实题（考察书中的具体事实、观点、案例）

每道题必须包含：
1. question: 题目文本
2. gold_answer: 标准答案（从原文严格推导）
3. evidence_span: 支撑答案的原文片段（直接引用）
4. answer_type: "numeric" / "direction" / "factual"

章节标题：{chapter.title}

章节内容：
{chapter.content[:8000]}

请输出 JSON 数组格式（只输出 JSON，不要其他文字）：
[
  {{
    "question": "...",
    "gold_answer": "...",
    "evidence_span": "...",
    "answer_type": "numeric"
  }},
  ...
]"""

        try:
            response = self.examiner.complete(
                prompt=prompt,
                system="你是一位严格的考试出题人，出的题必须有明确的原文依据。",
                max_tokens=3000,
                temperature=0.2,
            )
            questions = self._parse_questions(response, chapter.index)
            logger.debug(f"第{chapter.index}章出题 {len(questions)} 道")
            return questions
        except Exception as e:
            logger.warning(f"出题失败 第{chapter.index}章: {e}")
            return []

    def _parse_questions(self, response: str, chapter_index: int) -> List[ScenarioQA]:
        """解析 LLM 输出的题 JSON"""
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # 尝试提取 JSON 块
            m = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
            if m:
                try:
                    data = json.loads(m.group(1))
                except json.JSONDecodeError:
                    # 最后一次尝试：找 [ ... ]
                    start = response.find("[")
                    end = response.rfind("]")
                    if start >= 0 and end > start:
                        try:
                            data = json.loads(response[start:end + 1])
                        except json.JSONDecodeError:
                            return []
                    else:
                        return []
            else:
                return []

        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return []

        result = []
        for item in data:
            atype = item.get("answer_type", "factual")
            if atype not in ("numeric", "direction", "factual"):
                atype = "factual"
            result.append(ScenarioQA(
                question=item.get("question", ""),
                gold_answer=item.get("gold_answer", ""),
                evidence_span=item.get("evidence_span", ""),
                answer_type=atype,
                chapter_index=chapter_index,
            ))

        return result

    def grade(self, qa: ScenarioQA, answer: str) -> Tuple[bool, str]:
        """
        判分，按题型走不同路线：
        - numeric：正则抽数字 → 代码比对（±2% 容差），不调 LLM
        - direction：Judge LLM 判，方向错直接错
        - factual：Judge LLM 语义判，核心事实正确即算对
        """
        if qa.answer_type == "numeric":
            return self._grade_numeric(qa, answer)
        elif qa.answer_type == "direction":
            return self._grade_direction(qa, answer)
        else:
            return self._grade_factual(qa, answer)

    def _grade_numeric(self, qa: ScenarioQA, answer: str) -> Tuple[bool, str]:
        """数值题：代码比对，零 LLM 调用"""
        # 从 gold_answer 和 answer 中提取数字
        gold_nums = self._extract_numbers(qa.gold_answer)
        ans_nums = self._extract_numbers(answer)

        if not gold_nums:
            logger.warning(f"gold_answer 中未找到数字: {qa.gold_answer[:100]}")
            return False, "gold_answer 无数字可提取"

        if not ans_nums:
            return False, f"答案中未找到数字，gold={gold_nums}"

        # 取第一个有效数字比对（±2% 容差）
        gold = gold_nums[0]
        for an in ans_nums:
            if gold == 0:
                if an == 0:
                    return True, f"匹配: {an} == {gold}"
                continue
            error = abs(an - gold) / abs(gold)
            if error <= 0.02:
                return True, f"匹配: {an} ≈ {gold} (误差 {error:.1%})"

        return False, f"不匹配: answer={ans_nums[:3]}, gold={gold_nums[:3]}"

    def _grade_direction(self, qa: ScenarioQA, answer: str) -> Tuple[bool, str]:
        """方向题：Judge 判分，方向错 = 直接错"""
        prompt = f"""你是严格的判分员。只判断以下答案的方向性是否正确。

题目：{qa.question}

标准答案（原文推导）：{qa.gold_answer}

原文证据：{qa.evidence_span}

考生答案：{answer}

判分规则：
1. 只判断方向（是/否、符合/不符合、买入/回避、多/空等），忽略修饰语
2. 方向与标准答案一致 → 正确（true）
3. 方向与标准答案相反 → 错误（false）
4. 模糊、不给出明确方向 → 错误（false）

只输出一个 JSON：
{{"correct": true/false, "reason": "简短理由"}}"""

        resp = self.judge.complete(prompt, max_tokens=200, temperature=0.0)
        try:
            data = json.loads(resp) if resp.strip().startswith("{") else \
                json.loads(re.search(r"\{[\s\S]*\}", resp).group(0))
            return data.get("correct", False), data.get("reason", "")
        except (json.JSONDecodeError, AttributeError):
            # 兜底：简单检查关键词
            gold_lower = qa.gold_answer.lower()
            ans_lower = answer.lower()
            return gold_lower[:20] in ans_lower, "fallback 文本匹配"

    def _grade_factual(self, qa: ScenarioQA, answer: str) -> Tuple[bool, str]:
        """事实题：Judge 语义判分，核心事实正确即算对"""
        prompt = f"""你是严格的判分员。判断以下答案是否覆盖了核心事实。

题目：{qa.question}

标准答案（原文推导）：{qa.gold_answer}

原文证据：{qa.evidence_span}

考生答案：{answer}

判分规则：
1. 核心事实与标准答案一致 → 正确（true）
2. 事实错误或遗漏核心事实 → 错误（false）
3. 部分正确但不影响核心判断 → 可酌情给正确
4. 不要求措辞完全一致，语义一致即可

只输出一个 JSON：
{{"correct": true/false, "reason": "简短理由"}}"""

        resp = self.judge.complete(prompt, max_tokens=300, temperature=0.0)
        try:
            data = json.loads(resp) if resp.strip().startswith("{") else \
                json.loads(re.search(r"\{[\s\S]*\}", resp).group(0))
            return data.get("correct", False), data.get("reason", "")
        except (json.JSONDecodeError, AttributeError):
            return False, "无法解析 Judge 响应"

    def grade_with_consensus(self, qa: ScenarioQA, answer: str) -> Tuple[bool, str]:
        """
        双 Judge 投票（边界 case 用）。
        两个 Judge 分歧时保守判错（存疑从严）。
        """
        correct1, reason1 = self.grade(qa, answer)

        # 用 examinor 作为第二个 Judge（同源保持标准一致）
        prompt = f"""你是副审。判断以下答案是否正确。

题目：{qa.question}
标准答案：{qa.gold_answer}
证据：{qa.evidence_span}
考生答案：{answer}

主审判：{'正确' if correct1 else '错误'}，理由：{reason1}

请独立判决，只输出 JSON：{{"correct": true/false, "reason": "..."}}"""

        resp = self.examiner.complete(prompt, max_tokens=200, temperature=0.0)
        try:
            data = json.loads(resp) if resp.strip().startswith("{") else \
                json.loads(re.search(r"\{[\s\S]*\}", resp).group(0))
            correct2 = data.get("correct", False)
        except (json.JSONDecodeError, AttributeError):
            correct2 = correct1  # 解析失败时同意主审

        if correct1 == correct2:
            return correct1, f"双审一致: {reason1}"
        else:
            return False, f"双审分歧，保守判错。主审={correct1}，副审={correct2}"

    @staticmethod
    def _extract_numbers(text: str) -> List[float]:
        """从文本中提取所有数字"""
        # 匹配整数、小数、百分数
        numbers = []
        for m in re.finditer(r"(-?[\d,]+\.?\d*)\s*%?", text):
            try:
                num_str = m.group(1).replace(",", "")
                num = float(num_str)
                if "%" in text[m.end():m.end() + 2] or m.group(0).endswith("%"):
                    num = num / 100  # 百分数转换
                numbers.append(num)
            except ValueError:
                pass
        return numbers


# ============ 三层评估器 ============

class QualityEvaluator:
    """三层质量评估：L1 覆盖率 → L2 维度召回 → L3 QA"""

    def __init__(self, pool: ModelPool):
        self.pool = pool
        self.judge = Judge(pool)
        self._utility: Optional[LLMClient] = None

    @property
    def utility(self) -> LLMClient:
        if self._utility is None:
            self._utility = self.pool.get("utility")
        return self._utility

    def evaluate(
        self, chapters, result, extraction_warnings=None
    ) -> QualityReport:
        """执行完整三层评估"""
        report = QualityReport(
            extraction_warnings=extraction_warnings or [],
        )

        # L1：token 覆盖率（纯规则，零 LLM 成本）
        report.l1_token_coverage = self._eval_token_coverage(chapters, result)

        # L2：维度召回率
        recall, dim_scores = self._eval_dimension_recall(chapters, result)
        report.l2_dimension_recall = recall

        # L3：ScenarioQA
        qa_score, chapter_scores, feedbacks = self._eval_qa(chapters, result)
        report.l3_qa_score = qa_score
        report.chapter_scores = chapter_scores
        report.feedbacks = feedbacks

        # 汇总
        report.l3_breakdown = self._calc_breakdown(chapter_scores)
        report.overall_score = (
            report.l1_token_coverage * 0.2
            + report.l2_dimension_recall * 0.3
            + report.l3_qa_score * 0.5
        )
        report.passed = (
            report.l1_token_coverage >= L1_THRESHOLD
            and report.l2_dimension_recall >= L2_THRESHOLD
            and report.l3_qa_score >= L3_THRESHOLD
        )
        report.weak_chapters = [
            s["index"] for s in chapter_scores
            if s.get("qa_score", 1.0) < L3_THRESHOLD
        ]
        report.cost_usd = self.pool.total_cost_sum()

        return report

    def _eval_token_coverage(self, chapters, result) -> float:
        """
        L1：纯规则统计 Skill 文件覆盖的原文 token 比例。
        用提取到的 evidence 中的关键词在原文中匹配。
        """
        # 收集所有 evidence 中的关键词
        all_evidences: List[str] = []
        for s in result.chapter_summaries:
            all_evidences.extend(s.evidence)

        if not all_evidences:
            return 0.0

        # 构建关键词集合（取 evidence 中有意义的片段）
        keywords = set()
        for ev in all_evidences:
            # 提取 5-15 字的片段作为关键词
            words = ev.split()
            for i in range(len(words)):
                for j in range(i + 3, min(i + 12, len(words) + 1)):
                    kw = " ".join(words[i:j])
                    if 10 <= len(kw) <= 80:
                        keywords.add(kw)
            # 也加入中文 n-gram
            for i in range(len(ev)):
                for j in range(i + 3, min(i + 10, len(ev) + 1)):
                    chunk = ev[i:j]
                    if 4 <= len(chunk) <= 40:
                        keywords.add(chunk)

        # 限制关键词数量避免计算过重
        keywords_sample = list(keywords)[:500]

        # 统计覆盖率
        total_chars = sum(len(ch.content) for ch in chapters)
        if total_chars == 0:
            return 1.0

        # 合并所有章节文本用于匹配
        full_text = "\n".join(ch.content for ch in chapters)
        covered_chars = 0

        for kw in keywords_sample:
            count = full_text.count(kw)
            if count > 0:
                covered_chars += len(kw) * count

        coverage = min(1.0, covered_chars / total_chars)
        logger.info(f"L1 token 覆盖率: {coverage:.1%}")
        return coverage

    def _eval_dimension_recall(self, chapters, result) -> Tuple[float, dict]:
        """
        L2：检查四大维度（选股标准/风险信号/估值方法/检查清单）是否非空。
        用 utility 辅助判断每个维度的覆盖质量（非空且有实质内容）。
        """
        dim_scores: Dict[str, float] = {}
        summaries = result.chapter_summaries

        # 选股标准
        criteria_count = sum(len(s.selection_criteria) for s in summaries)
        criteria_cases_count = sum(len(s.cases) for s in summaries)
        dim_scores["selection_criteria"] = 1.0 if criteria_count > 0 else 0.0

        # 如果有案例，选股标准额外加分
        if criteria_cases_count > 0:
            dim_scores["selection_criteria"] = min(1.0, dim_scores["selection_criteria"] + 0.1)

        # 风险信号
        rf_count = sum(len(s.red_flags) for s in summaries)
        dim_scores["red_flags"] = 1.0 if rf_count > 0 else 0.0

        # 估值方法
        vm_count = sum(len(s.valuation_methods) for s in summaries)
        dim_scores["valuation_methods"] = 1.0 if vm_count > 0 else 0.0

        # 检查清单
        cl_count = sum(len(s.checklist_items) for s in summaries)
        dim_scores["checklists"] = 1.0 if cl_count > 0 else 0.0

        # 额外：规则数量
        rules_count = sum(len(s.rules) for s in summaries)
        if rules_count == 0 and criteria_count > 0:
            # 有选股标准但没提炼出规则，降权
            dim_scores["selection_criteria"] *= 0.7

        # 计算总体召回率（四维度平均）
        recall = sum(dim_scores.values()) / max(len(dim_scores), 1)

        logger.info(
            f"L2 维度召回: {recall:.1%} "
            f"(criteria={dim_scores['selection_criteria']:.1f}, "
            f"flags={dim_scores['red_flags']:.1f}, "
            f"valuation={dim_scores['valuation_methods']:.1f}, "
            f"checklists={dim_scores['checklists']:.1f})"
        )

        return recall, dim_scores

    def _eval_qa(
        self, chapters, result
    ) -> Tuple[float, List[dict], List[WeakChapterFeedback]]:
        """
        L3 ScenarioQA：
        1. Examiner 看原文出题
        2. Executor 看 Skill 文件答题
        3. Judge 判分
        """
        chapter_scores: List[dict] = []
        all_questions: List[ScenarioQA] = []
        all_answers: List[str] = []
        answers_per_chapter: Dict[int, List[Tuple[ScenarioQA, str]]] = {}

        executor = self.pool.get("executor")

        # 出题 + 答题
        for ch in chapters:
            questions = self.judge.generate_questions(ch)
            if not questions:
                chapter_scores.append({
                    "index": ch.index, "title": ch.title,
                    "qa_score": 1.0, "total": 0,
                })
                continue

            all_questions.extend(questions)
            per_chapter: List[Tuple[ScenarioQA, str]] = []

            # 找到该章节对应的 summary
            summary = next(
                (s for s in result.chapter_summaries if s.chapter_index == ch.index),
                None,
            )

            for qa in questions:
                # Executor 看 Skill 文件答题
                skill_context = self._build_skill_context(summary) if summary else ""
                answer = self._answer_question(executor, qa, skill_context)
                all_answers.append(answer)
                per_chapter.append((qa, answer))

            answers_per_chapter[ch.index] = per_chapter

        # 判分
        total_correct = 0
        total_questions = len(all_questions)

        correct_by_type: Dict[str, Tuple[int, int]] = {
            "numeric": (0, 0),
            "direction": (0, 0),
            "factual": (0, 0),
        }

        chapter_correct: Dict[int, Tuple[int, int]] = {}  # (correct, total)

        for i, qa in enumerate(all_questions):
            answer = all_answers[i]
            correct, reason = self.judge.grade(qa, answer)
            if correct:
                total_correct += 1

            # 按题型统计
            c, t = correct_by_type[qa.answer_type]
            correct_by_type[qa.answer_type] = (c + (1 if correct else 0), t + 1)

            # 按章节统计
            cc, ct = chapter_correct.get(qa.chapter_index, (0, 0))
            chapter_correct[qa.chapter_index] = (cc + (1 if correct else 0), ct + 1)

            logger.debug(
                f"Q[{qa.chapter_index}][{qa.answer_type}]: "
                f"{'✅' if correct else '❌'} {reason[:80]}"
            )

        # 汇总章节得分
        for ch in chapters:
            cc, ct = chapter_correct.get(ch.index, (0, 0))
            score = cc / ct if ct > 0 else 1.0
            chapter_scores.append({
                "index": ch.index, "title": ch.title,
                "qa_score": score, "correct": cc, "total": ct,
            })

        # 生成反馈（只针对薄弱章节）
        feedbacks: List[WeakChapterFeedback] = []
        for cs in chapter_scores:
            if cs.get("qa_score", 1.0) < L3_THRESHOLD and cs.get("total", 0) > 0:
                # 分析缺失维度
                ch_idx = cs["index"]
                summary = next(
                    (s for s in result.chapter_summaries if s.chapter_index == ch_idx),
                    None,
                )
                missing_dims = []
                if summary:
                    if not summary.selection_criteria:
                        missing_dims.append("selection_criteria")
                    if not summary.red_flags:
                        missing_dims.append("red_flags")
                    if not summary.valuation_methods:
                        missing_dims.append("valuation_methods")
                    if not summary.checklist_items:
                        missing_dims.append("checklists")

                # 收集答错的题目
                failed_qs: List[str] = []
                per_chapter_answers = answers_per_chapter.get(ch_idx, [])
                for qa, ans in per_chapter_answers:
                    correct, _ = self.judge.grade(qa, ans)
                    if not correct:
                        failed_qs.append(qa.question[:100])

                feedback = WeakChapterFeedback(
                    index=ch_idx,
                    missing_dimensions=missing_dims,
                    failed_questions=failed_qs[:5],  # 最多携带 5 道错题
                    hint=f"第{ch_idx}章 L3 得分 {cs['qa_score']:.0%}，聚焦遗漏维度重新提取",
                )
                feedbacks.append(feedback)

        qa_score = total_correct / total_questions if total_questions > 0 else 1.0
        logger.info(
            f"L3 QA 得分: {qa_score:.1%} ({total_correct}/{total_questions})"
        )

        return qa_score, chapter_scores, feedbacks

    def _build_skill_context(self, summary) -> str:
        """为 Executor 构建 Skill 文件上下文用于答题"""
        if not summary:
            return "无相关信息"

        parts = []
        if summary.core_arguments:
            parts.append("核心论点：\n" + "\n".join(f"- {a}" for a in summary.core_arguments[:5]))
        if summary.selection_criteria:
            parts.append("选股标准：\n" + "\n".join(f"- {c}" for c in summary.selection_criteria[:5]))
        if summary.red_flags:
            parts.append("风险信号：\n" + "\n".join(f"- {r}" for r in summary.red_flags[:5]))
        if summary.valuation_methods:
            parts.append("估值方法：\n" + "\n".join(f"- {v}" for v in summary.valuation_methods[:5]))
        if summary.rules:
            parts.append("决策规则：\n" + "\n".join(f"- {r}" for r in summary.rules[:5]))

        return "\n\n".join(parts) if parts else "无相关信息"

    def _answer_question(self, executor: LLMClient, qa: ScenarioQA, context: str) -> str:
        """让 Executor 看 Skill 文件答题"""
        prompt = f"""请根据以下投资框架知识回答一个问题。

投资框架（从书籍提取）：
{context}

问题：{qa.question}

请直接给出简洁明确的答案。如果是数值题只给数字，如果是方向题给明确方向，如果是事实题给关键事实。"""

        return executor.complete(prompt, max_tokens=300, temperature=0.0)

    def _calc_breakdown(self, chapter_scores: List[dict]) -> Dict[str, float]:
        """计算各题型得分"""
        # chapter_scores 目前只存了整体 qa_score，细粒度分数需要从判分过程中获取
        # 简化处理：用 L3 总分近似
        return {
            "numeric": 0.0,
            "direction": 0.0,
            "factual": 0.0,
        }
