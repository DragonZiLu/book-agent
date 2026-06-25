"""
Judge + 三层评估器 V2（深度改进版）

核心改进：
  L1 = 信息密度覆盖（压缩比 + 结构完整性 + 实体保留率），替代简单 token 覆盖率
  L2 = 真召回率（gold concepts 抽取 + 语义匹配），替代"非空即过"
  L3 = 闭卷信息瓶颈测试（答题只看 Skill 不触原文），三态判分（对/错/未覆盖/幻觉）

设计原则：
  - L1 纯规则零 LLM 成本，L2 用 judge（异源于 executor）抽 gold concepts
  - L3 答题者只看 Skill 不能看原文，真正检验"信息压缩后关键决策信息是否丢失"
  - 所有 LLM 调用并行化（per-chapter），评估时间大幅缩短
"""

import re
import json
import hashlib
import logging
import time
import concurrent.futures
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Literal
from dataclasses import dataclass, field

from tqdm import tqdm

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

# ============ 本地嵌入模型（语义匹配）============
_EMBED_MODEL = None


def _get_embed_model():
    """延迟加载 sentence-transformers，避免启动开销"""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBED_MODEL = SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2"
            )
        except ImportError:
            logger.warning(
                "sentence-transformers 未安装，语义匹配将降级为关键词匹配"
            )
            _EMBED_MODEL = False
    return _EMBED_MODEL if _EMBED_MODEL is not False else None


# ============ 按策略类型的压缩比范围 ============
COMPRESSION_RATIO_RANGE = {
    "value":   (0.05, 0.35),
    "growth":  (0.05, 0.30),
    "quant":   (0.03, 0.25),
    "macro":   (0.04, 0.30),
    "general": (0.05, 0.35),
}

# 关键维度分类
VETO_DIMS = ["red_flags", "selection_criteria"]   # 完全为空 → 一票否决
WEAK_DIMS = ["valuation_methods", "checklists"]    # 召回不足 → 降权

# 实体抽取正则（投资领域关键数字/指标）
ENTITY_PATTERN = re.compile(
    r"\d+\.?\d*\s*[%倍]|PE|PB|ROE|ROA|ROIC|PEG|PS|EPS|"
    r"EV/EBITDA|流动比率|速动比率|负债率|市净率|市盈率|股息率|"
    r"安全边际|DCF|自由现金流|营收增速|利润增速|毛利率|净利率"
)


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
class GradeResult:
    """三态判分结果"""
    verdict: Literal["correct", "wrong", "not_covered", "hallucinated"]
    reason: str


@dataclass
class NumericConstraint:
    """数值约束三元组：指标 + 算子 + 值"""
    metric: str
    operator: str   # "<=" | ">=" | "<" | ">" | "==" | "!="
    value: float
    raw: str = ""


@dataclass
class WeakChapterFeedback:
    """薄弱章节的反馈信息，供重提取使用"""
    index: int
    missing_dimensions: List[str] = field(default_factory=list)
    missing_concepts: List[str] = field(default_factory=list)     # ← V2新增
    failed_questions: List[str] = field(default_factory=list)
    hint: str = ""


@dataclass
class L1ChapterScore:
    """L1 逐章信息密度分数"""
    index: int
    density: float         # 压缩比得分
    structure: float       # 结构完整性
    entity_retention: float # 实体保留率
    overall: float         # 加权综合


@dataclass
class QualityReport:
    """三层评估结果完整报告（V2 增强版）"""
    # L1
    l1_token_coverage: float = 0.0
    l1_chapter_scores: List[L1ChapterScore] = field(default_factory=list)
    # L2
    l2_dimension_recall: float = 0.0
    l2_dim_scores: Dict[str, float] = field(default_factory=dict)
    l2_missed_concepts: Dict[int, List[str]] = field(default_factory=dict)
    # L3
    l3_qa_score: float = 0.0
    l3_breakdown: Dict[str, float] = field(default_factory=dict)
    l3_ci: Tuple[float, float] = (0.0, 0.0)   # 置信区间
    # 汇总
    overall_score: float = 0.0
    passed: bool = False
    veto_reason: str = ""
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
        self._qa_cache: Dict[str, List[ScenarioQA]] = {}
        self._qa_cache_dir = Path(".cache/qa")

    # ── 出题（带缓存，温度归零）─────────────────────

    def generate_questions(self, chapter) -> List[ScenarioQA]:
        """
        Examiner 从原文出题，题型分布按 config.QA_TYPE_DISTRIBUTION。
        同一章内容永远出同一套题（hash 缓存 + temperature=0），确保分数可复现。
        """
        cache_key = hashlib.md5(chapter.content.encode("utf-8")).hexdigest()[:16]

        # 内存缓存
        if cache_key in self._qa_cache:
            logger.debug(f"QA缓存命中（内存）: ch{chapter.index}")
            return self._qa_cache[cache_key]

        # 磁盘缓存
        disk_cache = self._load_qa_from_disk(cache_key)
        if disk_cache is not None:
            self._qa_cache[cache_key] = disk_cache
            return disk_cache

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
                temperature=0.0,   # ← V2: 温度归零确保确定性
            )
            questions = self._parse_questions(response, chapter.index)
            logger.debug(f"第{chapter.index}章出题 {len(questions)} 道")

            # 写入缓存
            self._qa_cache[cache_key] = questions
            self._save_qa_to_disk(cache_key, questions)

            return questions
        except Exception as e:
            logger.warning(f"出题失败 第{chapter.index}章: {e}")
            return []

    def _parse_questions(self, response: str, chapter_index: int) -> List[ScenarioQA]:
        """解析 LLM 输出的题 JSON（多层容错）"""
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            m = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
            if m:
                try:
                    data = json.loads(m.group(1))
                except json.JSONDecodeError:
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

    def _load_qa_from_disk(self, cache_key: str) -> Optional[List[ScenarioQA]]:
        """从磁盘加载缓存的题目"""
        cache_file = self._qa_cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return [
                ScenarioQA(**item) for item in raw
            ]
        except Exception as e:
            logger.debug(f"QA磁盘缓存损坏 {cache_key}: {e}")
            return None

    def _save_qa_to_disk(self, cache_key: str, questions: List[ScenarioQA]) -> None:
        """保存题目到磁盘缓存"""
        self._qa_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self._qa_cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(
                    [{
                        "question": q.question,
                        "gold_answer": q.gold_answer,
                        "evidence_span": q.evidence_span,
                        "answer_type": q.answer_type,
                        "chapter_index": q.chapter_index,
                    } for q in questions],
                    f, ensure_ascii=False, indent=2,
                )
        except OSError as e:
            logger.debug(f"QA磁盘缓存写入失败: {e}")

    # ── 三态判分 ─────────────────────────────────

    def grade(self, qa: ScenarioQA, answer: str) -> GradeResult:
        """
        三态判分：
        - correct: 答案正确
        - wrong: 答案错误
        - not_covered: 模型诚实表示无法回答（Skill 未涵盖）
        - hallucinated: 模型编造了不存在的信息
        """
        # 先检查是否弃权
        if self._is_abstention(answer):
            return GradeResult("not_covered", "模型明确表示无法基于提供的信息回答")

        if qa.answer_type == "numeric":
            return self._grade_numeric(qa, answer)
        elif qa.answer_type == "direction":
            return self._grade_direction(qa, answer)
        else:
            return self._grade_factual(qa, answer)

    def _is_abstention(self, answer: str) -> bool:
        """检测答案是否为弃权/无法回答"""
        abstention_phrases = [
            "无法回答", "无法确定", "无从判断", "信息不足",
            "未提供", "未涵盖", "没有提及", "不知道", "不清楚",
            "无法基于", "资料不足", "cannot answer", "not covered",
            "not mentioned", "insufficient information", "unable to determine",
        ]
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in abstention_phrases)

    # ── 数值题：约束三元组比对 ──────────────────

    def _grade_numeric(self, qa: ScenarioQA, answer: str) -> GradeResult:
        """数值题：抽取约束三元组 (指标, 算子, 值) 比对"""
        gold_constraints = self._extract_constraints(qa.gold_answer)
        ans_constraints = self._extract_constraints(answer)

        if not gold_constraints:
            logger.warning(f"gold_answer 中未找到约束: {qa.gold_answer[:100]}")
            # 降级为纯数字比对
            return self._grade_numeric_fallback(qa, answer)

        if not ans_constraints:
            # 答案中没有任何约束 → 可能是弃权或没给明确数值
            if self._is_abstention(answer):
                return GradeResult("not_covered", "未提供数值答案")
            return GradeResult("wrong", "答案中未找到数值约束")

        # 逐一比对每个 gold 约束
        matched = 0
        details = []
        for gc in gold_constraints:
            found = False
            for ac in ans_constraints:
                if (ac.metric.upper() == gc.metric.upper()
                        and ac.operator == gc.operator
                        and abs(ac.value - gc.value) <= 0.02 * max(abs(gc.value), 0.01)):
                    found = True
                    break
            if found:
                matched += 1
            else:
                details.append(f"期望 {gc.metric}{gc.operator}{gc.value}")

        if matched == len(gold_constraints):
            return GradeResult("correct", f"全部 {len(gold_constraints)} 条约束匹配")
        elif matched > 0:
            return GradeResult(
                "wrong",
                f"部分匹配 ({matched}/{len(gold_constraints)})，缺失: {'; '.join(details)}"
            )
        else:
            # 检查是否幻觉（数字对不上但编造了数值）
            ans_numbers = self._extract_numbers(answer)
            if ans_numbers:
                return GradeResult(
                    "hallucinated",
                    f"约束不匹配: {'; '.join(details)}"
                )
            return GradeResult("wrong", f"约束不匹配: {'; '.join(details)}")

    def _grade_numeric_fallback(self, qa: ScenarioQA, answer: str) -> GradeResult:
        """降级：纯数字比对（±2% 容差）"""
        gold_nums = self._extract_numbers(qa.gold_answer)
        ans_nums = self._extract_numbers(answer)

        if not gold_nums:
            return GradeResult("wrong", "gold_answer 无数字可提取")
        if not ans_nums:
            return GradeResult("wrong", "答案中未找到数字")

        gold = gold_nums[0]
        for an in ans_nums:
            if gold == 0:
                if an == 0:
                    return GradeResult("correct", f"匹配: {an} == {gold}")
                continue
            error = abs(an - gold) / abs(gold)
            if error <= 0.02:
                return GradeResult("correct", f"匹配: {an} ≈ {gold} (误差 {error:.1%})")

        return GradeResult("wrong", f"不匹配: answer={ans_nums[:3]}, gold={gold_nums[:3]}")

    # ── 方向题判分 ──────────────────────────────

    def _grade_direction(self, qa: ScenarioQA, answer: str) -> GradeResult:
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
5. 考生编造原文不存在的事实 → 标记为幻觉

只输出一个 JSON：
{{"correct": true/false, "hallucinated": false, "reason": "简短理由"}}"""

        resp = self.judge.complete(prompt, max_tokens=200, temperature=0.0)
        try:
            data = json.loads(resp) if resp.strip().startswith("{") else \
                json.loads(re.search(r"\{[\s\S]*\}", resp).group(0))
            correct = data.get("correct", False)
            hallucinated = data.get("hallucinated", False)
            reason = data.get("reason", "")

            if hallucinated:
                return GradeResult("hallucinated", reason)
            if correct:
                return GradeResult("correct", reason)
            return GradeResult("wrong", reason)
        except (json.JSONDecodeError, AttributeError):
            gold_lower = qa.gold_answer.lower()
            ans_lower = answer.lower()
            if gold_lower[:20] in ans_lower:
                return GradeResult("correct", "fallback 文本匹配")
            return GradeResult("wrong", "fallback 判断")

    # ── 事实题判分 ──────────────────────────────

    def _grade_factual(self, qa: ScenarioQA, answer: str) -> GradeResult:
        """事实题：Judge 语义判分，核心事实正确即算对"""
        prompt = f"""你是严格的判分员。判断以下答案是否覆盖了核心事实。

题目：{qa.question}

标准答案（原文推导）：{qa.gold_answer}

原文证据：{qa.evidence_span}

考生答案：{answer}

判分规则：
1. 核心事实与标准答案一致 → 正确（true）
2. 事实错误或遗漏核心事实 → 错误（false）
3. 考生编造原文/框架中不存在的信息 → 标记为幻觉
4. 部分正确但不影响核心判断 → 可酌情给正确

只输出一个 JSON：
{{"correct": true/false, "hallucinated": false, "reason": "简短理由"}}"""

        resp = self.judge.complete(prompt, max_tokens=300, temperature=0.0)
        try:
            data = json.loads(resp) if resp.strip().startswith("{") else \
                json.loads(re.search(r"\{[\s\S]*\}", resp).group(0))
            correct = data.get("correct", False)
            hallucinated = data.get("hallucinated", False)
            reason = data.get("reason", "")

            if hallucinated:
                return GradeResult("hallucinated", reason)
            if correct:
                return GradeResult("correct", reason)
            return GradeResult("wrong", reason)
        except (json.JSONDecodeError, AttributeError):
            return GradeResult("wrong", "无法解析 Judge 响应")

    # ── 数值/约束抽取工具 ──────────────────────

    def _extract_constraints(self, text: str) -> List[NumericConstraint]:
        """抽取 '指标 算子 数值' 三元组"""
        pat = re.compile(
            r"(PE|PB|ROE|ROA|ROIC|PEG|PS|EPS|EV/EBITDA|"
            r"市盈率|市净率|流动比率|速动比率|负债率|股息率|营收增速|利润增速)\s*"
            r"([<>≤≥=]+|不超过|不低于|不高于|不大于|低于|高于|大于|小于|至少|不少于)\s*"
            r"([\d.]+)",
            re.IGNORECASE,
        )
        results = []
        for m in pat.finditer(text):
            metric = m.group(1)
            op_raw = m.group(2)
            try:
                value = float(m.group(3))
            except ValueError:
                continue
            op = self._normalize_operator(op_raw)
            results.append(NumericConstraint(
                metric=metric, operator=op, value=value,
                raw=m.group(0),
            ))
        return results

    @staticmethod
    def _normalize_operator(op_raw: str) -> str:
        """标准化比较运算符"""
        op = op_raw.strip()
        if op in ("不超过", "不高于", "不大于", "低于", "小于"):
            return "<="
        elif op in ("不低于", "至少", "不少于", "高于", "大于"):
            return ">="
        elif op in ("≤",):
            return "<="
        elif op in ("≥",):
            return ">="
        elif op in ("=", "==", "等于"):
            return "=="
        elif op in ("!=", "≠", "不等于"):
            return "!="
        elif op == "<":
            return "<="
        elif op == ">":
            return ">="
        return op

    @staticmethod
    def _extract_numbers(text: str) -> List[float]:
        """从文本中提取所有数字"""
        numbers = []
        for m in re.finditer(r"(-?[\d,]+\.?\d*)\s*%?", text):
            try:
                num_str = m.group(1).replace(",", "")
                num = float(num_str)
                if "%" in text[m.end():m.end() + 2] or m.group(0).endswith("%"):
                    num = num / 100
                numbers.append(num)
            except ValueError:
                pass
        return numbers


# ============ 三层评估器 V2 ============

class QualityEvaluator:
    """三层质量评估 V2：信息密度 L1 → 真召回 L2 → 闭卷 L3"""

    def __init__(self, pool: ModelPool, max_workers: int = 4):
        self.pool = pool
        self.judge = Judge(pool)
        self.max_workers = max_workers
        self._utility: Optional[LLMClient] = None

    @property
    def utility(self) -> LLMClient:
        if self._utility is None:
            self._utility = self.pool.get("utility")
        return self._utility

    # ═══════════════ 主入口 ═══════════════

    def evaluate(
        self, chapters, result, extraction_warnings=None, strategy_type: str = "general"
    ) -> QualityReport:
        """执行完整三层评估（并行化 per-chapter 操作）"""
        t_start = time.time()
        report = QualityReport(
            extraction_warnings=extraction_warnings or [],
        )

        strategy_type = getattr(result.rules_json, "get", lambda k, d: d)("strategy_type", strategy_type)
        n = len(chapters)

        # ── L1：信息密度覆盖（纯规则，极快）──
        logger.info("  🔍 L1 信息密度评估...")
        t0 = time.time()
        l1_overall, l1_chapter_scores = self._eval_information_density(
            chapters, result, strategy_type
        )
        report.l1_token_coverage = l1_overall
        report.l1_chapter_scores = l1_chapter_scores
        logger.info(f"     L1={l1_overall:.1%} | 耗时 {time.time()-t0:.1f}s")

        # ── L2 + L3：并行化 per-chapter ──
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # L2：gold concepts 抽取（每章并行）
            logger.info(f"  🔍 L2 抽取关键概念 (并行×{self.max_workers}, {n}章)...")
            t0 = time.time()
            l2_futures = {executor.submit(self._extract_gold_concepts, ch): ch for ch in chapters}
            gold_concepts_by_ch: Dict[int, List[str]] = {}
            l2_pbar = tqdm(total=n, desc="     L2 概念", unit="章", ncols=80)
            for future in concurrent.futures.as_completed(l2_futures):
                ch = l2_futures[future]
                try:
                    gold_concepts_by_ch[ch.index] = future.result()
                except Exception as e:
                    logger.warning(f"L2 gold concepts 抽取失败 ch{ch.index}: {e}")
                    gold_concepts_by_ch[ch.index] = []
                l2_pbar.update(1)
                l2_pbar.set_postfix_str(f"ch{ch.index}")
            l2_pbar.close()
            total_concepts = sum(len(v) for v in gold_concepts_by_ch.values())
            logger.info(f"     L2: {total_concepts} 个关键概念 | 耗时 {time.time()-t0:.1f}s")

            # L3：出题 + 答题（每章并行）
            logger.info(f"  🔍 L3 出题+答题+判分 (并行×{self.max_workers}, {n}章)...")
            t0 = time.time()
            l3_futures = {executor.submit(self._eval_chapter_l3, ch, result): ch for ch in chapters}
            l3_results: Dict[int, dict] = {}
            l3_pbar = tqdm(total=n, desc="     L3 评估", unit="章", ncols=80)
            for future in concurrent.futures.as_completed(l3_futures):
                ch = l3_futures[future]
                try:
                    l3_results[ch.index] = future.result()
                except Exception as e:
                    logger.warning(f"L3 评估失败 ch{ch.index}: {e}")
                    l3_results[ch.index] = {"qa_score": 1.0, "correct": 0, "total": 0, "not_covered": 0, "hallucinated": 0, "grade_results": [], "questions": []}
                l3_pbar.update(1)
                l3_pbar.set_postfix_str(f"ch{ch.index}")
            l3_pbar.close()
            logger.info(f"     L3: 耗时 {time.time()-t0:.1f}s")

        # L2 召回率计算
        recall, dim_scores, missed_map = self._eval_recall_from_gold(
            chapters, result, gold_concepts_by_ch
        )
        report.l2_dimension_recall = recall
        report.l2_dim_scores = dim_scores
        report.l2_missed_concepts = missed_map

        # L3 汇总
        qa_score, chapter_scores, feedbacks = self._aggregate_l3(
            chapters, l3_results, missed_map, result
        )
        report.l3_qa_score = qa_score
        report.chapter_scores = chapter_scores
        report.feedbacks = feedbacks

        # L3 置信区间
        total_q = sum(cs.get("total", 0) for cs in chapter_scores)
        total_c = sum(cs.get("correct", 0) for cs in chapter_scores)
        report.l3_ci = self._wilson_ci(total_c, total_q)

        # L3 题型 breakdown
        report.l3_breakdown = self._calc_type_breakdown(l3_results)

        # 汇总评分
        report.overall_score = (
            report.l1_token_coverage * 0.15          # V2: L1 权重降低，让给 L2
            + report.l2_dimension_recall * 0.40      # L2 权重提高（召回在投资场景更关键）
            + report.l3_qa_score * 0.45
        )

        # 通过判定（含硬否决）
        report.passed = self._check_pass(report, dim_scores)

        report.weak_chapters = [
            s["index"] for s in chapter_scores
            if s.get("qa_score", 1.0) < L3_THRESHOLD
        ]
        report.cost_usd = self.pool.total_cost_sum()

        elapsed = time.time() - t_start
        logger.info(f"  ✅ 评估完成, 总耗时 {elapsed:.1f}s, 成本 ${report.cost_usd:.4f}")

        return report

    # ═══════════════ L1：信息密度覆盖 ═══════════════

    def _eval_information_density(
        self, chapters, result, strategy_type: str
    ) -> Tuple[float, List[L1ChapterScore]]:
        """
        L1 V2 = 信息密度覆盖（纯规则，零 LLM 成本）：
          a) 压缩比是否合理
          b) 结构完整性（关键字段是否产出）
          c) 实体保留率（原文数字/指标是否被带进摘要）
        """
        ratio_range = COMPRESSION_RATIO_RANGE.get(strategy_type, (0.05, 0.35))
        lo, hi = ratio_range
        chapter_scores: List[L1ChapterScore] = []

        for ch in chapters:
            summary = self._find_summary(result, ch.index)
            if not summary:
                chapter_scores.append(L1ChapterScore(
                    index=ch.index, density=0.0, structure=0.0,
                    entity_retention=0.0, overall=0.0,
                ))
                continue

            # a) 压缩比
            summary_text = self._summary_to_text(summary)
            summary_tokens = max(len(summary_text) // 3, 1)
            ratio = summary_tokens / max(ch.token_count, 1)
            if lo <= ratio <= hi:
                density = 1.0
            elif ratio < lo:
                density = ratio / lo  # 太短，线性惩罚
            else:
                density = max(0.0, 1.0 - (ratio - hi) / hi)  # 太长

            # b) 结构完整性
            fields = [
                bool(summary.core_arguments),
                bool(summary.selection_criteria),
                bool(summary.red_flags or summary.valuation_methods),
            ]
            structure = sum(fields) / len(fields)

            # c) 实体保留率
            entity = self._entity_retention(ch.content, summary_text)

            overall = 0.3 * density + 0.3 * structure + 0.4 * entity
            chapter_scores.append(L1ChapterScore(
                index=ch.index, density=density, structure=structure,
                entity_retention=entity, overall=overall,
            ))

        if not chapter_scores:
            return 0.0, []

        overall = sum(cs.overall for cs in chapter_scores) / len(chapter_scores)
        logger.info(
            f"L1 信息密度: {overall:.1%} "
            f"(density={sum(cs.density for cs in chapter_scores)/len(chapter_scores):.1%}, "
            f"structure={sum(cs.structure for cs in chapter_scores)/len(chapter_scores):.1%}, "
            f"entity={sum(cs.entity_retention for cs in chapter_scores)/len(chapter_scores):.1%})"
        )
        return overall, chapter_scores

    def _entity_retention(self, original: str, summary: str) -> float:
        """原文关键数字/指标在摘要中的保留率（纯规则）"""
        orig_ents = set(ENTITY_PATTERN.findall(original))
        if not orig_ents:
            return 1.0
        # 在摘要中查找（用原始字符串匹配，保留上下文）
        kept = sum(1 for e in orig_ents if e in summary)
        return kept / len(orig_ents)

    @staticmethod
    def _summary_to_text(summary) -> str:
        """将 ChapterSummary 转为纯文本用于分析"""
        parts = []
        for attr in ["core_arguments", "selection_criteria", "red_flags",
                      "valuation_methods", "rules", "checklist_items", "evidence"]:
            vals = getattr(summary, attr, [])
            if vals:
                parts.extend(vals)
        return "\n".join(parts)

    # ═══════════════ L2：真召回率 ═══════════════

    def _extract_gold_concepts(self, chapter) -> List[str]:
        """
        用 judge 模型（异源于 executor）从原文抽取"应该被提取"的关键概念清单。
        这是 L2 的 ground truth。
        """
        judge_client = self.pool.get("judge")

        prompt = f"""你是一位投资分析审查员。请从以下投资书籍章节中，列出所有应该被提取到投资框架中的关键概念。

要求：
1. 只列出具体的、可验证的概念（如 PE≤15、流动比率≥2、ROE>15%、安全边际原则等）
2. 排除泛泛而谈的一般性讨论
3. 每个概念一行，格式: "- 概念描述（如有数值则包含阈值）"
4. 至少列出 3 个概念，最多列出 10 个

章节标题：{chapter.title}

章节内容：
{chapter.content[:6000]}

请列出关键概念（每行一个，以 - 开头）："""

        try:
            response = judge_client.complete(
                prompt=prompt, max_tokens=600, temperature=0.0,
            )
            # 解析以 - 或 * 开头的行
            concepts = []
            for line in response.split("\n"):
                line = line.strip()
                if line.startswith(("-", "*", "•", "·")):
                    concept = line.lstrip("-*•· ").strip()
                    if len(concept) > 3:
                        concepts.append(concept)
            logger.debug(f"ch{chapter.index} gold concepts: {len(concepts)} 条")
            return concepts
        except Exception as e:
            logger.warning(f"Gold concepts 抽取失败 ch{chapter.index}: {e}")
            return []

    def _eval_recall_from_gold(
        self, chapters, result, gold_concepts_by_ch: Dict[int, List[str]]
    ) -> Tuple[float, Dict[str, float], Dict[int, List[str]]]:
        """
        L2 V2 = 真召回率：
        检查每个 gold concept 是否被 Skill 文件覆盖（语义匹配）。
        """
        # 构建 Skill 文本（四个维度）
        skill_texts = {
            "selection_criteria": result.criteria or "",
            "red_flags": result.red_flags or "",
            "valuation_methods": result.valuation or "",
            "checklists": result.checklists or "",
        }

        # 汇总所有 gold concepts
        all_gold: List[Tuple[int, str]] = []
        for ch_idx, concepts in gold_concepts_by_ch.items():
            for c in concepts:
                all_gold.append((ch_idx, c))

        if not all_gold:
            logger.warning("L2: 无 gold concepts，跳过召回评估")
            return 1.0, {d: 1.0 for d in REQUIRED_DIMENSIONS}, {}

        # 对每个概念检查是否被 Skill 覆盖
        dim_hits: Dict[str, int] = {d: 0 for d in REQUIRED_DIMENSIONS}
        dim_total: Dict[str, int] = {d: 0 for d in REQUIRED_DIMENSIONS}
        missed_by_ch: Dict[int, List[str]] = {}

        # 概念→维度映射（简单关键词映射）
        def _map_to_dims(concept: str) -> List[str]:
            dims = []
            c_lower = concept.lower()
            if any(kw in c_lower for kw in ["pe", "pb", "roe", "选股", "标准", "条件", "筛选"]):
                dims.append("selection_criteria")
            if any(kw in c_lower for kw in ["风险", "危险", "警示", "造假", "负债", "亏损", "red flag"]):
                dims.append("red_flags")
            if any(kw in c_lower for kw in ["估值", "dcf", "现金流", "折现", "安全边际", "valuation"]):
                dims.append("valuation_methods")
            if any(kw in c_lower for kw in ["检查", "清单", "确认", "核查", "checklist"]):
                dims.append("checklists")
            if not dims:
                dims = ["selection_criteria"]  # 默认归入选股标准
            return dims

        for ch_idx, concept in all_gold:
            dims = _map_to_dims(concept)
            covered = self._concept_covered_in_skill(concept, skill_texts)
            for d in dims:
                dim_total[d] = dim_total.get(d, 0) + 1
                if covered:
                    dim_hits[d] = dim_hits.get(d, 0) + 1
            if not covered:
                missed_by_ch.setdefault(ch_idx, []).append(concept)

        # 计算各维度召回率
        dim_scores = {}
        for d in REQUIRED_DIMENSIONS:
            if dim_total.get(d, 0) > 0:
                dim_scores[d] = dim_hits.get(d, 0) / dim_total[d]
            else:
                dim_scores[d] = 1.0  # 无概念则满分

        # 加权平均（关键维度权重更高）
        weights = {
            "selection_criteria": 0.30,
            "red_flags": 0.35,       # 风险信号权重最高
            "valuation_methods": 0.20,
            "checklists": 0.15,
        }
        recall = sum(dim_scores[d] * weights.get(d, 0.25) for d in REQUIRED_DIMENSIONS)
        recall = recall / sum(weights.get(d, 0.25) for d in REQUIRED_DIMENSIONS)

        logger.info(
            f"L2 真召回: {recall:.1%} "
            f"(criteria={dim_scores.get('selection_criteria', 0):.1%}, "
            f"flags={dim_scores.get('red_flags', 0):.1%}, "
            f"val={dim_scores.get('valuation_methods', 0):.1%}, "
            f"check={dim_scores.get('checklists', 0):.1%})"
        )

        return recall, dim_scores, missed_by_ch

    def _concept_covered_in_skill(self, concept: str, skill_texts: Dict[str, str]) -> bool:
        """
        检查概念是否被 Skill 文件覆盖。
        优先使用语义匹配（sentence-transformers），降级为关键词匹配。
        """
        model = _get_embed_model()
        if model is not None:
            return self._semantic_match(concept, skill_texts, model)
        else:
            return self._keyword_match(concept, skill_texts)

    def _semantic_match(
        self, concept: str, skill_texts: Dict[str, str], model
    ) -> bool:
        """嵌入向量相似度匹配"""
        try:
            from sentence_transformers import util
            concept_emb = model.encode(concept, convert_to_tensor=True)

            for dim, text in skill_texts.items():
                if not text or len(text) < 20:
                    continue
                # 按段落切分
                paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 15]
                if not paragraphs:
                    continue
                para_embs = model.encode(paragraphs, convert_to_tensor=True)
                scores = util.cos_sim(concept_emb, para_embs)[0]
                if scores.max().item() >= 0.70:
                    return True
            return False
        except Exception as e:
            logger.debug(f"语义匹配失败，降级关键词: {e}")
            return self._keyword_match(concept, skill_texts)

    @staticmethod
    def _keyword_match(concept: str, skill_texts: Dict[str, str]) -> bool:
        """降级：关键词子串匹配"""
        # 提取概念中的关键词
        keywords = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z/]{2,}|\d+\.?\d*", concept)
        if not keywords:
            keywords = [concept[:20]]

        all_skill_text = " ".join(skill_texts.values()).lower()
        # 至少 60% 的关键词出现在 Skill 中
        hits = sum(1 for kw in keywords if kw.lower() in all_skill_text)
        return hits / len(keywords) >= 0.5

    # ═══════════════ L3：闭卷信息瓶颈测试 ═══════════════

    def _eval_chapter_l3(self, chapter, result) -> dict:
        """
        L3 V2 单章评估（闭卷）：
        1. Examiner 看原文出题
        2. Answerer (judge) 只看 Skill 文件答题 ← 关键改进！
        3. Judge 三态判分
        """
        questions = self.judge.generate_questions(chapter)
        if not questions:
            return {
                "qa_score": 1.0, "correct": 0, "total": 0,
                "not_covered": 0, "hallucinated": 0,
                "grade_results": [], "questions": [],
            }

        # 该章的 Skill 摘要
        summary = self._find_summary(result, chapter.index)
        skill_context = self._build_skill_context(summary) if summary else ""

        answerer = self.pool.get("judge")  # ← V2: 用 judge 答题，异源于 executor

        grade_results: List[GradeResult] = []
        for qa in questions:
            # 关键：answerer 只拿到 Skill，拿不到原文
            answer = self._answer_closed_book(answerer, qa, skill_context)
            gr = self.judge.grade(qa, answer)
            grade_results.append(gr)

            logger.debug(
                f"Q[ch{chapter.index}][{qa.answer_type}]: "
                f"{gr.verdict} {gr.reason[:80]}"
            )

        # 统计
        correct = sum(1 for gr in grade_results if gr.verdict == "correct")
        not_covered = sum(1 for gr in grade_results if gr.verdict == "not_covered")
        hallucinated = sum(1 for gr in grade_results if gr.verdict == "hallucinated")
        total = len(questions)

        # L3 章节得分 = correct / (correct + wrong)，not_covered 不计入分母
        wrong = sum(1 for gr in grade_results if gr.verdict == "wrong")
        evaluable = correct + wrong
        qa_score = correct / evaluable if evaluable > 0 else 1.0

        return {
            "qa_score": qa_score,
            "correct": correct,
            "wrong": wrong,
            "total": total,
            "not_covered": not_covered,
            "hallucinated": hallucinated,
            "grade_results": grade_results,
            "questions": questions,
        }

    def _answer_closed_book(
        self, answerer: LLMClient, qa: ScenarioQA, context: str
    ) -> str:
        """闭卷答题：只看 Skill，不能看原文"""
        if len(context.split()) < 30:
            # Skill 太短，直接标记为不充分，不浪费 LLM 调用
            return "无法回答，提供的投资框架信息不足。"

        prompt = f"""请仅根据以下投资框架（不得编造框架之外的信息）回答问题。
如果框架中没有相关信息，请如实回答"框架未涵盖此内容"。

投资框架（从书籍提取）：
{context}

问题：{qa.question}

答题规则：
1. 仅使用上述框架中提供的信息
2. 框架未涵盖的信息直接说"框架未涵盖"，不要猜测
3. 数值题只给数字和单位
4. 方向题给明确方向
5. 事实题给关键事实"""

        return answerer.complete(prompt, max_tokens=300, temperature=0.0)

    def _build_skill_context(self, summary) -> str:
        """为答题者构建 Skill 上下文（与 V1 相同）"""
        if not summary:
            return ""

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

        return "\n\n".join(parts) if parts else ""

    def _aggregate_l3(
        self, chapters, l3_results: dict, missed_map: dict, result
    ) -> Tuple[float, List[dict], List[WeakChapterFeedback]]:
        """汇总 L3 结果，生成反馈"""
        chapter_scores: List[dict] = []

        for ch in chapters:
            ch_result = l3_results.get(ch.index, {
                "qa_score": 1.0, "correct": 0, "total": 0,
                "not_covered": 0, "hallucinated": 0,
            })
            chapter_scores.append({
                "index": ch.index,
                "title": ch.title,
                "qa_score": ch_result["qa_score"],
                "correct": ch_result["correct"],
                "wrong": ch_result.get("wrong", 0),
                "total": ch_result["total"],
                "not_covered": ch_result.get("not_covered", 0),
                "hallucinated": ch_result.get("hallucinated", 0),
                "ci": self._wilson_ci(
                    ch_result["correct"],
                    ch_result["correct"] + ch_result.get("wrong", 0)
                ),
            })

        total_correct = sum(cs["correct"] for cs in chapter_scores)
        total_wrong = sum(cs.get("wrong", 0) for cs in chapter_scores)
        total_evaluable = total_correct + total_wrong
        qa_score = total_correct / total_evaluable if total_evaluable > 0 else 1.0

        # 生成反馈（含 L2 missed concepts）
        feedbacks: List[WeakChapterFeedback] = []
        for cs in chapter_scores:
            ch_idx = cs["index"]
            if cs.get("qa_score", 1.0) < L3_THRESHOLD and cs.get("total", 0) > 0:
                summary = self._find_summary(result, ch_idx)
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

                # V2: 携带 L2 发现的遗漏概念
                missing_concepts = missed_map.get(ch_idx, [])[:5]

                feedback = WeakChapterFeedback(
                    index=ch_idx,
                    missing_dimensions=missing_dims,
                    missing_concepts=missing_concepts,
                    failed_questions=[],
                    hint=f"第{ch_idx}章 L3={cs['qa_score']:.0%}, "
                         f"not_covered={cs.get('not_covered', 0)}, "
                         f"missed_concepts={len(missing_concepts)}",
                )
                feedbacks.append(feedback)

        logger.info(
            f"L3 QA 得分: {qa_score:.1%} ({total_correct}/{total_evaluable} 可评估, "
            f"{sum(cs.get('not_covered', 0) for cs in chapter_scores)} 未覆盖, "
            f"{sum(cs.get('hallucinated', 0) for cs in chapter_scores)} 幻觉)"
        )

        return qa_score, chapter_scores, feedbacks

    # ═══════════════ 通过判定 ═══════════════

    def _check_pass(self, report: QualityReport, dim_scores: Dict[str, float]) -> bool:
        """综合判定，含硬否决"""
        # 硬否决：关键维度完全为空
        for d in VETO_DIMS:
            if dim_scores.get(d, 0) == 0.0:
                report.veto_reason = f"关键维度 {d} 完全缺失，Skill 不可用于投资决策"
                logger.warning(f"❌ 一票否决: {report.veto_reason}")
                return False

        # 标准阈值判定
        l1_ok = report.l1_token_coverage >= L1_THRESHOLD
        l2_ok = report.l2_dimension_recall >= L2_THRESHOLD
        l3_ok = report.l3_qa_score >= L3_THRESHOLD

        if not l2_ok and any(dim_scores.get(d, 1.0) < 0.5 for d in WEAK_DIMS):
            logger.warning("⚠️ L2 关键维度召回不足（非硬否决但属高危）")

        return l1_ok and l2_ok and l3_ok

    # ═══════════════ 辅助方法 ═══════════════

    @staticmethod
    def _find_summary(result, chapter_index: int):
        """从结果中查找章节摘要"""
        for s in result.chapter_summaries:
            if s.chapter_index == chapter_index:
                return s
        return None

    @staticmethod
    def _wilson_ci(correct: int, total: int, alpha: float = 0.10) -> Tuple[float, float]:
        """Wilson score confidence interval（适用小样本）"""
        if total == 0:
            return (0.0, 1.0)
        try:
            from scipy.stats import binomtest
            ci = binomtest(correct, total).proportion_ci(confidence_level=1 - alpha)
            return (max(0.0, ci.low), min(1.0, ci.high))
        except ImportError:
            # 简易近似
            import math
            p = correct / total
            z = 1.645  # 90% z-score
            se = math.sqrt(p * (1 - p) / total)
            low = max(0.0, p - z * se)
            high = min(1.0, p + z * se)
            return (low, high)

    def _calc_type_breakdown(self, l3_results: dict) -> Dict[str, float]:
        """按题型统计得分"""
        type_correct: Dict[str, int] = {"numeric": 0, "direction": 0, "factual": 0}
        type_total: Dict[str, int] = {"numeric": 0, "direction": 0, "factual": 0}

        for ch_result in l3_results.values():
            questions = ch_result.get("questions", [])
            grade_results = ch_result.get("grade_results", [])
            for qa, gr in zip(questions, grade_results):
                atype = qa.answer_type if hasattr(qa, "answer_type") else "factual"
                if gr.verdict == "correct":
                    type_correct[atype] = type_correct.get(atype, 0) + 1
                if gr.verdict in ("correct", "wrong"):
                    type_total[atype] = type_total.get(atype, 0) + 1

        return {
            atype: type_correct[atype] / type_total[atype]
            if type_total[atype] > 0 else 0.0
            for atype in ("numeric", "direction", "factual")
        }


# ============ 黄金集校准接口 ============

class EvaluatorCalibrator:
    """
    元评估：用人工标注的黄金测试集校准评估器自身的判分准确性。
    每次修改 Judge 判分逻辑后运行，确保与人工标注一致率 > 90%。
    """

    def __init__(self, pool: ModelPool):
        self.pool = pool
        self.judge = Judge(pool)

    def run_calibration(self, golden_path: str) -> dict:
        """
        运行校准，返回一致率报告。

        golden_path: 黄金测试集 JSON 文件路径
          格式: [{"question": "...", "gold_answer": "...", "evidence_span": "...",
                  "answer_type": "numeric", "test_cases": [
                     {"answer": "...", "expect": "correct|wrong|not_covered"}
                  ]}]
        """
        with open(golden_path, "r", encoding="utf-8") as f:
            test_cases = json.load(f)

        results = {
            "total": 0,
            "agreed": 0,
            "by_type": {},
            "disagreements": [],
        }

        for item in test_cases:
            qa = ScenarioQA(
                question=item["question"],
                gold_answer=item["gold_answer"],
                evidence_span=item.get("evidence_span", ""),
                answer_type=item.get("answer_type", "factual"),
                chapter_index=item.get("chapter_index", 0),
            )

            for tc in item.get("test_cases", []):
                expected = tc["expect"]
                answer = tc["answer"]

                gr = self.judge.grade(qa, answer)
                actual = gr.verdict

                results["total"] += 1
                atype = qa.answer_type
                if atype not in results["by_type"]:
                    results["by_type"][atype] = {"total": 0, "agreed": 0}
                results["by_type"][atype]["total"] += 1

                if actual == expected:
                    results["agreed"] += 1
                    results["by_type"][atype]["agreed"] += 1
                else:
                    results["disagreements"].append({
                        "question": qa.question[:100],
                        "answer": answer[:100],
                        "expected": expected,
                        "actual": actual,
                        "reason": gr.reason,
                        "type": atype,
                    })

        if results["total"] > 0:
            results["agreement_rate"] = results["agreed"] / results["total"]
            for atype, stats in results["by_type"].items():
                stats["rate"] = stats["agreed"] / stats["total"] if stats["total"] > 0 else 0

        return results

    def print_report(self, results: dict):
        """打印校准报告"""
        print(f"\n{'='*60}")
        print(f"  🎯 黄金集校准报告")
        print(f"{'='*60}")
        print(f"  总题数: {results['total']}")
        print(f"  一致数: {results['agreed']}")
        print(f"  一致率: {results.get('agreement_rate', 0):.1%}")
        print()
        print(f"  按题型:")
        for atype, stats in sorted(results.get("by_type", {}).items()):
            status = "✅" if stats.get("rate", 0) >= 0.90 else "⚠️"
            print(f"    {status} {atype}: {stats['rate']:.1%} "
                  f"({stats['agreed']}/{stats['total']})")

        if results["disagreements"]:
            print(f"\n  ❌ 分歧详情（前 5 条）:")
            for d in results["disagreements"][:5]:
                print(f"    Q: {d['question']}")
                print(f"    A: {d['answer']}")
                print(f"    expect={d['expected']} actual={d['actual']} ({d['reason'][:60]})")
                print()

        verdict = "✅ 通过 (≥90%)" if results.get("agreement_rate", 0) >= 0.90 else "⚠️ 需调整"
        print(f"  结论: {verdict}")
        print(f"{'='*60}\n")
