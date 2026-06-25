"""
投资顾问 Agent：
1. rule_engine 加载 rules.json → 硬性量化筛查（代码执行）
2. Judge 处理无法量化的定性部分（护城河、管理层等）
3. 关键指标缺失 >50% → verdict = "INSUFFICIENT_DATA"
4. 输出完整投资报告
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from rule_engine import RuleEngine, RuleResult
from llm_client import ModelPool, LLMClient

logger = logging.getLogger(__name__)


@dataclass
class InvestmentReport:
    """投资分析完整报告"""
    target: str = ""
    framework_source: str = ""
    overall_verdict: str = "INSUFFICIENT_DATA"  # BUY|HOLD|AVOID|INSUFFICIENT_DATA
    score: Optional[int] = None
    rule_results: List[RuleResult] = field(default_factory=list)
    qualitative_analysis: str = ""
    matched_criteria: List[str] = field(default_factory=list)
    triggered_red_flags: List[str] = field(default_factory=list)
    valuation_summary: str = ""
    missing_metrics: List[str] = field(default_factory=list)
    data_as_of: str = ""
    data_source: str = ""
    confidence: float = 1.0
    disclaimer: str = "本分析仅为书中框架的机械应用，不构成投资建议"


class InvestmentAdvisor:
    """
    投资顾问：机械执行书中框架，不做主观判断。
    量化规则用代码执行（零幻觉），定性分析用 Judge LLM。
    """

    def __init__(self, skill_dir: str | Path):
        self.skill_dir = Path(skill_dir)
        rules_path = self.skill_dir / "rules.json"
        self.engine = RuleEngine(rules_path)

        # 加载框架元数据
        self.strategy_type = self.engine.rules.get("strategy_type", "unknown")
        self.book_name = self.engine.rules.get("book", "unknown")

        # 初始化 Judge 用于定性分析
        self.pool = ModelPool()
        self.judge: LLMClient = self.pool.get("judge")

        # 加载 Skill 文件供定性分析参考
        self._load_skill_context()

    def _load_skill_context(self):
        """加载 Skill 文件内容作为定性分析上下文"""
        self.skill_context = ""

        # 尝试加载 SKILL.md
        skill_md = self.skill_dir / "SKILL.md"
        if skill_md.exists():
            self.skill_context += f"\n## 投资框架\n{skill_md.read_text(encoding='utf-8')[:4000]}"

        # 尝试加载 criteria.md
        criteria_md = self.skill_dir / "criteria.md"
        if criteria_md.exists():
            self.skill_context += f"\n## 选股标准\n{criteria_md.read_text(encoding='utf-8')[:2000]}"

        # 尝试加载 red_flags.md
        red_flags_md = self.skill_dir / "red_flags.md"
        if red_flags_md.exists():
            self.skill_context += f"\n## 风险信号\n{red_flags_md.read_text(encoding='utf-8')[:2000]}"

    def analyze(
        self, target: str, metrics: Dict[str, Any], context: str = ""
    ) -> InvestmentReport:
        """
        对目标标的进行投资分析。

        Args:
            target: 目标名称/代码（如 "贵州茅台 (600519)"）
            metrics: 财务指标字典（如 {"PE": 30, "PB": 8, ...}）
            context: 额外上下文信息（公司业务、行业等）

        Returns:
            InvestmentReport: 完整分析报告
        """
        # 标准化 metrics：确保都是 float 或 None
        std_metrics: Dict[str, Optional[float]] = {}
        for k, v in metrics.items():
            if v is None:
                std_metrics[k] = None
            elif isinstance(v, (int, float)):
                std_metrics[k] = float(v)
            else:
                try:
                    std_metrics[k] = float(v)
                except (ValueError, TypeError):
                    std_metrics[k] = None  # 无法转换的标记为缺失

        # Step 1: 量化规则执行（代码比对，零 LLM）
        rule_results = self.engine.evaluate(std_metrics)
        missing_ratio = self.engine.missing_metrics_ratio(rule_results)

        # 收集缺失指标
        missing_metrics = [
            r.rule_id for r in rule_results
            if r.passed is None
        ]

        # 无量化规则 → 跳过量化层，直接进入定性分析
        has_rules = len(rule_results) > 0

        # Step 2: 关键指标缺失 >50% 且 有规则可执行 → INSUFFICIENT_DATA
        if has_rules and missing_ratio > 0.5:
            return InvestmentReport(
                target=target,
                framework_source=self.book_name,
                overall_verdict="INSUFFICIENT_DATA",
                score=None,
                rule_results=rule_results,
                missing_metrics=missing_metrics,
                data_as_of=datetime.now().strftime("%Y-%m-%d"),
                data_source="用户输入",
                confidence=1.0 - missing_ratio,
                matched_criteria=[
                    r.rule_id for r in rule_results if r.passed is True
                ],
                triggered_red_flags=[
                    r.rule_id for r in rule_results if r.passed is False
                ],
            )

        # Step 3: Judge 处理定性部分
        qualitative_analysis = self._qualitative_analysis(target, context, std_metrics)

        # Step 4: 汇总评分
        score = self._calculate_score(rule_results, missing_ratio)

        # Step 5: 生成综合结论
        if has_rules:
            verdict = self._determine_verdict(rule_results, missing_ratio, context)
        else:
            # 无量化规则，由 Judge 定性判断
            verdict = self._qualitative_verdict(qualitative_analysis)

        # 收集匹配和触发的规则
        matched_criteria = [
            f"{r.rule_id} ({r.reason.split('→')[0].strip() if '→' in r.reason else '通过'})"
            for r in rule_results if r.passed is True
        ]
        triggered_red_flags = [
            f"{r.rule_id} ({r.reason.split('→')[0].strip() if '→' in r.reason else '不通过'})"
            for r in rule_results if r.passed is False
        ]

        # 生成估值摘要
        valuation_summary = self._generate_valuation_summary(
            verdict, score, rule_results, missing_ratio, has_rules
        )

        # 构建置信度
        if has_rules:
            confidence = max(0.0, min(1.0, 1.0 - missing_ratio * 0.7))
        else:
            confidence = 0.60  # 纯定性分析，置信度有限

        return InvestmentReport(
            target=target,
            framework_source=self.book_name,
            overall_verdict=verdict,
            score=score,
            rule_results=rule_results,
            qualitative_analysis=qualitative_analysis,
            matched_criteria=matched_criteria,
            triggered_red_flags=triggered_red_flags,
            valuation_summary=valuation_summary,
            missing_metrics=missing_metrics,
            data_as_of=datetime.now().strftime("%Y-%m-%d"),
            data_source="用户输入",
            confidence=confidence,
        )

    def _qualitative_analysis(
        self, target: str, context: str, metrics: Dict[str, Optional[float]]
    ) -> str:
        """
        Judge LLM 处理定性部分：
        - 护城河分析（品牌/技术/网络效应/规模等）
        - 管理层评价
        - 行业地位
        - 无法量化的风险
        """
        # 构建上下文
        metrics_str = "\n".join(
            f"  {k}: {v if v is not None else '缺失'}"
            for k, v in sorted(metrics.items())
        )

        prompt = f"""你是一位投资分析师，请根据以下投资框架对目标公司进行定性分析。

投资框架（来源：{self.book_name}，流派：{self.strategy_type}）：
{self.skill_context[:3000]}

目标公司：{target}

可用量化指标：
{metrics_str}

额外信息：
{context if context else '无'}

请从以下维度进行分析（每个维度简短回答）：
1. 护城河（品牌/技术/网络效应/转换成本/规模优势）
2. 管理层质量
3. 行业地位与竞争格局
4. 无法量化的风险点
5. 整体定性评价（用框架视角）

输出简洁的 Markdown 格式分析，总字数控制在 500 字以内。"""

        try:
            return self.judge.complete(prompt, max_tokens=1000, temperature=0.0)
        except Exception as e:
            logger.warning(f"定性分析失败: {e}")
            return "定性分析暂不可用（Judge 调用失败）"

    def _calculate_score(
        self, rule_results: List[RuleResult], missing_ratio: float
    ) -> Optional[int]:
        """
        计算综合评分（0-100）。
        只对可用规则打分，数据缺失的不计入。
        """
        valid_results = [r for r in rule_results if r.passed is not None]
        if not valid_results:
            return None

        passed = sum(1 for r in valid_results if r.passed)
        base_score = (passed / len(valid_results)) * 100

        # 数据缺失惩罚
        penalty = missing_ratio * 30  # 最多扣 30 分
        final_score = max(0, min(100, base_score - penalty))

        return round(final_score)

    def _determine_verdict(
        self,
        rule_results: List[RuleResult],
        missing_ratio: float,
        context: str,
    ) -> str:
        """基于规则执行结果判断 BUY/HOLD/AVOID"""
        valid_results = [r for r in rule_results if r.passed is not None]
        if not valid_results:
            return "INSUFFICIENT_DATA"

        passed = sum(1 for r in valid_results if r.passed)
        failed = sum(1 for r in valid_results if r.passed is False)
        pass_rate = passed / len(valid_results)

        # 严格标准：
        # 全部通过 + 数据缺失 < 20% → BUY
        # 通过率 ≥ 80% → HOLD
        # 通过率 ≥ 60% → HOLD（偏弱）
        # 通过率 < 60% 或任何关键风险触发 → AVOID

        if pass_rate >= 0.95 and missing_ratio < 0.2:
            return "BUY"
        elif pass_rate >= 0.80:
            return "HOLD"
        elif pass_rate >= 0.60 and missing_ratio < 0.4:
            return "HOLD"
        else:
            return "AVOID"

    def _qualitative_verdict(self, qualitative_analysis: str) -> str:
        """从定性分析文字中提取 Judge 的判断倾向"""
        analysis_lower = qualitative_analysis.lower()
        # 关键词判断
        buy_signals = ["买入", "buy", "符合框架", "优势明显", "极佳"]
        avoid_signals = ["回避", "avoid", "不符合", "风险过高", "严重超出", "致命"]
        hold_count = sum(1 for s in buy_signals if s in analysis_lower)
        avoid_count = sum(1 for s in avoid_signals if s in analysis_lower)
        if avoid_count > hold_count:
            return "AVOID"
        elif hold_count > avoid_count:
            return "BUY"
        return "HOLD"

    def _generate_valuation_summary(
        self,
        verdict: str,
        score: Optional[int],
        rule_results: List[RuleResult],
        missing_ratio: float,
        has_rules: bool = True,
    ) -> str:
        """生成估值摘要文字"""
        if not has_rules:
            return "该框架无量化规则，结论基于 Judge 定性分析。"
            
        valid = [r for r in rule_results if r.passed is not None]
        passed = sum(1 for r in valid if r.passed)
        failed = sum(1 for r in valid if r.passed is False)
        missing = sum(1 for r in rule_results if r.passed is None)

        parts = [
            f"共执行 {len(rule_results)} 条规则：✅ {passed} 条通过，"
            f"❌ {failed} 条不通过，⚠️ {missing} 条数据缺失不可判断。"
        ]

        if verdict == "BUY":
            parts.append("标的符合框架全部量化要求，可考虑买入。")
        elif verdict == "HOLD":
            parts.append("标的部分符合框架要求，但存在一定风险或数据不足，建议观望等待更好时机。")
        elif verdict == "AVOID":
            parts.append("标的未通过多项框架量化标准，建议回避。")
        else:
            parts.append("关键数据不足，无法给出明确结论，请补充更多指标数据。")

        if missing_ratio > 0:
            parts.append(f"数据完整度：{1 - missing_ratio:.0%}，结论置信度受限。")

        return " ".join(parts)
