"""
量化规则引擎：加载 rules.json，执行数值比对
支持单值比较 / expr 表达式 / 数据缺失标记
"""

import json
import re
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RuleResult:
    """单条规则的执行结果"""
    rule_id: str
    passed: Optional[bool]  # True=通过, False=不通过, None=数据缺失
    input_value: Optional[float]
    threshold: float
    source_chapter: str
    reason: str = ""


class RuleEngine:
    """加载 rules.json，对输入指标执行全部量化规则"""

    # 运算符映射
    OP_MAP: Dict[str, str] = {
        "<=": "<=",
        "≥": ">=",
        ">=": ">=",
        "<": "<",
        ">": ">",
        "==": "==",
        "=": "==",
        "!=": "!=",
    }

    def __init__(self, rules_path: str | Path):
        self.rules_path = Path(rules_path)
        self.rules: Dict[str, Any] = {}
        self._load_rules()

    def _load_rules(self):
        """加载 rules.json"""
        if not self.rules_path.exists():
            logger.warning(f"rules.json 不存在: {self.rules_path}")
            return

        with open(self.rules_path, "r", encoding="utf-8") as f:
            self.rules = json.load(f)

        logger.info(
            f"加载规则: strategy={self.rules.get('strategy_type', '?')}, "
            f"book={self.rules.get('book', '?')}"
        )

    def evaluate(self, metrics: Dict[str, float | None]) -> List[RuleResult]:
        """
        执行全部规则。
        metrics: {"PE": 30, "PB": 8, "current_ratio": 2.1, ...}
        值可为 None 表示数据缺失。
        """
        results: List[RuleResult] = []

        # 估值规则
        for rule in self.rules.get("valuation_rules", []):
            result = self._evaluate_rule(rule, metrics)
            results.append(result)

        # 选股规则
        for rule in self.rules.get("selection_criteria", []):
            result = self._evaluate_rule(rule, metrics)
            results.append(result)

        # 风险规则
        for rule in self.rules.get("red_flag_rules", []):
            result = self._evaluate_rule(rule, metrics)
            results.append(result)

        return results

    def _evaluate_rule(
        self, rule: Dict[str, Any], metrics: Dict[str, float | None]
    ) -> RuleResult:
        """评估单条规则"""
        rule_id = rule.get("id", "unknown")
        source = rule.get("source", "?")
        threshold = rule.get("value", 0)

        # 支持表达式：expr = "PE * PB"
        expr = rule.get("expr")
        if expr:
            return self._evaluate_expr(rule_id, expr, rule, metrics, source)

        # 单指标比对
        metric = rule.get("metric", "")
        op_raw = rule.get("op", "<=")
        op = self.OP_MAP.get(op_raw, op_raw)

        input_val = metrics.get(metric)
        if input_val is None:
            return RuleResult(
                rule_id=rule_id,
                passed=None,
                input_value=None,
                threshold=threshold,
                source_chapter=source,
                reason=f"指标 {metric} 数据缺失",
            )

        passed = self._do_compare(input_val, op, threshold)
        reason = f"{metric}={input_val} {op} {threshold} → {'✅ 通过' if passed else '❌ 不通过'}"

        return RuleResult(
            rule_id=rule_id,
            passed=passed,
            input_value=float(input_val),
            threshold=threshold,
            source_chapter=source,
            reason=reason,
        )

    def _evaluate_expr(
        self, rule_id: str, expr: str, rule: Dict, metrics: Dict[str, float | None], source: str
    ) -> RuleResult:
        """评估表达式规则，如 PE * PB"""
        op_raw = rule.get("op", "<=")
        op = self.OP_MAP.get(op_raw, op_raw)
        threshold = rule.get("value", 0)

        # 提取表达式中的变量
        variables = re.findall(r"[A-Za-z_/]+", expr)
        expr_value = None

        try:
            # 替换变量为实际值
            eval_expr = expr
            all_found = True
            for var in variables:
                val = metrics.get(var)
                if val is None:
                    all_found = False
                    break
                eval_expr = eval_expr.replace(var, str(val))

            if all_found:
                expr_value = float(eval(eval_expr))
        except Exception as e:
            logger.warning(f"表达式计算失败: {expr} → {e}")
            expr_value = None

        if expr_value is None:
            return RuleResult(
                rule_id=rule_id,
                passed=None,
                input_value=None,
                threshold=threshold,
                source_chapter=source,
                reason=f"表达式 {expr} 中变量数据缺失",
            )

        passed = self._do_compare(expr_value, op, threshold)
        reason = f"{expr}={expr_value:.2f} {op} {threshold} → {'✅ 通过' if passed else '❌ 不通过'}"

        return RuleResult(
            rule_id=rule_id,
            passed=passed,
            input_value=expr_value,
            threshold=threshold,
            source_chapter=source,
            reason=reason,
        )

    @staticmethod
    def _do_compare(left: float, op: str, right: float) -> bool:
        """执行数值比较"""
        if op in ("<=", "<="):
            return left <= right
        elif op in ("≥", ">="):
            return left >= right
        elif op in ("<", "<"):
            return left < right
        elif op in (">", ">"):
            return left > right
        elif op in ("==", "="):
            return abs(left - right) < 0.001
        elif op == "!=":
            return abs(left - right) >= 0.001
        return False

    def missing_metrics_ratio(self, results: List[RuleResult]) -> float:
        """计算数据缺失的规则比例"""
        if not results:
            return 1.0
        missing = sum(1 for r in results if r.passed is None)
        return missing / len(results)

    def summary(self, results: List[RuleResult]) -> Dict[str, Any]:
        """汇总规则执行结果"""
        total = len(results)
        passed = sum(1 for r in results if r.passed is True)
        failed = sum(1 for r in results if r.passed is False)
        missing = sum(1 for r in results if r.passed is None)

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "missing": missing,
            "pass_rate": passed / total if total > 0 else 0.0,
            "missing_rate": missing / total if total > 0 else 0.0,
            "results": [
                {
                    "rule_id": r.rule_id,
                    "passed": r.passed,
                    "input_value": r.input_value,
                    "threshold": r.threshold,
                    "source": r.source_chapter,
                    "reason": r.reason,
                }
                for r in results
            ],
        }
