"""
主入口 CLI：extract / advise 两个子命令
完整 Pipeline：提取 → 质量评估 → 反馈闭环重试 → 投资顾问
"""

import sys
import os
import json
import time
import logging
from pathlib import Path
from typing import Optional

from config import (
    MAX_COST_USD,
    MAX_RETRIES,
    USE_EVALUATOR_V2,
    validate_role_config,
)
from llm_client import ModelPool
from cache import ExtractionCache
from extractor import BookExtractor, ExtractionResult
from summarizer import BookSummarizer
from evaluator import QualityEvaluator as QualityEvaluatorV1, QualityReport
from rule_engine import RuleEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("book_agent")


# ============ extract 子命令 ============

def cmd_extract(source: str, output_dir: str = "./output"):
    """
    完整提取 Pipeline：输入书籍 → 输出 Skill 文件 + rules.json + 质量报告。
    含成本熔断 + 带反馈重试（最多 2 次）。
    """
    output_path = Path(output_dir)
    skill_dir = output_path / "skill"
    reports_dir = output_path / "reports"
    skill_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 初始化组件
    pool = ModelPool()

    # 打印角色配置警告
    for w in validate_role_config():
        logger.warning(w)

    # Step 1: 文本提取
    logger.info("━" * 50)
    logger.info(f"📖 Step 1/3: 文本提取")
    logger.info(f"   源文件: {source}")
    t0 = time.time()
    extractor = BookExtractor()
    chapters_result = extractor.extract(source)
    chapters = chapters_result.chapters
    logger.info(f"   ✅ 提取 {len(chapters)} 个章节, 耗时 {time.time()-t0:.1f}s")
    for w in chapters_result.warnings:
        logger.warning(f"   ⚠️  {w}")

    # Step 2-3: 提取 + 评估（带反馈闭环）
    summarizer = BookSummarizer(pool)
    if USE_EVALUATOR_V2:
        from evaluator_v2 import QualityEvaluator as QualityEvaluatorV2
        from config import MAX_EVAL_WORKERS
        evaluator = QualityEvaluatorV2(pool, max_workers=MAX_EVAL_WORKERS)
        logger.info("   评估器: V2 (信息密度L1 + 真召回L2 + 闭卷L3)")
    else:
        evaluator = QualityEvaluatorV1(pool)
        logger.info("   评估器: V1 (兼容模式)")
    cache = ExtractionCache()

    report: Optional[QualityReport] = None
    result = None
    feedbacks = None

    for attempt in range(MAX_RETRIES + 1):
        # 成本熔断检查
        current_cost = pool.total_cost_sum()
        if current_cost > MAX_COST_USD:
            logger.warning(f"⚠️ 成本超限 ${current_cost:.2f} > ${MAX_COST_USD}，强制停止")
            if result:
                save_output(result, report, skill_dir, reports_dir)
            _print_cost_report(pool, report)
            return

        round_t0 = time.time()
        logger.info(f"\n{'━'*50}")
        logger.info(f"🔄 第 {attempt + 1}/{MAX_RETRIES + 1} 轮")
        logger.info(f"   累计成本: ${pool.total_cost_sum():.4f}")

        # Step 2: 执行提取（带缓存 + 反馈）
        logger.info(f"📝 Step 2/3: 投资框架提取")
        result = summarizer.summarize(
            chapters, feedbacks=feedbacks, cache=cache, book_name=Path(source).stem
        )

        # Step 3: 三层质量评估（SKIP_EVAL=1 则跳过）
        if os.environ.get("SKIP_EVAL") == "1":
            logger.info(f"⏩ Step 3/3: 跳过评估 (SKIP_EVAL=1)")
            save_output(result, None, skill_dir, reports_dir)
            _print_cost_report(pool, report)
            return

        logger.info(f"📊 Step 3/3: 质量评估")
        strategy_type = getattr(summarizer, "_strategy_type", "general")
        if USE_EVALUATOR_V2:
            report = evaluator.evaluate(
                chapters, result, chapters_result.warnings, strategy_type=strategy_type
            )
        else:
            report = evaluator.evaluate(chapters, result, chapters_result.warnings)

        round_elapsed = time.time() - round_t0
        logger.info(f"\n  ═══ 本轮结果 ({round_elapsed:.1f}s) ═══")
        logger.info(
            f"  L1={report.l1_token_coverage:.0%} "
            f"L2={report.l2_dimension_recall:.0%} "
            f"L3={report.l3_qa_score:.0%} "
            f"→ 整体={report.overall_score:.2f}"
        )
        logger.info(f"  累计成本=${report.cost_usd:.4f}")
        if USE_EVALUATOR_V2:
            if getattr(report, 'l2_dim_scores', None):
                dims = report.l2_dim_scores
                logger.info(f"  L2维度: criteria={dims.get('selection_criteria',0):.0%} "
                           f"flags={dims.get('red_flags',0):.0%} "
                           f"valuation={dims.get('valuation_methods',0):.0%} "
                           f"check={dims.get('checklists',0):.0%}")
            if getattr(report, 'l3_ci', None):
                ci = report.l3_ci
                logger.info(f"  L3 CI(90%): [{ci[0]:.1%}, {ci[1]:.1%}]")
            if getattr(report, 'veto_reason', ''):
                logger.warning(f"  ❌ 硬否决: {report.veto_reason}")

        if report.passed:
            logger.info(f"\n✅ 质量达标！({round_elapsed:.1f}s)")
            save_output(result, report, skill_dir, reports_dir)
            _print_cost_report(pool, report)
            return

        # 未达标，准备重试
        feedbacks = report.feedbacks
        if feedbacks:
            logger.info(f"\n  ⚠️  薄弱章节: {report.weak_chapters}")
            for fb in feedbacks:
                missing_list = getattr(fb, 'missing_concepts', []) or fb.missing_dimensions
                logger.info(f"     第{fb.index}章: {', '.join(missing_list[:3])}")
            logger.info(f"  携带 {len(feedbacks)} 条反馈进入重试...")
        else:
            logger.warning("  无具体反馈, 仍重试...")

        if attempt == MAX_RETRIES:
            logger.warning(f"\n⚠️ 已达最大重试次数，保存当前结果（未达标）")
            save_output(result, report, skill_dir, reports_dir)

    _print_cost_report(pool, report)


def save_output(result, report, skill_dir: Path, reports_dir: Path):
    """保存提取结果到 output 目录"""
    if result is None:
        logger.error("无提取结果可保存")
        return

    # 保存 SKILL.md
    (skill_dir / "SKILL.md").write_text(result.skill_md, encoding="utf-8")
    logger.info(f"  保存 SKILL.md ({len(result.skill_md)} 字符)")

    # 保存分章节摘要
    chapters_dir = skill_dir / "chapters"
    chapters_dir.mkdir(exist_ok=True)
    for cs in result.chapter_summaries:
        ch_file = chapters_dir / f"ch{cs.chapter_index:02d}_{_safe_filename(cs.title)}.json"
        ch_file.write_text(
            json.dumps(cs.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # 保存 criteria.md
    (skill_dir / "criteria.md").write_text(result.criteria, encoding="utf-8")
    logger.info(f"  保存 criteria.md")

    # 保存 red_flags.md
    (skill_dir / "red_flags.md").write_text(result.red_flags, encoding="utf-8")
    logger.info(f"  保存 red_flags.md")

    # 保存 valuation.md
    (skill_dir / "valuation.md").write_text(result.valuation, encoding="utf-8")
    logger.info(f"  保存 valuation.md")

    # 保存 checklists.md
    (skill_dir / "checklists.md").write_text(result.checklists, encoding="utf-8")
    logger.info(f"  保存 checklists.md")

    # 保存 rules.json
    rules_path = skill_dir / "rules.json"
    rules_path.write_text(
        json.dumps(result.rules_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"  保存 rules.json")

    # 保存 quality_report.json
    if report:
        report_path = reports_dir / "quality_report.json"
        report_data = {
            "strategy_type": result.rules_json.get("strategy_type", "unknown"),
            "extraction_warnings": report.extraction_warnings,
            "l1_token_coverage": report.l1_token_coverage,
            "l2_dimension_recall": report.l2_dimension_recall,
            "l3_qa_score": report.l3_qa_score,
            "l3_breakdown": report.l3_breakdown,
            "overall_score": report.overall_score,
            "passed": report.passed,
            "chapter_scores": report.chapter_scores,
            "weak_chapters": report.weak_chapters,
            "total_cost_usd": report.cost_usd,
        }
        # V2 额外字段
        if USE_EVALUATOR_V2:
            report_data["l1_chapter_scores"] = [
                {"index": cs.index, "density": cs.density,
                 "structure": cs.structure, "entity_retention": cs.entity_retention,
                 "overall": cs.overall}
                for cs in getattr(report, "l1_chapter_scores", [])
            ]
            report_data["l2_dim_scores"] = getattr(report, "l2_dim_scores", {})
            report_data["l2_missed_concepts"] = {
                str(k): v for k, v in getattr(report, "l2_missed_concepts", {}).items()
            }
            if getattr(report, "l3_ci", None):
                report_data["l3_ci"] = list(report.l3_ci)
            if getattr(report, "veto_reason", ""):
                report_data["veto_reason"] = report.veto_reason
        report_path.write_text(
            json.dumps(report_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"  保存 quality_report.json")


# ============ advise 子命令 ============

def cmd_advise(
    skill_dir: str,
    target: str,
    metrics: dict = None,
    context: str = "",
    data_as_of: str = None,
):
    """
    投资顾问：基于提取的框架对目标标的进行分析。
    用法: python book_agent.py advise ./output/skill 贵州茅台 --pe 30 --pb 8 --as-of 2024-Q3
    """
    from investment_advisor import InvestmentAdvisor

    skill_path = Path(skill_dir)
    rules_path = skill_path / "rules.json"

    if not rules_path.exists():
        logger.error(f"rules.json 不存在: {rules_path}，请先运行 extract")
        return

    if metrics is None:
        metrics = {}

    advisor = InvestmentAdvisor(skill_path)
    report = advisor.analyze(target, metrics, context, data_as_of=data_as_of)

    # 输出报告
    print(format_investment_report(report))


def format_investment_report(report) -> str:
    """格式化投资报告为人类可读文本"""
    from investment_advisor import InvestmentReport

    verdict_emoji = {
        "BUY": "🟢",
        "HOLD": "🟡",
        "AVOID": "🔴",
        "INSUFFICIENT_DATA": "⚪",
    }
    emoji = verdict_emoji.get(report.overall_verdict, "❓")

    lines = [
        f"{'='*60}",
        f"目标：{report.target} | 框架源：{report.framework_source}",
        f"",
        f"{emoji} 总评：{report.overall_verdict} | 评分：{report.score or 'N/A'}/100",
        f"置信度：{report.confidence:.2f} | 数据时点：{report.data_as_of}",
        f"",
        f"── 量化规则（代码执行）{'─'*36}",
    ]

    for r in report.rule_results:
        if r.passed is True:
            status = "✅"
        elif r.passed is False:
            status = "❌"
        else:
            status = "⚠️"
        extra = f" ← {r.reason.split('→')[1]}" if r.reason and "→" in r.reason else ""
        lines.append(
            f"{status} {r.rule_id:<18s} 阈值={r.threshold:<8} "
            f"实际={'N/A' if r.input_value is None else r.input_value:<8} "
            f"[{r.source_chapter}]{extra}"
        )

    if report.qualitative_analysis:
        lines.append(f"")
        lines.append(f"── 定性分析（Judge）{'─'*41}")
        lines.append(report.qualitative_analysis)

    if report.missing_metrics:
        lines.append(f"")
        lines.append(f"── 缺失指标 {'─'*47}")
        lines.append(", ".join(report.missing_metrics))

    if report.matched_criteria:
        lines.append(f"")
        lines.append(f"── 匹配标准 {'─'*47}")
        lines.append("\n".join(f"  ✅ {c}" for c in report.matched_criteria[:10]))

    if report.triggered_red_flags:
        lines.append(f"")
        lines.append(f"── 触发风险信号 {'─'*44}")
        lines.append("\n".join(f"  🔴 {f}" for f in report.triggered_red_flags[:10]))

    lines.append(f"")
    lines.append(f"── 结论 {'─'*52}")
    lines.append(report.valuation_summary or "无")
    lines.append(f"")
    lines.append(f"⚠️ {report.disclaimer}")
    lines.append(f"{'='*60}")

    return "\n".join(lines)


# ============ CLI 入口 ============

def main():
    """CLI 主入口"""
    if len(sys.argv) < 2:
        print("用法:")
        print("  python book_agent.py extract <source> [--output ./output]")
        print("  python book_agent.py advise <skill_dir> <target> [--pe 30 --pb 8 ...]")
        print()
        print("示例:")
        print("  python book_agent.py extract ./intelligent_investor.pdf --output ./output")
        print("  python book_agent.py advise ./output/skill 贵州茅台 --pe 30 --pb 8 --as-of 2024-Q3 --context \"白酒龙头\"")
        return

    command = sys.argv[1]

    if command == "extract":
        # 解析参数
        source = sys.argv[2] if len(sys.argv) > 2 else None
        if not source:
            print("错误: 需要提供 source 参数")
            return

        output_dir = "./output"
        for i, arg in enumerate(sys.argv):
            if arg == "--output" and i + 1 < len(sys.argv):
                output_dir = sys.argv[i + 1]

        cmd_extract(source, output_dir)

    elif command == "advise":
        if len(sys.argv) < 4:
            print("错误: 需要 skill_dir 和 target 参数")
            print("示例: python book_agent.py advise ./output/skill 贵州茅台 --pe 30 --pb 8")
            return

        skill_dir = sys.argv[2]
        target = sys.argv[3]

        # 解析 --key value 指标 / --context / --as-of
        metrics = {}
        context = ""
        data_as_of = None
        i = 4
        while i < len(sys.argv):
            arg = sys.argv[i]
            if arg == "--context" and i + 1 < len(sys.argv):
                context = sys.argv[i + 1]
                i += 2
            elif arg == "--as-of" and i + 1 < len(sys.argv):
                data_as_of = sys.argv[i + 1]
                i += 2
            elif arg.startswith("--") and i + 1 < len(sys.argv):
                key = arg[2:]  # 去掉 --
                val = sys.argv[i + 1]
                try:
                    metrics[key] = float(val)
                except ValueError:
                    metrics[key] = val  # 非数字保留字符串
                i += 2
            else:
                i += 1

        cmd_advise(skill_dir, target, metrics, context, data_as_of=data_as_of)

    else:
        print(f"未知命令: {command}")
        print("可用命令: extract, advise")


def _print_cost_report(pool: ModelPool, report=None):
    """打印成本报告"""
    print()
    print(pool.report())
    if report:
        print(f"  最终质量: {'✅ 达标' if report.passed else '⚠️ 未达标'}")


def _safe_filename(title: str) -> str:
    """将标题转为安全的文件名"""
    import re
    # 保留中文、英文、数字、下划线、连字符
    safe = re.sub(r'[^\w\u4e00-\u9fff\-]', '_', title)
    return safe[:50]


if __name__ == "__main__":
    main()
