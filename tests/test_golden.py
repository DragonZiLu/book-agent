"""
黄金测试集回归脚本
每次修改 prompt 后运行，确认 evaluator 未退化。

用法：
  python -m tests.test_golden            # 仅结构校验（离线）
  python -m tests.test_golden --calibrate # 运行 LLM 自洽校准（需 API Key）
"""

import json
import os
import sys
import argparse
from pathlib import Path

# 将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import validate_role_config, ROLE_MODELS, PROVIDER_CONFIG


def run_golden_tests(calibrate: bool = False):
    """加载黄金测试集并运行验证；--calibrate 时运行 LLM 判分回归。"""
    golden_path = Path(__file__).parent / "golden" / "intelligent_investor_qa.json"

    if not golden_path.exists():
        print(f"黄金测试集不存在: {golden_path}")
        return False

    with open(golden_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"加载 {len(questions)} 道黄金测试题")
    print()

    # 统计
    by_type = {"numeric": 0, "direction": 0, "factual": 0}
    for q in questions:
        by_type[q["answer_type"]] += 1

    print("题型分布:")
    for t, count in by_type.items():
        print(f"  {t}: {count}")
    print()

    # 验证每道题的结构完整性
    errors = []
    for i, q in enumerate(questions):
        required_fields = ["question", "gold_answer", "evidence_span", "answer_type"]
        for field in required_fields:
            if field not in q or not q[field]:
                errors.append(f"  第{i+1}题缺失字段: {field}")

        if q["answer_type"] not in ("numeric", "direction", "factual"):
            errors.append(f"  第{i+1}题无效的 answer_type: {q['answer_type']}")

    if errors:
        print("结构错误:")
        for e in errors:
            print(e)
        return False

    print("✅ 所有题结构验证通过")
    print()

    # 验证角色配置
    warnings = validate_role_config()
    if warnings:
        print("角色配置警告:")
        for w in warnings:
            print(f"  {w}")
    else:
        print("✅ 角色配置无警告")

    print()
    print("黄金测试集结构就绪。")
    if not calibrate:
        print("提示: 加 --calibrate 可运行 LLM 判分回归（需配置对应厂商 API Key）。")
        print("运行完整提取验证: python book_agent.py extract <pdf_path>")
        return True

    # ── LLM 自洽校准 ──
    return _run_calibration(golden_path)


def _run_calibration(golden_path: Path) -> bool:
    """运行 EvaluatorCalibrator 自洽校准，检测 prompt 退化。"""
    judge_cfg = ROLE_MODELS["judge"]
    provider = judge_cfg["provider"]
    env_key = PROVIDER_CONFIG.get(provider, {}).get("env_key", "")
    api_key = os.getenv(env_key, "")

    if not api_key:
        print(f"⚠️ 未检测到 {provider} 的 API Key（环境变量 {env_key}），跳过校准。")
        print("   配置后重新运行: python -m tests.test_golden --calibrate")
        return True

    from llm_client import ModelPool
    from evaluator_v2 import EvaluatorCalibrator

    print("🔍 运行 LLM 判分校准（自洽回归）...")
    pool = ModelPool()
    calibrator = EvaluatorCalibrator(pool)
    results = calibrator.run_calibration(str(golden_path))
    calibrator.print_report(results)

    # 一致率 < 90% 视为未通过
    rate = results.get("agreement_rate", 0)
    return rate >= 0.90


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="黄金测试集回归")
    parser.add_argument(
        "--calibrate", action="store_true",
        help="运行 LLM 判分校准（需 API Key），检测 prompt 退化",
    )
    args = parser.parse_args()
    success = run_golden_tests(calibrate=args.calibrate)
    sys.exit(0 if success else 1)
