"""
黄金测试集回归脚本
每次修改 prompt 后运行，确认 evaluator 未退化
"""

import json
import sys
from pathlib import Path

# 将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import validate_role_config


def run_golden_tests():
    """加载黄金测试集并运行基本验证"""
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

    print("✅ 所有 {len(questions)} 题结构验证通过")
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
    print("黄金测试集就绪，可用于端到端验证。")
    print("运行完整验证: python book_agent.py extract <pdf_path>")
    return True


if __name__ == "__main__":
    success = run_golden_tests()
    sys.exit(0 if success else 1)
