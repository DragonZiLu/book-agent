"""
批量跑帕伯莱 Top 10 演讲 + 聚合输出
用法: source .env && python batch_pabrai.py
"""
import sys
import json
import time
import subprocess
from pathlib import Path

from config import PABRAI_TRANSCRIPTS_DIR

TRANSCRIPTS_DIR = Path(PABRAI_TRANSCRIPTS_DIR)
OUTPUT_ROOT = Path("./output/pabrai")

FILES = [
    ("harvard",       "20231018_mohnish_pabrais_q_a_with_students_at_the_harvard_business_school_on_sep_15_2023.pdf"),
    ("columbia_uno",  "20250611_mohnish_pabrais_sessions_at_uno_on_may_2_2025_and_columbia_business_school_on_march_25_2025_v2.pdf"),
    ("oxford",        "20231228_mohnish_pabrais_q_a_with_oxford_university_-_oxford_alpha_fund_on_november_21_2023.pdf"),
    ("lse",           "20240219_mohnish_pabrais_q_a_with_london_school_of_economics_-_value_investing_society_on_january_30_2024.pdf"),
    ("nebraska",      "20240620_mohnish_pabrais_session_at_the_university_of_nebraska_omaha_on_may_3_2024_v2.pdf"),
    ("cambridge",     "20250214_mohnish_pabrais_session_with_cibs_at_university_of_cambridge_on_january_31_2025.pdf"),
    ("boston_college","20231211_mohnish_pabrais_q_a_with_students_at_the_boston_college_carroll_school_of_management_on_oct_12_2023.pdf"),
    ("flame",         "20240212_mohnish_pabrais_presentation_and_q_a_at_the_flame_university_on_december_25_2023.pdf"),
    ("notre_dame",    "20231020_mohnish_pabrais_q_a_at_the_mendoza_college_of_business_notre_dame_on_september_29_2023.pdf"),
    ("sumzero",       "20240308_mohnish_pabrais_session_at_the_sumzero_virtual_investor_summit_2024_on_february_8_2024_v2.pdf"),
    ("peking",        "mohnish_pabrai_lecture_at_peking_university__guanghua_school_of_mgmt__on_dec_22_2017_v2.pdf"),
    ("mit",           "20240424_mohnish_pabrais_session_with_mits_brass_rat_investments_on_march_12_2024.pdf"),
]


def run_one(label: str, filename: str, i: int, total: int) -> dict:
    """跑单个 PDF，显示实时输出"""
    pdf_path = TRANSCRIPTS_DIR / filename
    out_dir = OUTPUT_ROOT / label
    skill_md = out_dir / "skill" / "SKILL.md"

    if skill_md.exists():
        print(f"\n[{i}/{total}] ⏭️  {label}: 已完成，跳过")
        report_path = out_dir / "reports" / "quality_report.json"
        if report_path.exists():
            with open(report_path) as f:
                return json.load(f)
        return {"skipped": True, "label": label}

    print(f"\n{'─'*60}")
    print(f"[{i}/{total}] 🏃 {label}")
    print(f"{'─'*60}")
    t0 = time.time()

    result = subprocess.run(
        ["python", "book_agent.py", "extract", str(pdf_path), "--output", str(out_dir)],
        timeout=900,
        env={**__import__('os').environ, "SKIP_EVAL": "1"},
    )

    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  ❌ 失败 (exit={result.returncode}, {elapsed:.0f}s)")
        return {"error": f"exit={result.returncode}", "label": label, "elapsed": elapsed}

    print(f"\n  ✅ 完成 ({elapsed:.0f}s)")

    report_path = out_dir / "reports" / "quality_report.json"
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
            print(f"     L1={report.get('l1_token_coverage',0):.0%} "
                  f"L2={report.get('l2_dimension_recall',0):.0%} "
                  f"L3={report.get('l3_qa_score',0):.0%}")
            return report
    return {"done": True, "label": label}


def aggregate():
    """聚合所有结果"""
    print(f"\n{'='*60}")
    print("📊 聚合框架...")
    print(f"{'='*60}")

    all_criteria = []
    all_flags = []
    all_valuation = []
    all_checklists = []

    reports = []
    count = 0
    for label, fname in FILES:
        out_dir = OUTPUT_ROOT / label
        skill_md = out_dir / "skill" / "SKILL.md"
        if not skill_md.exists():
            continue
        count += 1

        for dim, filename, collector in [
            ("criteria", "criteria.md", all_criteria),
            ("red_flags", "red_flags.md", all_flags),
            ("valuation", "valuation.md", all_valuation),
            ("checklists", "checklists.md", all_checklists),
        ]:
            f = out_dir / "skill" / filename
            if f.exists():
                collector.append((label, f.read_text(encoding="utf-8")))

        report_path = out_dir / "reports" / "quality_report.json"
        if report_path.exists():
            with open(report_path) as f:
                reports.append((label, json.load(f)))

    # 去重
    def dedup(items):
        seen = set()
        result = []
        for label, text in items:
            for line in text.split("\n"):
                line = line.strip().lstrip("-*• 1234567890. ")
                if len(line) > 8:
                    fp = line[:60].lower()
                    if fp not in seen:
                        seen.add(fp)
                        result.append(line)
        return result

    uc = dedup(all_criteria)
    uf = dedup(all_flags)
    uv = dedup(all_valuation)
    ul = dedup(all_checklists)

    master = OUTPUT_ROOT / "master"
    master.mkdir(parents=True, exist_ok=True)

    (master / "criteria.md").write_text(
        f"# Pabrai 选股标准（{count}场演讲去重）\n\n" + "\n".join(f"- {c}" for c in uc), encoding="utf-8")
    (master / "red_flags.md").write_text(
        f"# Pabrai 风险信号（{count}场演讲去重）\n\n" + "\n".join(f"- {r}" for r in uf), encoding="utf-8")
    (master / "valuation.md").write_text(
        f"# Pabrai 估值方法（{count}场演讲去重）\n\n" + "\n".join(f"- {v}" for v in uv), encoding="utf-8")
    (master / "checklists.md").write_text(
        f"# Pabrai 检查清单（{count}场演讲去重）\n\n" + "\n".join(f"- {c}" for c in ul), encoding="utf-8")

    # 生成 SKILL.md
    skill_text = f"""# Mohnish Pabrai 投资框架（{count}场演讲综合）

## 流派
Value Investing / Cloning / Concentrated Portfolio

## 核心哲学
Pabrai 的投资体系围绕「Heads I Win, Tails I Don't Lose Much」的非对称下注展开：
- 克隆巴菲特/芒格已做功课的标的
- 少下注、大下注、低频交易
- 在市场恐慌时买入被遗弃的优质资产
- 寻找清算价值远低于市值的极端安全边际
- 长期持有，利用复利的力量

## 选股标准 ({len(uc)}条)
{chr(10).join(f'- {c}' for c in uc[:30])}

## 风险信号 ({len(uf)}条)
{chr(10).join(f'- {r}' for r in uf[:30])}

## 估值方法 ({len(uv)}条)
{chr(10).join(f'- {v}' for v in uv[:15])}

## 检查清单 ({len(ul)}条)
{chr(10).join(f'- {c}' for c in ul[:20])}
"""
    (master / "SKILL.md").write_text(skill_text, encoding="utf-8")

    # 生成 rules.json（空但有 book 元信息）
    (master / "rules.json").write_text(json.dumps({
        "strategy_type": "value",
        "book": f"Pabrai Speeches ({count} lectures)"
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "total_lectures": count,
        "unique_criteria": len(uc),
        "unique_red_flags": len(uf),
        "unique_valuation_methods": len(uv),
        "unique_checklist_items": len(ul),
        "individual_reports": [
            {"label": l, "l1": r.get("l1_token_coverage", 0),
             "l2": r.get("l2_dimension_recall", 0),
             "l3": r.get("l3_qa_score", 0),
             "overall": r.get("overall_score", 0)}
            for l, r in reports
        ],
    }
    (master / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n📦 聚合完成: {count}场演讲")
    print(f"   选股标准: {len(uc)} | 风险信号: {len(uf)} | 估值方法: {len(uv)} | 检查清单: {len(ul)}")
    print(f"\n   使用: python book_agent.py advise ./output/pabrai/master 标的名 --pe 15 --pb 2")


if __name__ == "__main__":
    total = len(FILES)
    print(f"🚀 Pabrai 批量提取: {total} 场演讲")
    t_start = time.time()

    for i, (label, fname) in enumerate(FILES):
        run_one(label, fname, i + 1, total)

    elapsed = time.time() - t_start
    print(f"\n⏱️  总耗时: {elapsed/60:.1f} 分钟")
    aggregate()
