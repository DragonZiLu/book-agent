"""
定向提取帕伯莱「投资案例」——他本人实际操作的标的

两阶段策略：
  Phase 1: 规则扫描（0 LLM 成本）→ 识别含案例讨论的演讲
  Phase 2: 深度 LLM 提取 → 逐章调用，提取结构化案例信息
  → 跨演讲去重合并 → 输出 Markdown + JSON

用法：
    python extract_cases.py
    # 或指定子步骤：
    python extract_cases.py --phase1        # 仅扫描
    python extract_cases.py --phase2        # 仅深度提取（需先完成 Phase 1）
    python extract_cases.py --dedup         # 仅去重合并
    python extract_cases.py --output        # 仅生成输出文件
"""

import os
import sys
import re
import json
import time
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# 切到脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import CHAPTER_SUMMARY_TOKENS, PABRAI_TRANSCRIPTS_DIR

# 案例提取需要更大的 token 预算（多案例 JSON 更长）
CASE_EXTRACTION_TOKENS = 3000
from llm_client import ModelPool
from extractor import BookExtractor
from summarizer import INVESTMENT_CASES_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("extract_cases")

# ============ 配置 ============
TRANSCRIPTS_DIR = Path(PABRAI_TRANSCRIPTS_DIR)
OUTPUT_DIR = Path("output/pabrai/cases")
PROGRESS_FILE = OUTPUT_DIR / ".progress.json"

# Phase 1 关键词配置
CASE_SIGNAL_KEYWORDS = [
    # 买卖动作
    r'\bbought\b', r'\binvested\sin\b', r'\bposition\sin\b', r'\bpurchased\b',
    r'\bacquired\b', r'\bsold\b', r'\bexited\b', r'\bliquidated\b',
    r'\bentered\b', r'\binitiated\b', r'\baccumulated\b',
    # 结果信号
    r'\bIRR\b', r'\breturn\b', r'\bmade\s+\d', r'\blost\b', r'\bbagger\b',
    r'\bcompounded\b', r'\bCAGR\b', r'\bmultiple\b', r'\bupside\b',
    r'\bwrite.off\b', r'\bwrite.down\b', r'\bwipe.out\b',
    # 案例相关词
    r'\bcase\s+study\b', r'\bexample\b', r'\binvestment\b',
    r'\bportfolio\b', r'\bholding\b', r'\bstake\b',
    # 中文关键词（部分演讲含中文）
    r'投资了', r'买入', r'卖出', r'持有', r'收益率', r'亏损',
]

# 已知帕伯莱涉及过的公司名（辅助信号增强）
KNOWN_PABRAI_COMPANIES = [
    r'Rain\s+Industries', r'NHAI', r'Sun\s+Micro', r'Satyam',
    r'Delta\s+Financial', r'Tecumseh', r'Pinnacle\s+Airlines',
    r'Horsehead\s+Holding', r'Crimson\s+Wine', r'Banca\s+Monte',
    r'Reliance\s+Industries', r'Edelweiss', r'Tata\s+Motors',
    r'Sunteck\s+Realty', r'Shipping\s+Corp', r'Dishman',
    r'Repligen', r'Micron\s+Technology', r'Alphabet',
    r'Berkshire\s+Hathaway', r'Alibaba', r'Tile\s+Shop',
    r'Valeant', r'Fiat', r'Ferrari',
]

# Phase 1 每 PDF 读取的字符数
PHASE1_READ_CHARS = 5000


def load_progress() -> dict:
    """加载断点续跑进度"""
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    return {"phase1_done": False, "phase2_processed": [], "cases_raw": []}


def save_progress(progress: dict):
    """保存进度"""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


# ============ Phase 1: 规则扫描 ============

def scan_pdfs_for_cases() -> List[Path]:
    """
    遍历所有 PDF，用关键词匹配快速识别含案例讨论的演讲。
    返回命中 PDF 列表。
    """
    logger.info("=" * 60)
    logger.info("📡 Phase 1: 规则扫描 — 识别含案例的演讲")
    logger.info("=" * 60)

    pdf_files = sorted(TRANSCRIPTS_DIR.glob("*.pdf"))
    logger.info(f"共 {len(pdf_files)} 个 PDF 文件")

    hit_pdfs: List[Path] = []
    scan_results = {"total": len(pdf_files), "hits": [], "misses": [], "errors": []}

    for pdf_path in pdf_files:
        try:
            text = _read_pdf_head(pdf_path, PHASE1_READ_CHARS)
            if not text:
                scan_results["errors"].append({"file": pdf_path.name, "reason": "empty text"})
                continue

            score, matched_keywords = _score_for_cases(text)

            logger.info(f"  {'✅' if score >= 2 else '⬜'} [{score:2d}] {pdf_path.name[:60]} — {', '.join(matched_keywords[:3])}")

            if score >= 2:
                hit_pdfs.append(pdf_path)
                scan_results["hits"].append({
                    "file": pdf_path.name,
                    "score": score,
                    "matched_keywords": matched_keywords[:10],
                })
            else:
                scan_results["misses"].append(pdf_path.name)

        except Exception as e:
            logger.warning(f"  ❌ {pdf_path.name}: {e}")
            scan_results["errors"].append({"file": pdf_path.name, "reason": str(e)})

    # 保存扫描报告
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scan_report_path = OUTPUT_DIR / "scan_report.json"
    scan_report_path.write_text(json.dumps(scan_results, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"\n📊 扫描结果：{len(hit_pdfs)} 命中 / {len(pdf_files)} 总计")
    logger.info(f"   报告已保存: {scan_report_path}")

    return hit_pdfs


def _read_pdf_head(pdf_path: Path, max_chars: int) -> str:
    """读取 PDF 前 max_chars 个字符"""
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    text_parts = []
    total = 0
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text_parts.append(t)
            total += len(t)
            if total >= max_chars:
                break
    return " ".join(text_parts)[:max_chars]


def _score_for_cases(text: str) -> Tuple[int, List[str]]:
    """
    对文本打分：匹配信号关键词 + 已知公司名。
    返回 (score, matched_keywords)
    """
    text_lower = text.lower()
    matched = []

    # 匹配通用信号词
    for pattern in CASE_SIGNAL_KEYWORDS:
        if re.search(pattern, text_lower):
            matched.append(pattern.replace(r'\b', ''))

    # 匹配已知公司名（额外加分）
    company_hits = []
    for pattern in KNOWN_PABRAI_COMPANIES:
        if re.search(pattern, text):
            company_hits.append(pattern.replace(r'\s+', ' '))
            matched.append(f"🏢 {pattern.replace(r'\s+', ' ')}")

    # 基础分 = 信号词命中数，公司名命中额外 +2
    signal_count = len(matched) - len(company_hits)
    score = signal_count + len(company_hits) * 2

    return score, matched


# ============ Phase 2: 深度 LLM 提取 ============

def extract_cases_from_talk(pdf_path: Path, pool: ModelPool) -> List[dict]:
    """
    从单场演讲深度提取投资案例。
    返回该场演讲中提取到的案例列表。
    """
    logger.info(f"\n  📖 深度提取: {pdf_path.name}")

    # Step 1: PDF → 章节
    try:
        extractor = BookExtractor()
        result = extractor.extract(str(pdf_path))
        chapters = result.chapters
        logger.info(f"    提取 {len(chapters)} 个章节")
    except Exception as e:
        logger.error(f"    ❌ PDF 提取失败: {e}")
        return []

    if not chapters:
        return []

    # Step 2: 逐章 LLM 提取
    executor = pool.get("executor")
    all_cases: List[dict] = []

    for ch in chapters:
        content = ch.content
        max_chars = 8000
        if len(content) > max_chars:
            content = content[:max_chars]

        user_prompt = f"""演讲章节标题：{ch.title}

章节内容：
{content}

请提取帕伯莱本人投资过的案例。只输出 JSON，不要其他文字。如果没有本人案例，返回 {{"cases": []}}。"""

        try:
            response = executor.complete(
                prompt=user_prompt,
                system=INVESTMENT_CASES_PROMPT,
                max_tokens=CASE_EXTRACTION_TOKENS,
                temperature=0.0,
            )

            data = _parse_json_response(response)
            if data:
                cases = data.get("cases", [])
                # 只保留 is_pabrai_own_case == true 的案例
                own_cases = [c for c in cases if c.get("is_pabrai_own_case", False)]
                if own_cases:
                    for c in own_cases:
                        c["source_talk"] = pdf_path.name
                        c["source_chapter_title"] = ch.title
                    all_cases.extend(own_cases)
                    logger.info(f"      第{ch.index}章 → {len(own_cases)} 个本人案例")
                else:
                    logger.debug(f"      第{ch.index}章 → 无本人案例")

        except Exception as e:
            logger.error(f"      第{ch.index}章 LLM 调用失败: {e}")

    logger.info(f"    本场共提取 {len(all_cases)} 个案例")
    return all_cases


def run_phase2(hit_pdfs: List[Path], progress: dict) -> List[dict]:
    """Phase 2 主流程：对各命中演讲深度提取"""
    logger.info("\n" + "=" * 60)
    logger.info("🔍 Phase 2: 深度 LLM 提取")
    logger.info("=" * 60)

    pool = ModelPool()
    all_cases = list(progress.get("cases_raw", []))
    processed = set(progress.get("phase2_processed", []))

    for i, pdf_path in enumerate(hit_pdfs):
        if pdf_path.name in processed:
            logger.info(f"  [{i+1}/{len(hit_pdfs)}] ⏭️  {pdf_path.name} (已处理，跳过)")
            continue

        logger.info(f"\n  [{i+1}/{len(hit_pdfs)}] 🏃 {pdf_path.name}")
        cases = extract_cases_from_talk(pdf_path, pool)
        if cases:
            all_cases.extend(cases)
            logger.info(f"    累计: {len(all_cases)} 个案例")

        processed.add(pdf_path.name)
        # 每个 PDF 处理后保存进度
        progress["phase2_processed"] = list(processed)
        progress["cases_raw"] = all_cases
        save_progress(progress)

    logger.info(f"\n📊 Phase 2 完成：共 {len(all_cases)} 个案例（去重前）")
    logger.info(pool.report())

    return all_cases


# ============ 去重合并 ============

def _normalize_company_name(name: str) -> str:
    """标准化公司名用于比较"""
    if not name:
        return ""
    n = name.strip().lower()
    n = re.sub(r'\s+', ' ', n)
    n = re.sub(r'[^a-z0-9\s]', '', n)
    # 去除常见后缀
    for suffix in ['inc', 'ltd', 'corp', 'corporation', 'group', 'plc', 'sa', 'spa',
                   'limited', 'holdings', 'holding', 'llc', 'nv', 'ag', 'se']:
        n = re.sub(r'\b' + suffix + r'\b', '', n)
    return n.strip()


def deduplicate_cases(raw_cases: List[dict]) -> List[dict]:
    """
    跨演讲去重合并：
    1. 精确匹配公司名
    2. 模糊匹配（标准化后）
    3. 同一公司的案例合并
    """
    logger.info("\n" + "=" * 60)
    logger.info("🔄 去重合并")
    logger.info("=" * 60)
    logger.info(f"输入: {len(raw_cases)} 个原始案例条目")

    # 按标准化名称分组
    groups: Dict[str, List[dict]] = defaultdict(list)
    for case in raw_cases:
        name = case.get("company_name") or "Unknown"
        norm = _normalize_company_name(name) or "unknown"
        groups[norm].append(case)

    # 合并每组
    merged: List[dict] = []
    for norm_name, entries in groups.items():
        # 取第一个非空 company_name
        company_name = entries[0].get("company_name", "Unknown")
        ticker = next((e.get("ticker") for e in entries if e.get("ticker")), None)
        industry = next((e.get("industry") for e in entries if e.get("industry")), "Unknown")

        # 合并 buy_info（取最早提到的）
        buy_info = entries[0].get("buy_info", {})
        # 合并 sell_info
        sell_info = entries[0].get("sell_info", {})
        # 合并 outcome
        outcome = entries[0].get("outcome", {})

        # 收集所有教训和证据
        lessons = []
        evidence_quotes = []
        source_talks = []
        for e in entries:
            lessons.extend(e.get("lessons", []))
            evidence_quotes.extend(e.get("evidence_quotes", []))
            source_talks.append(e.get("source_talk", ""))

        # 去重
        lessons = list(dict.fromkeys(lessons))
        source_talks = list(dict.fromkeys(source_talks))

        merged.append({
            "company_name": company_name,
            "ticker": ticker,
            "industry": industry,
            "buy_info": buy_info,
            "sell_info": sell_info,
            "outcome": outcome,
            "lessons": lessons,
            "evidence_quotes": evidence_quotes,
            "source_talks": source_talks,
            "mention_count": len(entries),
        })

    logger.info(f"输出: {len(merged)} 个去重案例")

    # 按提及次数排序（高频案例更值得关注）
    merged.sort(key=lambda x: x["mention_count"], reverse=True)

    return merged


# ============ 输出生成 ============

def generate_output(merged_cases: List[dict]):
    """生成所有输出文件"""
    logger.info("\n" + "=" * 60)
    logger.info("📄 生成输出文件")
    logger.info("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- cases.json ----
    cases_json_path = OUTPUT_DIR / "cases.json"
    cases_json_path.write_text(
        json.dumps(merged_cases, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"  保存 cases.json ({len(merged_cases)} 个案例)")

    # ---- 每个公司独立 JSON ----
    for case in merged_cases:
        slug = _safe_slug(case["company_name"])
        case_file = OUTPUT_DIR / f"{slug}.json"
        case_file.write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"  保存 {len(merged_cases)} 个独立案例 JSON")

    # ---- index.md ----
    index_md = _render_index_md(merged_cases)
    (OUTPUT_DIR / "index.md").write_text(index_md, encoding="utf-8")
    logger.info(f"  保存 index.md ({len(index_md)} 字)")


def _render_index_md(cases: List[dict]) -> str:
    """生成案例速查 Markdown"""
    lines = [
        "# Mohnish Pabrai 投资案例集",
        "",
        f"> 从 {len(set(t for c in cases for t in c['source_talks']))} 场演讲中提取",
        f"> 共 {len(cases)} 个标的",
        "> 提取方式：LLM 结构化提取 (INVESTMENT_CASES_PROMPT)",
        "",
        "---",
        "",
    ]

    # 按行业分组
    by_industry = defaultdict(list)
    for case in cases:
        by_industry[case.get("industry", "Unknown")].append(case)

    for industry, ind_cases in sorted(by_industry.items(), key=lambda x: -len(x[1])):
        lines.append(f"## {industry} ({len(ind_cases)} 个标的)")
        lines.append("")
        lines.append("| 公司 | 代码 | 结果 | 提及次数 | 关键教训 |")
        lines.append("|------|------|------|----------|----------|")
        for c in ind_cases:
            ticker = c.get("ticker") or "-"
            result_emoji = {"win": "🟢", "loss": "🔴", "ongoing": "🟡"}.get(
                c.get("outcome", {}).get("result", ""), "⚪"
            )
            top_lesson = (c.get("lessons") or ["-"])[0][:60]
            lines.append(
                f"| {c['company_name']} | {ticker} | {result_emoji} {c.get('outcome', {}).get('result', '-')} "
                f"| {c['mention_count']} | {top_lesson} |"
            )
        lines.append("")

    # 按时间线列出详情
    lines.append("---")
    lines.append("")
    lines.append("## 案例详情")
    lines.append("")

    for i, case in enumerate(cases):
        lines.append(f"### {i + 1}. {case['company_name']}")
        lines.append("")
        lines.append(f"- **行业**: {case.get('industry', '-')}")
        lines.append(f"- **代码**: {case.get('ticker') or '-'}")

        buy = case.get("buy_info", {})
        if buy:
            lines.append(f"- **买入时间**: {buy.get('date', '-')}")
            lines.append(f"- **买入逻辑**: {buy.get('thesis', '-')}")

        sell = case.get("sell_info", {})
        if sell:
            lines.append(f"- **卖出时间**: {sell.get('date', '-')}")
            lines.append(f"- **卖出原因**: {sell.get('reason', '-')}")

        outcome = case.get("outcome", {})
        if outcome:
            lines.append(f"- **IRR**: {outcome.get('irr', '-')}")
            lines.append(f"- **回报倍数**: {outcome.get('multiple', '-')}")
            lines.append(f"- **结果**: {outcome.get('result', '-')}")

        lessons = case.get("lessons", [])
        if lessons:
            lines.append("")
            lines.append("**教训**:")
            for l in lessons:
                lines.append(f"  - {l}")

        quotes = case.get("evidence_quotes", [])
        if quotes:
            lines.append("")
            lines.append("**原文证据**:")
            for q in quotes[:3]:
                lines.append(f"  > {q}")

        talks = case.get("source_talks", [])
        if talks:
            lines.append("")
            lines.append(f"**来源演讲** ({len(talks)} 场): {', '.join(talks[:5])}")

        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def _safe_slug(name: str) -> str:
    """生成安全的文件名"""
    slug = name.strip().lower()
    slug = re.sub(r'[^a-z0-9]+', '_', slug)
    slug = slug.strip('_')
    return slug[:60] or "unknown"


def _parse_json_response(response: str) -> dict:
    """从 LLM 响应中提取 JSON（容错处理，支持截断修复）"""
    # 尝试直接解析
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # 尝试提取 ```json ... ``` 块
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试找第一个 { 和最后一个 }
    start = response.find("{")
    end = response.rfind("}")
    if start >= 0 and end > start:
        candidate = response[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            # 尝试修复截断的 JSON：补齐缺失的 } 和 ]
            try:
                repaired = _repair_truncated_json(candidate)
                if repaired:
                    return json.loads(repaired)
            except json.JSONDecodeError:
                pass
            logger.warning(f"无法解析 JSON (pos={e.pos}): {candidate[max(0,e.pos-50):e.pos+30]}...")

    logger.warning(f"无法解析 JSON: {response[:200]}...")
    return {}


def _repair_truncated_json(s: str) -> str:
    """修复 LLM 输出被截断的 JSON（补齐缺失的括号）"""
    # 统计未闭合的括号
    open_braces = 0
    open_brackets = 0
    in_string = False
    escape_next = False

    for ch in s:
        if escape_next:
            escape_next = False
            continue
        if ch == '\\':
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            open_braces += 1
        elif ch == '}':
            open_braces -= 1
        elif ch == '[':
            open_brackets += 1
        elif ch == ']':
            open_brackets -= 1

    # 补齐
    repaired = s.rstrip()
    # 如果以逗号结尾，去掉
    if repaired.endswith(','):
        repaired = repaired[:-1]
    # 补齐 ]
    repaired += ']' * max(0, open_brackets)
    # 补齐 }
    repaired += '}' * max(0, open_braces)

    return repaired if (open_braces > 0 or open_brackets > 0) else None


# ============ 主入口 ============

def main():
    """主流程"""
    import argparse

    parser = argparse.ArgumentParser(description="帕伯莱投资案例提取")
    parser.add_argument("--phase1", action="store_true", help="仅运行 Phase 1 扫描")
    parser.add_argument("--phase2", action="store_true", help="仅运行 Phase 2 深度提取")
    parser.add_argument("--dedup", action="store_true", help="仅运行去重合并")
    parser.add_argument("--output", action="store_true", help="仅生成输出文件")
    args = parser.parse_args()

    # 如果没有任何参数，运行全流程
    run_all = not (args.phase1 or args.phase2 or args.dedup or args.output)

    t_start = time.time()
    progress = load_progress()

    # Phase 1
    if run_all or args.phase1:
        hit_pdfs = scan_pdfs_for_cases()
        progress["phase1_done"] = True
        progress["hit_pdfs"] = [p.name for p in hit_pdfs]
        save_progress(progress)
    else:
        scan_report = OUTPUT_DIR / "scan_report.json"
        if scan_report.exists():
            report = json.loads(scan_report.read_text(encoding="utf-8"))
            hit_pdfs = [TRANSCRIPTS_DIR / h["file"] for h in report.get("hits", [])]
            logger.info(f"从 {scan_report} 加载 Phase 1 结果: {len(hit_pdfs)} 命中")
        else:
            logger.error("Phase 1 未完成，请先运行 --phase1")
            return

    # Phase 2
    if run_all or args.phase2:
        raw_cases = run_phase2(hit_pdfs, progress)
        progress["cases_raw"] = raw_cases
        progress["phase2_done"] = True
        save_progress(progress)
    else:
        raw_cases = progress.get("cases_raw", [])

    if not raw_cases:
        logger.warning("没有提取到任何案例，跳过后续步骤")
        return

    # 去重合并
    if run_all or args.dedup:
        merged_cases = deduplicate_cases(raw_cases)
        progress["cases_merged"] = merged_cases
        save_progress(progress)
    else:
        merged_cases = progress.get("cases_merged", raw_cases)

    # 生成输出
    if run_all or args.output:
        generate_output(merged_cases)

    # 成本报告
    elapsed = time.time() - t_start
    logger.info(f"\n⏱️ 总耗时: {elapsed/60:.1f} 分钟")
    logger.info(f"📦 输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
