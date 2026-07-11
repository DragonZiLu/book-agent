"""
定向提取帕伯莱"抄作业（Cloning）"方法论

输入：LSE / Harvard / Nebraska 三场克隆内容最密集的演讲 PDF
输出：output/pabrai/cloning/skill/ 目录下的方法论文件

用法：
    python extract_cloning.py
    # 或通过 SKIP_EVAL=1 跳过评估（默认跳过）
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List

# 切到脚本所在目录，确保模块导入正确
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import CHAPTER_SUMMARY_TOKENS, PABRAI_TRANSCRIPTS_DIR
from llm_client import ModelPool
from extractor import BookExtractor
from summarizer import CLONING_PROMPT, BookSummarizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("extract_cloning")

# 强制跳过评估
os.environ["SKIP_EVAL"] = "1"

# ============ 三场目标演讲 ============
TRANSCRIPTS_DIR = Path(PABRAI_TRANSCRIPTS_DIR)

TARGET_TALKS = {
    "lse": "20240219_mohnish_pabrais_q_a_with_london_school_of_economics_-_value_investing_society_on_january_30_2024.pdf",
    "harvard": "20231018_mohnish_pabrais_q_a_with_students_at_the_harvard_business_school_on_sep_15_2023.pdf",
    "nebraska": "20240620_mohnish_pabrais_session_at_the_university_of_nebraska_omaha_on_may_3_2024_v2.pdf",
}

OUTPUT_DIR = Path("output/pabrai/cloning")


def merge_cloning_results(all_results: List[dict]) -> dict:
    """合并多章节提取结果，去重并按维度归类"""
    merged = {
        "cloning_philosophy": [],
        "operational_steps": [],
        "filter_criteria": [],
        "red_flags": [],
        "checklist_items": [],
        "case_examples": [],
        "evidence": [],
        "source_talks": list(TARGET_TALKS.keys()),
    }

    for result in all_results:
        for key in merged:
            if key == "source_talks":
                continue
            items = result.get(key, [])
            if isinstance(items, list):
                for item in items:
                    if item and item not in merged[key]:
                        merged[key].append(item)

    return merged


def extract_cloning_from_talk(
    talk_id: str,
    pdf_path: Path,
    pool: ModelPool,
) -> dict:
    """
    从单场演讲中提取克隆方法论。
    使用 BookExtractor 分割章节，逐章调用 LLM（CLONING_PROMPT），合并结果。
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"📖 处理: {talk_id} → {pdf_path.name}")
    logger.info(f"{'='*60}")

    if not pdf_path.exists():
        logger.error(f"  文件不存在: {pdf_path}")
        return {}

    # Step 1: 文本提取 + 章节分割
    extractor = BookExtractor()
    result = extractor.extract(str(pdf_path))
    chapters = result.chapters
    logger.info(f"  提取 {len(chapters)} 个章节")

    for w in result.warnings:
        logger.warning(f"  ⚠️ {w}")

    if not chapters:
        logger.error("  无有效章节")
        return {}

    # Step 2: 逐章 LLM 提取（使用 CLONING_PROMPT）
    executor = pool.get("executor")
    chapter_results = []

    for ch in chapters:
        content = ch.content
        # 限制每章输入长度，避免超出 token 限制
        max_chars = 8000
        if len(content) > max_chars:
            content = content[:max_chars]

        user_prompt = f"""章节标题：{ch.title}

章节内容：
{content}

请按上述 JSON 格式提取结构化信息。只输出 JSON，不要其他文字。"""

        logger.info(f"  📝 提取第{ch.index}章: {ch.title[:40]}... ({len(content)} 字)")

        try:
            response = executor.complete(
                prompt=user_prompt,
                system=CLONING_PROMPT,
                max_tokens=CHAPTER_SUMMARY_TOKENS,
                temperature=0.0,
            )

            # 解析 JSON 响应
            data = _parse_json_response(response)
            if data:
                chapter_results.append(data)
                logger.info(f"     ✅ 提取到 {sum(len(v) if isinstance(v,list) else 0 for v in data.values())} 条信息")
            else:
                logger.warning(f"     ⚠️ 无法解析 JSON，跳过")

        except Exception as e:
            logger.error(f"     ❌ 提取失败: {e}")

    # Step 3: 合并本场演讲的所有章节结果
    talk_merged = merge_cloning_results(chapter_results)
    logger.info(f"  合并结果: {sum(len(v) for v in talk_merged.values() if isinstance(v,list))} 条记录")

    return talk_merged


def generate_output_files(all_talks: Dict[str, dict]):
    """生成最终输出文件到 output/pabrai/cloning/skill/"""
    skill_dir = OUTPUT_DIR / "skill"
    skill_dir.mkdir(parents=True, exist_ok=True)

    # 合并所有演讲
    combined = merge_cloning_results(list(all_talks.values()))

    # ---- SKILL.md ----
    skill_md = _render_skill_md(combined)
    (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")
    logger.info(f"  保存 SKILL.md ({len(skill_md)} 字)")

    # ---- cloning_philosophy.md ----
    philo_content = "\n\n".join(f"### {i+1}\n{item}" for i, item in enumerate(combined["cloning_philosophy"]))
    (skill_dir / "cloning_philosophy.md").write_text(
        f"# 克隆哲学与操作原则\n\n{philo_content or '无'}", encoding="utf-8"
    )

    # ---- operational_steps.md ----
    steps_content = "\n\n".join(f"### 步骤 {i+1}\n{item}" for i, item in enumerate(combined["operational_steps"]))
    (skill_dir / "operational_steps.md").write_text(
        f"# 可执行操作步骤\n\n{steps_content or '无'}", encoding="utf-8"
    )

    # ---- checklists.md ----
    checklist_content = "\n".join(f"- [ ] {item}" for item in combined["checklist_items"])
    (skill_dir / "checklists.md").write_text(
        f"# 人工核查清单\n\n{checklist_content or '无'}", encoding="utf-8"
    )

    # ---- 保存原始 JSON（调试用）----
    (skill_dir / "raw_combined.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- 每场演讲的独立结果 ----
    for talk_id, talk_data in all_talks.items():
        talk_file = skill_dir / f"{talk_id}_extracted.json"
        talk_file.write_text(
            json.dumps(talk_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"  保存 {talk_id}_extracted.json")

    logger.info(f"\n✅ 全部输出已保存到: {skill_dir}")


def _render_skill_md(combined: dict) -> str:
    """将合并结果渲染为 Markdown"""
    lines = [
        "# 帕伯莱「抄作业」方法论（Cloning Framework）",
        "",
        "> 来源：帕伯莱 LSE / Harvard / Nebraska 三场演讲",
        "> 提取方式：LLM 结构化提取 (CLONING_PROMPT)",
        "",
        "---",
        "",
        "## 一、克隆哲学",
        "",
    ]
    for item in combined.get("cloning_philosophy", []):
        lines.append(f"- {item}")

    lines.extend([
        "",
        "## 二、操作步骤",
        "",
    ])
    for i, item in enumerate(combined.get("operational_steps", [])):
        lines.append(f"{i+1}. {item}")

    lines.extend([
        "",
        "## 三、筛选标准",
        "",
    ])
    for item in combined.get("filter_criteria", []):
        lines.append(f"- {item}")

    lines.extend([
        "",
        "## 四、风险与盲区",
        "",
    ])
    for item in combined.get("red_flags", []):
        lines.append(f"- ⚠️ {item}")

    lines.extend([
        "",
        "## 五、检查清单",
        "",
    ])
    for item in combined.get("checklist_items", []):
        lines.append(f"- [ ] {item}")

    lines.extend([
        "",
        "## 六、典型案例",
        "",
    ])
    for item in combined.get("case_examples", []):
        lines.append(f"- {item}")

    lines.extend([
        "",
        "## 七、原文证据",
        "",
    ])
    for item in combined.get("evidence", []):
        lines.append(f"> {item}")

    lines.extend([
        "",
        "---",
        "",
        f"*数据源: {', '.join(combined.get('source_talks', []))}*",
    ])

    return "\n".join(lines)


def _parse_json_response(response: str) -> dict:
    """从 LLM 响应中提取 JSON（容错处理）"""
    import re

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
        try:
            return json.loads(response[start:end + 1])
        except json.JSONDecodeError:
            pass

    logger.warning(f"无法解析 JSON: {response[:200]}...")
    return {}


def main():
    """主流程：依次处理三场演讲，合并输出"""
    t_start = time.time()

    logger.info("=" * 60)
    logger.info("🧬 帕伯莱「抄作业」方法论 — 定向提取")
    logger.info("=" * 60)

    pool = ModelPool()
    all_talks = {}

    for talk_id, filename in TARGET_TALKS.items():
        pdf_path = TRANSCRIPTS_DIR / filename
        talk_data = extract_cloning_from_talk(talk_id, pdf_path, pool)
        if talk_data:
            all_talks[talk_id] = talk_data

    if not all_talks:
        logger.error("❌ 所有演讲提取均失败，退出")
        return

    # 生成输出文件
    logger.info(f"\n{'='*60}")
    logger.info("📄 生成输出文件...")
    logger.info(f"{'='*60}")
    generate_output_files(all_talks)

    # 成本报告
    elapsed = time.time() - t_start
    logger.info(f"\n⏱️ 总耗时: {elapsed:.1f}s")
    logger.info(pool.report())


if __name__ == "__main__":
    main()
