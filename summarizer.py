"""
Executor 提取 Agent：流派识别 + 结构化提取 + rules.json 生成
按流派选择 Prompt 模板，支持携带 WeakChapterFeedback 重提取
"""

import json
import logging
import time
from typing import List, Optional, Dict
from dataclasses import dataclass, field

from tqdm import tqdm

from config import (
    REQUIRED_DIMENSIONS,
    CHAPTER_SUMMARY_TOKENS,
    SKILL_MD_TOKENS,
)
from llm_client import ModelPool, LLMClient

logger = logging.getLogger(__name__)


# ============ 流派 Prompt 模板 ============
VALUE_INVESTING_PROMPT = """你是一位精通价值投资的分析师。请从以下投资书籍章节中提取结构化信息。

重点关注：
- 选股正向标准（PE/PB/安全边际/护城河/ROE等量化阈值）
- 风险信号（财务造假/高负债/管理层问题等）
- 估值方法体系（DCF/格雷厄姆公式/资产重估等）
- 投资前检查清单
- 典型案例（书中明确提到的公司/行业案例）

对于每条规则，尽可能提取：
1. 明确的数值阈值（如 PE≤15）
2. 规则的适用条件
3. 规则的出处（书中原文片段作为 evidence）

输出格式（JSON）：
{
  "core_arguments": ["核心论点1", ...],
  "evidence": ["原文证据1", ...],
  "selection_criteria": ["选股标准1", ...],
  "red_flags": ["风险信号1", ...],
  "valuation_methods": ["估值方法1", ...],
  "rules": ["如果...则...的决策规则", ...],
  "checklist_items": ["检查项1", ...],
  "cases": ["案例1: ...", ...]
}
"""

GROWTH_INVESTING_PROMPT = """你是一位精通成长投资的分析师。请从以下投资书籍章节中提取结构化信息。

重点关注：
- 成长性指标（PEG/营收增速/利润增速/行业空间等）
- 选股正向标准（市场规模/竞争格局/技术壁垒等）
- 风险信号（增速放缓/竞争加剧/技术替代等）
- 估值方法（PEG/PS/未来现金流折现等）
- 投资前检查清单
- 典型案例（书中明确提到的公司/行业案例）

对于每条规则，尽可能提取明确的数值阈值和原文证据。

输出格式（JSON）：{...同上...}
"""

QUANT_INVESTING_PROMPT = """你是一位精通量化投资的分析师。请从以下投资书籍章节中提取结构化信息。

重点关注：
- 因子定义与信号强度（动量/价值/质量/波动率等因子）
- 选股模型参数（因子权重/调仓频率/持仓数量等）
- 风险控制规则（最大回撤/止损/仓位管理等）
- 回测统计显著性要求
- 投资前检查清单

对于每条规则，尽可能提取明确的数值阈值和原文证据。

输出格式（JSON）：{...同上...}
"""

MACRO_INVESTING_PROMPT = """你是一位精通宏观投资的分析师。请从以下投资书籍章节中提取结构化信息。

重点关注：
- 宏观指标阈值（利率/通胀/GDP/PMI等）
- 资产配置比例规则
- 周期判断标准
- 风险信号（政策转向/地缘政治等）
- 投资前检查清单

对于每条规则，尽可能提取明确的数值阈值和原文证据。

输出格式（JSON）：{...同上...}
"""

GENERAL_PROMPT = """你是一位资深投资分析师。请从以下投资书籍章节中提取结构化信息。

重点关注：
- 选股正向标准
- 风险信号
- 估值方法体系
- 投资前检查清单
- 典型案例

对于每条规则，尽可能提取明确的数值阈值和原文证据。

输出格式（JSON）：
{
  "core_arguments": [],
  "evidence": [],
  "selection_criteria": [],
  "red_flags": [],
  "valuation_methods": [],
  "rules": [],
  "checklist_items": [],
  "cases": []
}
"""

PROMPT_TEMPLATES = {
    "value":   VALUE_INVESTING_PROMPT,
    "growth":  GROWTH_INVESTING_PROMPT,
    "quant":   QUANT_INVESTING_PROMPT,
    "macro":   MACRO_INVESTING_PROMPT,
    "general": GENERAL_PROMPT,
}


# ============ 数据结构 ============
@dataclass
class ChapterSummary:
    chapter_index: int
    title: str
    core_arguments: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    selection_criteria: List[str] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)
    valuation_methods: List[str] = field(default_factory=list)
    rules: List[str] = field(default_factory=list)
    checklist_items: List[str] = field(default_factory=list)
    cases: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "chapter_index": self.chapter_index,
            "title": self.title,
            "core_arguments": self.core_arguments,
            "evidence": self.evidence,
            "selection_criteria": self.selection_criteria,
            "red_flags": self.red_flags,
            "valuation_methods": self.valuation_methods,
            "rules": self.rules,
            "checklist_items": self.checklist_items,
            "cases": self.cases,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChapterSummary":
        return cls(
            chapter_index=d.get("chapter_index", 0),
            title=d.get("title", ""),
            core_arguments=d.get("core_arguments", []),
            evidence=d.get("evidence", []),
            selection_criteria=d.get("selection_criteria", []),
            red_flags=d.get("red_flags", []),
            valuation_methods=d.get("valuation_methods", []),
            rules=d.get("rules", []),
            checklist_items=d.get("checklist_items", []),
            cases=d.get("cases", []),
        )


@dataclass
class ExtractionOutput:
    chapter_summaries: List[ChapterSummary]
    skill_md: str
    criteria: str
    red_flags: str
    valuation: str
    checklists: str
    rules_json: dict

    def to_dict(self) -> dict:
        return {
            "chapter_summaries": [cs.to_dict() for cs in self.chapter_summaries],
            "skill_md": self.skill_md,
            "criteria": self.criteria,
            "red_flags": self.red_flags,
            "valuation": self.valuation,
            "checklists": self.checklists,
            "rules_json": self.rules_json,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExtractionOutput":
        summaries = [
            ChapterSummary.from_dict(cs)
            for cs in d.get("chapter_summaries", [])
        ]
        return cls(
            chapter_summaries=summaries,
            skill_md=d.get("skill_md", ""),
            criteria=d.get("criteria", ""),
            red_flags=d.get("red_flags", ""),
            valuation=d.get("valuation", ""),
            checklists=d.get("checklists", ""),
            rules_json=d.get("rules_json", {}),
        )


class BookSummarizer:
    """Executor 角色：投资框架结构化提取"""

    def __init__(self, pool: ModelPool):
        self.pool = pool
        self.executor: LLMClient = pool.get("executor")
        self.utility: LLMClient = pool.get("utility")
        self._strategy_type: str = "general"

    def detect_strategy_type(self, chapters: list) -> str:
        """
        流派识别：utility 模型看前 2 章 + 目录分类。
        返回 "value"|"growth"|"quant"|"macro"|"general"
        """
        sample_chapters = chapters[:2]
        sample_text = "\n\n".join(
            f"## {ch.title}\n{ch.content[:2000]}" for ch in sample_chapters
        )

        prompt = f"""请判断以下投资书籍属于哪种流派。只输出一个单词。

选项：value（价值投资）、growth（成长投资）、quant（量化投资）、macro（宏观投资）、general（通用）

书籍内容样本：
{sample_text[:4000]}

流派："""

        result = self.utility.complete(prompt, max_tokens=20).strip().lower()
        valid_types = {"value", "growth", "quant", "macro", "general"}

        for t in valid_types:
            if t in result:
                self._strategy_type = t
                logger.info(f"流派识别: {t}")
                return t

        self._strategy_type = "general"
        logger.info(f"流派识别: general (无法确定，结果: {result})")
        return "general"

    def summarize_chapter(self, chapter, feedback=None) -> ChapterSummary:
        """提取单个章节，可携带上次评估的反馈"""
        template = PROMPT_TEMPLATES.get(self._strategy_type, GENERAL_PROMPT)

        # 构建用户 prompt
        user_prompt = f"""章节标题：{chapter.title}

章节内容：
{chapter.content}

请按上述 JSON 格式提取结构化信息。只输出 JSON，不要其他文字。"""

        # 如果有反馈，补充缺失信息提示
        if feedback:
            missing = getattr(feedback, "missing_dimensions", [])
            missing_concepts = getattr(feedback, "missing_concepts", [])
            failed_qs = getattr(feedback, "failed_questions", [])
            hint = getattr(feedback, "hint", "")

            feedback_text = "\n\n⚠️ 上次提取遗漏/错误："
            if missing:
                feedback_text += f"\n- 遗漏维度: {', '.join(missing)}"
            if missing_concepts:
                feedback_text += f"\n- 遗漏概念: {', '.join(missing_concepts[:8])}"
            if failed_qs:
                feedback_text += f"\n- 答错题目: {', '.join(failed_qs)}"
            if hint:
                feedback_text += f"\n- 提示: {hint}"
            user_prompt += feedback_text

        system_prompt = template

        try:
            response = self.executor.complete(
                prompt=user_prompt,
                system=system_prompt,
                max_tokens=CHAPTER_SUMMARY_TOKENS,
                temperature=0.0,
            )

            # 尝试从响应中提取 JSON
            data = self._parse_json_response(response)
            return ChapterSummary(
                chapter_index=chapter.index,
                title=chapter.title,
                core_arguments=data.get("core_arguments", []),
                evidence=data.get("evidence", []),
                selection_criteria=data.get("selection_criteria", []),
                red_flags=data.get("red_flags", []),
                valuation_methods=data.get("valuation_methods", []),
                rules=data.get("rules", []),
                checklist_items=data.get("checklist_items", []),
                cases=data.get("cases", []),
            )
        except Exception as e:
            logger.warning(f"章节提取失败 [{chapter.title}]: {e}")
            return ChapterSummary(
                chapter_index=chapter.index,
                title=chapter.title,
            )

    def _parse_json_response(self, response: str) -> dict:
        """从 LLM 响应中提取 JSON（容错处理）"""
        # 尝试直接解析
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # 尝试提取 ```json ... ``` 块
        json_match = __import__("re").search(
            r"```(?:json)?\s*([\s\S]*?)```", response
        )
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

        logger.warning(f"无法解析 JSON 响应: {response[:200]}...")
        return {}

    def extract_all_chapters(
        self, chapters, feedbacks=None, cache=None
    ) -> tuple[List[ChapterSummary], dict]:
        """
        提取所有章节。
        feedbacks: List[WeakChapterFeedback] | None，重试时携带
        cache: ExtractionCache | None，有缓存时只重提取 weak 章节
        """
        # 构建 feedback 索引
        fb_index: dict[int, any] = {}
        if feedbacks:
            for fb in feedbacks:
                fb_index[fb.index] = fb

        summaries: List[ChapterSummary] = []
        chapter_keys: dict[int, str] = {}  # 用于缓存

        cached_count = 0
        retry_count = len(fb_index)
        label = "重提取" if feedbacks else "提取"
        pbar = tqdm(chapters, desc=f"  📝 {label}章节", unit="章", ncols=80)

        for ch in pbar:
            pbar.set_postfix_str(f"第{ch.index}章")

            # 尝试从缓存读取
            if cache:
                key = cache.get_key(ch.content)
                chapter_keys[ch.index] = key
                cached = cache.get(key)
                if cached and ch.index not in fb_index:
                    cs = ChapterSummary.from_dict(cached)
                    summaries.append(cs)
                    cached_count += 1
                    pbar.set_postfix_str(f"第{ch.index}章 (缓存)")
                    continue

            # 提取（可能带反馈）
            fb = fb_index.get(ch.index)
            is_retry = fb is not None
            t0 = time.time()
            cs = self.summarize_chapter(ch, feedback=fb)
            elapsed = time.time() - t0
            summaries.append(cs)

            status = "重试" if is_retry else "完成"
            pbar.set_postfix_str(f"第{ch.index}章 {status} ({elapsed:.1f}s)")

            # 写入缓存
            if cache and ch.index in chapter_keys:
                cache.set(chapter_keys[ch.index], cs.to_dict())

        pbar.close()

        total = len(chapters)
        logger.info(
            f"  章节提取: {cached_count} 缓存命中, "
            f"{total - cached_count - retry_count} 新提取, "
            f"{retry_count} 重提取"
        )

        # 按章节序号排序
        summaries.sort(key=lambda s: s.chapter_index)

        # 生成全局 rules_json
        logger.info("  🔧 生成 rules.json...")
        rules_json = self._generate_rules_json(summaries)

        return summaries, rules_json

    def _generate_rules_json(self, summaries: List[ChapterSummary]) -> dict:
        """只提取有明确数值阈值的规则，模糊描述不输出"""

        def extract_numeric_rules(summary: ChapterSummary) -> List[dict]:
            """从单个章节摘要中提取数值规则"""
            results = []
            all_rules = summary.rules + summary.selection_criteria + summary.red_flags

            # 匹配数值模式："PE ≤ 15" / "PE不超过15" / "PE<15" 等
            num_pattern = __import__("re").compile(
                r"([A-Za-z_/]+)\s*(?:[≤>=<]|不超过|低于|高于|大于|小于|至少|不少于|不高于|不大于)\s*([\d.]+)"
            )
            for rule_text in all_rules:
                for m in num_pattern.finditer(rule_text):
                    metric = m.group(1).strip()
                    try:
                        value = float(m.group(2))
                    except ValueError:
                        continue
                    if (
                        metric.lower()
                        not in ("至少", "不超过", "低于", "高于", "大于", "小于")
                    ):
                        op = self._infer_operator(rule_text, metric)
                        results.append({
                            "id": f"{metric.lower()}_{len(results)}",
                            "metric": metric,
                            "op": op,
                            "value": value,
                            "source": f"ch{summary.chapter_index}",
                        })
            return results

        all_rules = []
        for s in summaries:
            all_rules.extend(extract_numeric_rules(s))

        return {
            "strategy_type": self._strategy_type,
            "valuation_rules": [
                r for r in all_rules
                if r["metric"].upper() in ("PE", "PB", "PS", "PEG", "EV/EBITDA", "ROE", "ROA")
            ],
            "selection_criteria": all_rules,
            "red_flag_rules": [
                r for r in all_rules
                if "flag" in r["metric"].lower() or "risk" in r["metric"].lower()
            ],
        }

    @staticmethod
    def _infer_operator(text: str, metric: str) -> str:
        """从规则文本推断比较运算符"""
        text_lower = text.lower()
        if any(w in text_lower for w in ["不超过", "低于", "小于", "不高于", "不大于"]):
            return "<="
        elif any(w in text_lower for w in ["至少", "不少于", "不低于", "高于", "大于"]):
            return ">="
        elif "≥" in text or ">=" in text:
            return ">="
        elif "≤" in text or "<=" in text:
            return "<="
        elif ">" in text:
            return ">="
        elif "<" in text:
            return "<="
        elif "=" in text:
            return "=="
        return "<="

    def generate_global_files(self, summaries: List[ChapterSummary], rules_json: dict) -> ExtractionOutput:
        """生成全局 Skill 文件"""
        executor = self.executor

        # 聚合所有章节数据
        all_criteria = []
        all_red_flags = []
        all_valuation = []
        all_checklists = []
        all_cases = []
        all_args = []

        for s in summaries:
            all_criteria.extend(s.selection_criteria)
            all_red_flags.extend(s.red_flags)
            all_valuation.extend(s.valuation_methods)
            all_checklists.extend(s.checklist_items)
            all_cases.extend(s.cases)
            all_args.extend(s.core_arguments)

        # 生成 SKILL.md（核心投资哲学 + 框架索引）
        skill_prompt = f"""请根据以下从投资书籍提取的信息，生成一份 SKILL.md 文档。

文档结构：
1. 流派：{self._strategy_type}
2. 核心投资哲学（基于 core_arguments）
3. 选股框架概览
4. 风险识别体系
5. 估值方法论
6. 决策检查清单

核心论点：
{chr(10).join(f'- {a}' for a in all_args[:20])}

选股标准：
{chr(10).join(f'- {c}' for c in all_criteria[:20])}

风险信号：
{chr(10).join(f'- {r}' for r in all_red_flags[:20])}

估值方法：
{chr(10).join(f'- {v}' for v in all_valuation[:20])}

请输出 Markdown 格式的 SKILL.md，字数控制在 {SKILL_MD_TOKENS} tokens 以内。"""

        skill_md = executor.complete(skill_prompt, max_tokens=SKILL_MD_TOKENS, temperature=0.0)

        # 生成 criteria.md
        criteria_content = "\n".join(f"- {c}" for c in all_criteria) if all_criteria else "无"
        criteria = f"# 选股正向标准\n\n{criteria_content}"

        # 生成 red_flags.md
        red_flags_content = "\n".join(f"- {r}" for r in all_red_flags) if all_red_flags else "无"
        red_flags = f"# 风险信号\n\n{red_flags_content}"

        # 生成 valuation.md
        valuation_content = "\n".join(f"- {v}" for v in all_valuation) if all_valuation else "无"
        valuation = f"# 估值方法体系\n\n{valuation_content}"

        # 生成 checklists.md
        checklist_content = "\n".join(f"- {c}" for c in all_checklists) if all_checklists else "无"
        checklists = f"# 投资前检查表\n\n{checklist_content}"

        return ExtractionOutput(
            chapter_summaries=summaries,
            skill_md=skill_md,
            criteria=criteria,
            red_flags=red_flags,
            valuation=valuation,
            checklists=checklists,
            rules_json=rules_json,
        )

    def summarize(
        self, chapters, feedbacks=None, cache=None
    ) -> ExtractionOutput:
        """
        完整提取流程：
        1. 流派识别（首次）
        2. 逐章结构化提取（带缓存 + 进度条）
        3. 生成全局文件
        """
        t_start = time.time()

        if not self._strategy_type or self._strategy_type == "general":
            self.detect_strategy_type(chapters)

        logger.info(f"  📖 流派: {self._strategy_type} | 章节数: {len(chapters)}")

        summaries, rules_json = self.extract_all_chapters(chapters, feedbacks, cache)

        logger.info("  📄 生成 Skill 文件...")
        output = self.generate_global_files(summaries, rules_json)

        elapsed = time.time() - t_start
        logger.info(f"  ✅ 提取完成, 耗时 {elapsed:.1f}s")

        return output
