# 投资书籍核心提取 Agent（双模型 Executor + Judge 架构 v3）

## 目标

构建一个纯 Python CLI 工具，专用于投资类书籍，输入 PDF/EPUB/URL，通过 **Executor（执行）+ Judge（评判）双模型交叉验证**架构，提取结构化投资决策框架并验证提取充分性。

最终产出物可直接用于：对个股/资产进行投资建议生成、风险评估、策略检验。

**成功标准**：
- 输入任意投资书籍 → 自动输出投资框架 Skill 文件 + 机器可读 `rules.json` + 质量报告
- L1 token 覆盖率 ≥ 95%，L2 投资维度召回 ≥ 80%，L3 QA 准确率 ≥ 75%
- numeric 类题用代码规则比对（零 LLM 幻觉），direction/factual 类由 Judge 语义判分
- 重试时携带 Judge 发现的具体缺失反馈（不盲目重跑），成本设硬熔断 $5

---

<!-- anchor:dual-model-design -->
## 1. 双模型架构设计原则

```
核心隔离边界：
  [Executor: GPT-4o]   ←── 这条线必须异源 ──→   [Examiner + Judge: Claude]

  Executor 负责：提取 Skill 文件（批量、量大，需要质量）
  Examiner 负责：从原文出题 + 生成标准答案（不看 Skill 文件）
  Judge    负责：评判 Executor 的回答（以原文为唯一真相源）
  Utility  负责：廉价杂活（流派识别、L2 概念抽取）
```

**为什么能解决循环依赖**：
- 单模型：Claude 提取 → Claude 出题 → Claude 判分，同源盲区贯穿始终
- 双模型：GPT 提取 → Claude 独立出题（看原文）→ Claude 判分（看原文），漏的概念会被 Judge 发现

**三条核心原则**：
1. Judge 出题和判分都**只看原文**，不接触 Skill 文件
2. Executor 与 Judge/Examiner 必须是**不同厂商模型**
3. Examiner 与 Judge 可以**同源**（保持判分标准自洽）

---

<!-- anchor:project-structure -->
## 2. 项目结构

```
book-agent/
├── book_agent.py          # 主入口 CLI（extract / advise 两个子命令）
├── extractor.py           # 文本提取（PDF/EPUB/URL），含章节容错降级
├── summarizer.py          # Executor：流派识别 + 投资框架结构化提取
├── evaluator.py           # Judge + 三层评估器（含 ScenarioQA 分层判分）
├── rule_engine.py         # 规则引擎：执行 rules.json 量化比对
├── investment_advisor.py  # 投资顾问：规则引擎 + Judge 定性分析
├── llm_client.py          # LLMClient + ModelPool（角色化多模型管理）
├── cache.py               # 磁盘缓存（hash(chapter) 为 key）
├── config.py              # 角色模型配置 + 阈值 + 成本熔断
├── requirements.txt
├── tests/
│   └── golden/
│       └── intelligent_investor_qa.json   # 黄金测试集（人工标注）
└── output/
    ├── skill/
    │   ├── SKILL.md           # 投资哲学 + 框架索引
    │   ├── chapters/          # 分章节摘要
    │   ├── criteria.md        # 选股正向标准（人类可读）
    │   ├── red_flags.md       # 风险信号（人类可读）
    │   ├── valuation.md       # 估值方法体系（人类可读）
    │   ├── checklists.md      # 投资前检查表
    │   └── rules.json         # 机器可读量化规则（供规则引擎执行）
    └── reports/
        └── quality_report.json
```

---

<!-- anchor:pipeline -->
## 3. 完整 Pipeline

```
输入(PDF/EPUB/URL)
    │
    ▼
[extractor.py] 文本提取 + 章节分割
    ├── 优先按标题匹配（Chapter N / 第N章 / CHAPTER FOURTEEN 等）
    ├── 失败时降级为 token 滑动分块（3000 tokens/块，overlap 200）
    └── 章节质量告警（过短 <200 / 过大 >均值×5 / 只有1块）
    │ chapters: List[Chapter]
    ▼
[summarizer.py] Executor（GPT-4o）
    Step 1: 流派识别（utility 模型）→ "value"|"growth"|"quant"|"macro"|"general"
    Step 2: 按流派选 Prompt 模板 → 章节结构化提取（带反馈时携带缺失信息）
    Step 3: 生成全局文件（SKILL.md / criteria / red_flags / valuation / checklists）
    Step 4: 生成 rules.json（量化规则，只提取有明确数值阈值的规则）
    │ extraction: ExtractionResult（含 cache，重试时未变章节直接读缓存）
    ▼
[evaluator.py] Judge（Claude Sonnet）三层评估
    ├── L1（纯规则，0 LLM 成本）：原文 token 覆盖率 ≥ 95%
    ├── L2（utility 模型）：四维度召回 ≥ 80%
    │   └── 选股标准 / 风险信号 / 估值方法 / 典型案例
    └── L3 ScenarioQA（Examiner 出题，Executor 答，Judge 判）：≥ 75%
        ├── Examiner 看原文出题，同时生成 gold_answer + evidence_span
        ├── Executor 看 Skill 文件答题
        ├── numeric 类：正则提取数字 → 代码比对（不走 LLM）
        ├── direction 类：Judge 判分，方向错=直接错，需引用 evidence_span
        └── factual 类：Judge 语义判分，核心事实正确即算对
    │ QualityReport（含 WeakChapterFeedback，具体缺失维度 + 答错题目）
    ▼
[book_agent.py] 决策 + 反馈闭环
    ├── 全部达标 → save_output + 打印成本报告
    ├── 未达标 → 把 feedbacks 传给 Executor 重提取（最多 2 次）
    └── 成本超 $5 → 强制熔断，保存当前结果
    │
    ▼（可选）
[investment_advisor.py] 投资顾问
    ├── rule_engine 加载 rules.json → 硬性量化筛查（代码执行）
    ├── Judge 处理无法量化的定性部分（护城河、管理层等）
    ├── 关键指标缺失 >50% → verdict = "INSUFFICIENT_DATA"
    └── 输出：verdict + score + 规则命中详情 + 数据时点 + 免责声明
```

---

<!-- anchor:modules -->
## 4. 模块详细设计

### 4.1 `config.py` — 角色化模型配置

DeepSeek 通过 OpenAI 兼容接口接入（`base_url = "https://api.deepseek.com"`），无需额外 SDK。

**推荐默认角色分配（国产优先）**：

| 角色 | 推荐模型 | 理由 |
|------|---------|------|
| executor | `deepseek-chat`（V3/Pro） | 批量提取，性价比极高，中文投资书支持好 |
| judge | `deepseek-reasoner`（R1） | 推理能力强，与 executor 同厂但不同系列（reasoning vs chat） |
| examiner | `deepseek-chat` | 出题难度不高，便宜够用 |
| utility | `deepseek-chat`（flash 档） | 流派识别等廉价杂活 |

> 注：若希望 judge 与 executor 完全异源（跨厂商交叉验证），可将 judge 改为 Claude Sonnet，由环境变量控制，不需改代码。

```python
import os

# ============ 角色模型配置（优先读环境变量，默认国产 DeepSeek）============
ROLE_MODELS = {
    # 执行者：批量章节提取，量大需性价比，DeepSeek V3 已足够
    "executor": {
        "provider": os.getenv("EXECUTOR_PROVIDER", "deepseek"),
        "model":    os.getenv("EXECUTOR_MODEL",    "deepseek-chat"),
    },
    # 评委：质量关口，推荐 R1（推理型）或跨厂商 Claude Sonnet
    "judge": {
        "provider": os.getenv("JUDGE_PROVIDER", "deepseek"),
        "model":    os.getenv("JUDGE_MODEL",    "deepseek-reasoner"),
    },
    # 出题者：从原文出题，与 judge 同厂保持标准一致
    "examiner": {
        "provider": os.getenv("EXAMINER_PROVIDER", "deepseek"),
        "model":    os.getenv("EXAMINER_MODEL",    "deepseek-chat"),
    },
    # 杂活：流派识别、L2 概念抽取，最便宜即可
    "utility": {
        "provider": os.getenv("UTILITY_PROVIDER", "deepseek"),
        "model":    os.getenv("UTILITY_MODEL",    "deepseek-chat"),
    },
}

# executor 与 judge 同源时打警告（同为 deepseek-chat 时），不阻断运行
def validate_role_config() -> list[str]:
    warnings = []
    e = ROLE_MODELS["executor"]
    j = ROLE_MODELS["judge"]
    if e["provider"] == j["provider"] and e["model"] == j["model"]:
        warnings.append(
            "⚠️  Executor 与 Judge 完全相同，跨模型交叉验证失效。"
            "建议 judge 使用 deepseek-reasoner 或 claude-sonnet-4-5"
        )
    return warnings

# 质量阈值
L1_THRESHOLD = 0.95
L2_THRESHOLD = 0.80
L3_THRESHOLD = 0.75
MAX_RETRIES = 2
MAX_COST_USD = 5.0          # 成本熔断上限（DeepSeek 价格低，实际消耗远低于此）

# 每章题型分布：numeric×2 + direction×2 + factual×1
QA_QUESTIONS_PER_CHAPTER = 5
QA_TYPE_DISTRIBUTION = {"numeric": 2, "direction": 2, "factual": 1}

# token 预算
MAX_CHAPTER_CHARS = 12000
CHAPTER_SUMMARY_TOKENS = 1200
SKILL_MD_TOKENS = 4000

# 投资四大必须覆盖的维度
REQUIRED_DIMENSIONS = ["selection_criteria", "red_flags", "valuation_methods", "checklists"]

# 成本定价表（$/1M tokens，DeepSeek 按官网定价）
PRICING = {
    # DeepSeek
    "deepseek-chat":     {"in": 0.27, "out": 1.10},   # V3，缓存命中更便宜
    "deepseek-reasoner": {"in": 0.55, "out": 2.19},   # R1
    # Anthropic（备用 Judge）
    "claude-sonnet-4-5":         {"in": 3.00, "out": 15.00},
    "claude-3-5-haiku-20241022": {"in": 0.80, "out": 4.00},
    # OpenAI（备用）
    "gpt-4o":      {"in": 2.50, "out": 10.00},
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
}
```

### 4.2 `llm_client.py` — LLMClient + ModelPool

DeepSeek 兼容 OpenAI 接口，`provider="deepseek"` 时复用 `openai.OpenAI` 客户端，只需替换 `base_url`：

```python
PROVIDER_CONFIG = {
    "deepseek":  {"base_url": "https://api.deepseek.com",   "env_key": "DEEPSEEK_API_KEY"},
    "openai":    {"base_url": "https://api.openai.com/v1",  "env_key": "OPENAI_API_KEY"},
    "anthropic": {"base_url": None,                          "env_key": "ANTHROPIC_API_KEY"},
}

class LLMClient:
    def __init__(self, provider: str, model: str, role: str = "")
    # deepseek / openai → openai.OpenAI(base_url=..., api_key=...)
    # anthropic → anthropic.Anthropic(api_key=...)
    # 统计 in_tokens / out_tokens，支持 cost() 计算

    def complete(self, prompt: str, system: str = "", max_tokens: int = 2000,
                 temperature: float = 0.0) -> str
    # deepseek-reasoner 特殊处理：R1 不支持 system prompt，
    # 需将 system 内容合并到 user message 首行

    def cost(self) -> float

class ModelPool:
    def get(self, role: str) -> LLMClient
    def total_cost_sum(self) -> float
    def report(self) -> str  # 各角色 token/成本明细
```

**DeepSeek R1 特殊说明**：`deepseek-reasoner` 返回 `reasoning_content`（思考过程）+ `content`（最终答案），`complete()` 只返回 `content`，但可选择性记录 reasoning 用于调试。

### 4.3 `cache.py` — 磁盘缓存

```python
# 存储路径：.cache/extractions/<hash16>.json
# 重试时：weak_chapters 重新调用，其余章节直接读缓存

class ExtractionCache:
    def get_key(self, text: str) -> str          # md5(text)[:16]
    def get(self, key: str) -> ChapterSummary | None
    def set(self, key: str, value: ChapterSummary) -> None
```

### 4.4 `extractor.py` — 文本提取（含容错）

```python
@dataclass
class Chapter:
    index: int
    title: str
    content: str        # 原文
    token_count: int

@dataclass
class ExtractionResult:
    chapters: List[Chapter]
    warnings: List[str]   # 章节质量告警

class BookExtractor:
    def extract(self, source: str) -> ExtractionResult
    # PDF / EPUB（MVP 即支持）/ URL（下载后走 PDF 流程）

    def _split_chapters(self, text: str) -> List[Chapter]
    # 优先正则匹配标题（支持 Chapter N / 第N章 / CHAPTER FOURTEEN / 第十四章）
    # 失败 → token 滑动分块降级（3000/块，overlap 200）

    def _validate_chapters(self, chapters) -> List[str]
    # 告警：≤1块 / 单章<200 token / 单章>均值×5
```

### 4.5 `summarizer.py` — Executor 提取 Agent

```python
PROMPT_TEMPLATES = {
    "value":   VALUE_INVESTING_PROMPT,   # 重点：PE/PB/安全边际/护城河
    "growth":  GROWTH_INVESTING_PROMPT,  # 重点：PEG/成长性/行业空间
    "quant":   QUANT_INVESTING_PROMPT,   # 重点：因子定义/信号强度
    "macro":   MACRO_INVESTING_PROMPT,   # 重点：周期/资产配置比例
    "general": GENERAL_PROMPT,
}

@dataclass
class ChapterSummary:
    chapter_index: int
    title: str
    core_arguments: List[str]
    evidence: List[str]
    selection_criteria: List[str]
    red_flags: List[str]
    valuation_methods: List[str]
    rules: List[str]       # if-then 决策规则

@dataclass
class ExtractionOutput:
    chapter_summaries: List[ChapterSummary]
    skill_md: str
    criteria: str
    red_flags: str
    valuation: str
    checklists: str
    rules_json: dict

class BookSummarizer:
    def summarize(self, chapters: List[Chapter],
                  feedbacks: List[WeakChapterFeedback] = None) -> ExtractionOutput

    def _detect_strategy_type(self, chapters) -> str
    # utility 模型，看前2章+目录分类

    def _summarize_chapter(self, chapter: Chapter,
                           feedback: WeakChapterFeedback = None) -> ChapterSummary
    # feedback 不为 None 时 prompt 中补充：
    # "上次遗漏：{missing_dimensions}，答错题目：{failed_questions}"

    def _generate_rules_json(self, summaries) -> dict
    # 只提取有明确数值阈值的规则，模糊描述不输出
    # 输出格式见 5. 输出格式
```

### 4.6 `evaluator.py` — Judge + 三层评估

```python
@dataclass
class ScenarioQA:
    question: str
    gold_answer: str        # Examiner 从原文同步生成的标准答案
    evidence_span: str      # 支撑答案的原文片段（判分溯源）
    answer_type: str        # "numeric" | "direction" | "factual"
    chapter_index: int

@dataclass
class WeakChapterFeedback:
    index: int
    missing_dimensions: List[str]     # 如 ["valuation_methods"]
    failed_questions: List[str]       # 答错的题目文本
    hint: str

@dataclass
class QualityReport:
    l1_token_coverage: float
    l2_dimension_recall: float
    l3_qa_score: float
    l3_breakdown: dict      # {"numeric": 0.9, "direction": 0.75, "factual": 0.8}
    overall_score: float
    passed: bool
    chapter_scores: List[dict]
    weak_chapters: List[int]
    feedbacks: List[WeakChapterFeedback]
    extraction_warnings: List[str]
    cost_usd: float

class Judge:
    """与 Executor 异源的独立评委，只信原文"""
    def __init__(self, pool: ModelPool)
    # examiner = pool.get("examiner")  # 出题：看原文
    # judge    = pool.get("judge")     # 判分：看原文

    def generate_questions(self, chapter: Chapter) -> List[ScenarioQA]
    # Examiner 从原文出题，同时产出 gold_answer + evidence_span
    # 题型分布按 config.QA_TYPE_DISTRIBUTION

    def grade(self, qa: ScenarioQA, answer: str) -> tuple[bool, str]
    # numeric：正则抽数字 → 代码比对（±2% 容差），不调 LLM
    # direction：Judge LLM 判，方向错直接错，防止废话淹没结论
    # factual：Judge LLM 语义判，核心事实正确即算对

    def grade_with_consensus(self, qa: ScenarioQA, answer: str) -> tuple[bool, str]
    # 边界 case（分数在阈值 ±5% 内）启用双 Judge 投票
    # 两个 Judge 分歧时保守判错（存疑从严）

class QualityEvaluator:
    def evaluate(self, chapters, result, extraction_warnings) -> QualityReport

    # L1：纯规则，统计 Skill 文件覆盖的原文 token 比例
    def _eval_token_coverage(self, chapters, result) -> float

    # L2：检查四大维度是否非空（utility 辅助判断质量）
    def _eval_dimension_recall(self, chapters, result) -> tuple[float, dict]

    # L3：Examiner 出题（看原文）→ Executor 答（看 Skill）→ Judge 判
    def _eval_qa(self, chapters, result) -> tuple[float, List[dict], List[WeakChapterFeedback]]
```

### 4.7 `rule_engine.py` — 量化规则引擎

```python
# 直接执行 rules.json，支持：单值比较 / expr 表达式 / 数据缺失标记

@dataclass
class RuleResult:
    rule_id: str
    passed: bool | None    # None = 数据缺失无法判断
    input_value: float | None
    threshold: float
    source_chapter: str

class RuleEngine:
    def __init__(self, rules_path: str)
    def evaluate(self, metrics: dict) -> List[RuleResult]
    # metrics: {"PE": 30, "PB": 8, "current_ratio": 2.1, ...}
    # 数据缺失时 passed=None，不强行给结论
```

### 4.8 `investment_advisor.py` — 投资顾问 Agent

```python
@dataclass
class InvestmentReport:
    target: str
    framework_source: str
    overall_verdict: str    # "BUY"|"HOLD"|"AVOID"|"INSUFFICIENT_DATA"
    score: int | None
    rule_results: List[RuleResult]      # 代码执行的量化结果
    qualitative_analysis: str           # Judge 处理的定性部分
    matched_criteria: List[str]
    triggered_red_flags: List[str]
    valuation_summary: str
    missing_metrics: List[str]          # 缺失指标列表
    data_as_of: str
    data_source: str
    confidence: float                   # 因数据缺失降低的置信度
    disclaimer: str = "本分析仅为书中框架的机械应用，不构成投资建议"

class InvestmentAdvisor:
    def analyze(self, target: str, metrics: dict, context: str = "") -> InvestmentReport
    # 1. rule_engine 先执行所有量化规则（代码比对）
    # 2. 关键指标缺失 >50% → verdict = "INSUFFICIENT_DATA"，停止
    # 3. Judge 处理定性部分（护城河/管理层等无法量化的内容）
    # 4. 汇总输出，必含数据时点和免责声明
```

### 4.9 `book_agent.py` — 主入口

```python
# CLI 子命令：
# python book_agent.py extract <source> [--output ./output]
# python book_agent.py advise <skill_dir> <target> --pe 30 --pb 8

def cmd_extract(source, output_dir):
    pool = ModelPool()
    for w in config.validate_role_config(): print(w)

    chapters = BookExtractor().extract(source)
    summarizer = BookSummarizer(pool)
    evaluator  = QualityEvaluator(pool)

    feedbacks = None
    for attempt in range(config.MAX_RETRIES + 1):
        # 成本熔断
        if pool.total_cost_sum() > config.MAX_COST_USD:
            print(f"⚠️ 成本超限 ${pool.total_cost_sum():.2f}，强制停止")
            save_output(result, report); break

        result = summarizer.summarize(chapters.chapters, feedbacks=feedbacks)
        report = evaluator.evaluate(chapters.chapters, result, chapters.warnings)

        print(f"[尝试{attempt+1}] L1={report.l1_token_coverage:.0%} "
              f"L2={report.l2_dimension_recall:.0%} L3={report.l3_qa_score:.0%} "
              f"成本=${report.cost_usd:.2f}")

        if report.passed:
            save_output(result, report); break

        feedbacks = report.feedbacks   # 携带具体缺失反馈重试
        print(f"  薄弱章节: {report.weak_chapters}")
        if attempt == config.MAX_RETRIES:
            save_output(result, report)  # 超限仍保存，标记未达标

    print(pool.report())  # 各角色成本明细
```

---

<!-- anchor:output-format -->
## 5. 输出格式

### `rules.json`
```json
{
  "strategy_type": "value",
  "book": "The Intelligent Investor",
  "valuation_rules": [
    {"id": "graham_pe",    "metric": "PE",    "op": "<=", "value": 15,   "source": "ch14"},
    {"id": "graham_combo", "expr": "PE * PB", "op": "<=", "value": 22.5, "source": "ch14"}
  ],
  "selection_criteria": [
    {"id": "current_ratio", "metric": "current_ratio", "op": ">=", "value": 2.0, "source": "ch14"},
    {"id": "eps_growth_10y","metric": "eps_growth_10y","op": ">=", "value": 0.33,"source": "ch14"}
  ],
  "red_flag_rules": [
    {"id": "any_loss_10y",  "metric": "min_annual_eps_10y", "op": ">=", "value": 0, "source": "ch14"}
  ]
}
```

### `quality_report.json`
```json
{
  "book": "The Intelligent Investor",
  "strategy_type": "value",
  "extraction_warnings": [],
  "l1_token_coverage": 0.97,
  "l2_dimension_recall": 0.85,
  "l3_qa_score": 0.80,
  "l3_breakdown": {"numeric": 0.90, "direction": 0.75, "factual": 0.80},
  "overall_score": 0.88,
  "passed": true,
  "attempts": 1,
  "chapter_scores": [
    {"index": 14, "title": "Stock Selection", "qa_score": 0.90},
    {"index": 8,  "title": "Market Fluctuations", "qa_score": 0.75}
  ],
  "weak_chapters": [],
  "role_costs": {"executor": 0.62, "examiner": 0.18, "judge": 0.41, "utility": 0.04},
  "total_cost_usd": 1.25
}
```

### `investment_advisor` 输出示例
```
目标：贵州茅台 (600519) | 数据时点：2024-Q3 | 框架：《聪明的投资者》

总评：HOLD（观望）| 评分：62/100 | 置信度：0.85

── 量化规则（代码执行）────────────────────────────
✅ 企业规模      revenue ≥ $1亿       实际：远超     [ch14]
✅ 连续盈利10年  min_annual_eps ≥ 0   实际：✓        [ch14]
❌ PE 估值       PE ≤ 15              实际：30        [ch14] ← 超出
❌ PE×PB 组合   PE×PB ≤ 22.5         实际：240       [ch14] ← 严重超出
⚠️ 流动比率      数据缺失，无法判断                    [ch14]

── 定性分析（Judge）───────────────────────────────
护城河：极强（品牌壁垒+渠道控制+文化属性）
管理层：稳定，无重大负面记录

── 结论 ─────────────────────────────────────────
公司质地极佳，但估值严重超出安全边际。
建议等待 PE 回落至 15 以下再考虑买入。

⚠️ 本分析仅为书中框架的机械应用，不构成投资建议
```

---

<!-- anchor:testing -->
## 6. 测试策略

### 黄金测试集 `tests/golden/intelligent_investor_qa.json`
人工标注 15-20 道题（三种题型各覆盖），每次修改 prompt 后回归，确认 evaluator 未退化：
```json
[
  {
    "question": "一家公司过去10年有2年亏损，格雷厄姆防御型标准怎么判？",
    "gold_answer": "不符合，要求过去10年每年均有盈利",
    "evidence_span": "an uninterrupted record of dividend payments for at least the past 20 years",
    "answer_type": "direction",
    "expected_verdict": "AVOID",
    "source_chapter": 14
  }
]
```

### 端到端验证
- 用《聪明的投资者》PDF 跑全流程，验证 `rules.json` 能正确提取所有已知阈值
- 用 `investment_advisor` 分析茅台/中石油，人工对比结论是否合理
- 运行 `python book_agent.py advise` 确认数据缺失时输出 `INSUFFICIENT_DATA`

---

<!-- anchor:dependencies -->
## 7. 依赖

```
anthropic>=0.25.0
openai>=1.0.0
pypdf>=4.0.0
ebooklib>=0.18
beautifulsoup4>=4.12.0
requests>=2.31.0
pydantic>=2.0.0
tqdm>=4.65.0
sentence-transformers>=2.6.0   # L3 factual 题语义相似度
```

可选：`pdfminer.six`、`docling`（技术类/公式密集书籍更佳）

---

<!-- anchor:implementation-order -->
## 8. 实现顺序

1. `config.py` — 角色化模型配置 + 成本定价表 + `validate_role_config()`
2. `llm_client.py` — `LLMClient` + `ModelPool`（角色化工厂，统计成本）
3. `cache.py` — 磁盘缓存
4. `extractor.py` — PDF + EPUB + URL，章节分割容错降级
5. `summarizer.py` — 流派识别 + 按模板提取 + `rules.json` 生成（Executor 角色）
6. `evaluator.py` — `Judge` 类（出题/判分）+ `QualityEvaluator` 三层评估
7. `rule_engine.py` — 量化规则执行引擎
8. `book_agent.py` — 主流程（成本熔断 + 带反馈重试）
9. `investment_advisor.py` — 规则引擎 + Judge 定性 + INSUFFICIENT_DATA 处理
10. `tests/golden/` — 黄金测试集 + 回归脚本
11. 端到端验证（《聪明的投资者》全流程）

---

## 9. 双模型架构注意事项

| 风险 | 处理方式 |
|------|---------|
| Judge 自身有偏见 | 边界 case（±5% 阈值附近）启用双 Judge 投票，分歧时保守判错 |
| Executor 用废话淹没结论 | direction 题 prompt 明确：「只提取最终方向，忽略其余修饰，模糊不给方向判错」 |
| 成本失控 | `MAX_COST_USD=5.0` 熔断，每次迭代前检查 |
| examiner+judge 同源质疑 | 故意设计：同源保持判分标准自洽，关键隔离在 executor ↔ judge |
