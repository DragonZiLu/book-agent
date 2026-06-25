"""
角色化模型配置 + 成本定价表 + 质量阈值 + validate_role_config()
DeepSeek 通过 OpenAI 兼容接口接入（base_url = "https://api.deepseek.com"）
"""

import os
from typing import Dict

# ============ 角色模型配置（优先读环境变量，默认国产 DeepSeek）============
RoleConfig = Dict[str, str]

ROLE_MODELS: Dict[str, RoleConfig] = {
    # 执行者：批量章节提取，量大需性价比
    "executor": {
        "provider": os.getenv("EXECUTOR_PROVIDER", "deepseek"),
        "model":    os.getenv("EXECUTOR_MODEL",    "deepseek-v4-flash"),
    },
    # 评委：质量关口，与 executor 异源（flash vs pro）
    "judge": {
        "provider": os.getenv("JUDGE_PROVIDER", "deepseek"),
        "model":    os.getenv("JUDGE_MODEL",    "deepseek-v4-pro"),
    },
    # 出题者：从原文出题，与 judge 同模型保持判分标准自洽
    "examiner": {
        "provider": os.getenv("EXAMINER_PROVIDER", "deepseek"),
        "model":    os.getenv("EXAMINER_MODEL",    "deepseek-v4-pro"),
    },
    # 杂活：流派识别、L2 概念抽取，最便宜即可
    "utility": {
        "provider": os.getenv("UTILITY_PROVIDER", "deepseek"),
        "model":    os.getenv("UTILITY_MODEL",    "deepseek-v4-flash"),
    },
}

# ============ 质量阈值 ============
L1_THRESHOLD: float = 0.95       # 原文 token 覆盖率
L2_THRESHOLD: float = 0.80       # 四维度召回率
L3_THRESHOLD: float = 0.75       # QA 准确率
MAX_RETRIES: int = 2             # 最多重试次数
MAX_COST_USD: float = 5.0        # 成本熔断上限

# ============ 每章题型分布 ============
QA_QUESTIONS_PER_CHAPTER: int = 5
QA_TYPE_DISTRIBUTION: Dict[str, int] = {"numeric": 2, "direction": 2, "factual": 1}

# ============ token 预算 ============
MAX_CHAPTER_CHARS: int = 12000
CHAPTER_SUMMARY_TOKENS: int = 1200
SKILL_MD_TOKENS: int = 4000
SLIDING_WINDOW_SIZE: int = 3000   # 滑动分块大小（tokens）
SLIDING_WINDOW_OVERLAP: int = 200

# ============ 投资四大必须覆盖的维度 ============
REQUIRED_DIMENSIONS: list = [
    "selection_criteria",
    "red_flags",
    "valuation_methods",
    "checklists",
]

# ============ 评估器 V2 配置 ============
# 使用 V2 评估器（设为 False 回退到 V1）
USE_EVALUATOR_V2: bool = True

# 并行评估的工作线程数（per-chapter 并行）
MAX_EVAL_WORKERS: int = 4

# L2 各维度权重（投资场景：风险信号 > 选股标准 > 估值 > 检查清单）
L2_DIM_WEIGHTS: dict = {
    "selection_criteria": 0.30,
    "red_flags": 0.35,       # 风险信号权重最高
    "valuation_methods": 0.20,
    "checklists": 0.15,
}

# 硬否决维度（完全为空 → 直接不通过）
VETO_DIMS: list = ["red_flags", "selection_criteria"]

# 软降权维度（召回不足 → 警告但不否决）
WEAK_DIMS: list = ["valuation_methods", "checklists"]

# overall_score 权重（V2: L2 权重提高，因为投资场景召回比准确更危险）
L1_WEIGHT: float = 0.15
L2_WEIGHT: float = 0.40
L3_WEIGHT: float = 0.45

# ============ 成本定价表（$/1M tokens）============
Pricing = Dict[str, Dict[str, float]]

PRICING: Pricing = {
    # DeepSeek V4
    "deepseek-v4-flash": {"in": 0.27, "out": 1.10},
    "deepseek-v4-pro":   {"in": 0.55, "out": 2.19},
    # DeepSeek (legacy)
    "deepseek-chat":     {"in": 0.27, "out": 1.10},
    "deepseek-reasoner": {"in": 0.55, "out": 2.19},
    # Anthropic（备用 Judge）
    "claude-sonnet-4-5":         {"in": 3.00, "out": 15.00},
    "claude-3-5-haiku-20241022": {"in": 0.80, "out": 4.00},
    # OpenAI（备用）
    "gpt-4o":      {"in": 2.50, "out": 10.00},
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
}

# ============ LLM 提供商配置 ============
PROVIDER_CONFIG: Dict[str, Dict[str, str | None]] = {
    "deepseek":  {"base_url": "https://api.deepseek.com",   "env_key": "DEEPSEEK_API_KEY"},
    "openai":    {"base_url": "https://api.openai.com/v1",  "env_key": "OPENAI_API_KEY"},
    "anthropic": {"base_url": None,                          "env_key": "ANTHROPIC_API_KEY"},
}


def validate_role_config() -> list[str]:
    """executor 与 judge 同源时打警告，不阻断运行"""
    warnings: list[str] = []
    e = ROLE_MODELS["executor"]
    j = ROLE_MODELS["judge"]

    if e["provider"] == j["provider"] and e["model"] == j["model"]:
        warnings.append(
            "⚠️  Executor 与 Judge 完全相同，跨模型交叉验证失效。"
            "建议 judge 使用 deepseek-v4-pro 或其他模型"
        )
    elif e["provider"] == j["provider"]:
        # deepseek-v4-flash + deepseek-v4-pro 同厂异模型，交叉验证有效
        pass
    return warnings
