"""
LLMClient + ModelPool：角色化多模型管理，统计 token/成本
DeepSeek 兼容 OpenAI 接口，provider="deepseek" 时复用 openai.OpenAI 客户端
"""

import time
import logging
from typing import Optional
from dataclasses import dataclass, field

import httpx
from openai import OpenAI
from anthropic import Anthropic

from config import ROLE_MODELS, PROVIDER_CONFIG, PRICING

logger = logging.getLogger(__name__)


@dataclass
class UsageStats:
    """单次调用的用量统计"""
    in_tokens: int = 0
    out_tokens: int = 0


class LLMClient:
    """单个 LLM 客户端，绑定特定 provider + model，自动统计成本"""

    def __init__(self, provider: str, model: str, role: str = ""):
        self.provider = provider
        self.model = model
        self.role = role
        self.usage = UsageStats()
        self._client: Optional[OpenAI | Anthropic] = None
        self._init_client()

    def _init_client(self):
        cfg = PROVIDER_CONFIG[self.provider]
        api_key = cfg["env_key"]
        key_val = __import__("os").getenv(api_key, "")

        if self.provider in ("deepseek", "openai"):
            base_url = cfg["base_url"]
            self._client = OpenAI(
                api_key=key_val,
                base_url=base_url,
                timeout=httpx.Timeout(connect=10.0, read=120.0, write=120.0, pool=10.0),
                max_retries=1,
            )
        elif self.provider == "anthropic":
            self._client = Anthropic(api_key=key_val)
        else:
            raise ValueError(f"不支持的 provider: {self.provider}")

    def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 2000,
        temperature: float = 0.0,
    ) -> str:
        """
        调用 LLM 完成请求，返回文本结果。
        deepseek-reasoner 特殊处理：R1 不支持 system prompt，
        将 system 内容合并到 user message 首行。
        """
        if self.provider == "anthropic":
            return self._complete_anthropic(prompt, system, max_tokens, temperature)
        else:
            return self._complete_openai_compat(prompt, system, max_tokens, temperature)

    def _complete_openai_compat(
        self, prompt: str, system: str, max_tokens: int, temperature: float
    ) -> str:
        messages = []
        # deepseek-reasoner (R1) 不支持 system prompt，需合并到 user message
        # deepseek-v4 系列原生支持 system prompt，无需特殊处理
        if self.model == "deepseek-reasoner":
            if system:
                prompt = f"[System Instruction]\n{system}\n\n[User Query]\n{prompt}"
        else:
            if system:
                messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        self.usage.in_tokens += resp.usage.prompt_tokens or 0
        self.usage.out_tokens += resp.usage.completion_tokens or 0

        choice = resp.choices[0]
        content = choice.message.content or ""

        # R1 的 reasoning_content 可选记录（调试用）
        if hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content:
            logger.debug(
                f"[{self.role}] R1 reasoning: {choice.message.reasoning_content[:200]}..."
            )

        return content

    def _complete_anthropic(
        self, prompt: str, system: str, max_tokens: int, temperature: float
    ) -> str:
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        resp = self._client.messages.create(**kwargs)

        self.usage.in_tokens += resp.usage.input_tokens or 0
        self.usage.out_tokens += resp.usage.output_tokens or 0

        return resp.content[0].text if resp.content else ""

    def cost(self) -> float:
        """计算当前累计成本（USD）"""
        pricing = PRICING.get(self.model, PRICING.get("gpt-4o-mini", {"in": 0.15, "out": 0.60}))
        in_cost = (self.usage.in_tokens / 1_000_000) * pricing["in"]
        out_cost = (self.usage.out_tokens / 1_000_000) * pricing["out"]
        return in_cost + out_cost

    def reset_usage(self):
        self.usage = UsageStats()


class ModelPool:
    """角色化模型池：按 role 获取 LLMClient，全局统计成本"""

    def __init__(self):
        self._clients: dict[str, LLMClient] = {}

    def get(self, role: str) -> LLMClient:
        """按角色获取 LLMClient，首次调用时创建并缓存"""
        if role not in self._clients:
            cfg = ROLE_MODELS[role]
            self._clients[role] = LLMClient(
                provider=cfg["provider"],
                model=cfg["model"],
                role=role,
            )
        return self._clients[role]

    def total_cost_sum(self) -> float:
        """所有角色的总成本"""
        return sum(c.cost() for c in self._clients.values())

    def report(self) -> str:
        """各角色 token/成本明细"""
        lines = ["── 角色成本明细 ──"]
        total_in = 0
        total_out = 0
        total_cost = 0.0

        for role, client in self._clients.items():
            in_t = client.usage.in_tokens
            out_t = client.usage.out_tokens
            c = client.cost()
            lines.append(
                f"  {role:12s} ({client.model:22s})  "
                f"in={in_t:>8,}  out={out_t:>8,}  ${c:.4f}"
            )
            total_in += in_t
            total_out += out_t
            total_cost += c

        lines.append(f"  {'总计':12s} {'':22s}  "
                      f"in={total_in:>8,}  out={total_out:>8,}  ${total_cost:.4f}")
        return "\n".join(lines)
