import os
from typing import Optional
from openai import OpenAI
from core.llm import AgentsLLM

class ExpandsLLM(AgentsLLM):
    """
    拓展LLM客户端，通过继承增加了对Gemini, Anthropic的支持。
    """

    def __init__(
            self,
            model: Optional[str] = None,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            provider: Optional[str] = "auto",
            **kwargs
    ):
        if provider == "Gemini":
            print("正在使用自定义的 Gemini Provider")
            self.provider = "Gemini"

            # 解析 ModelScope 的凭证
            self.api_key = api_key or os.getenv("GEMINI_API_KEY")
            self.base_url = base_url or "https://generativelanguage.googleapis.com/v1beta/"

            # 验证凭证是否存在
            if not self.api_key:
                raise ValueError("Gemini API key not found. Please set GEMINI_API_KEY environment variable.")

            # 设置默认模型和其他参数
            self.model = model or os.getenv("LLM_MODEL_ID") or "google/gemini-2.5-flash"
            self.temperature = kwargs.get('temperature', 0.7)
            self.max_tokens = kwargs.get('max_tokens')
            self.timeout = kwargs.get('timeout', 60)

            # 使用获取的参数创建OpenAI客户端实例
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

        elif provider == "Anthropic":
            print("正在使用自定义的 Anthropic Provider")
            self.provider = "Anthropic"

            # 解析 ModelScope 的凭证
            self.api_key = api_key or os.getenv("Anthropic_API_KEY")
            self.base_url = base_url or "https://generativelanguage.googleapis.com/v1beta/"

            # 验证凭证是否存在
            if not self.api_key:
                raise ValueError("Anthropic API key not found. Please set Anthropic_API_KEY environment variable.")

            # 设置默认模型和其他参数
            self.model = model or os.getenv("LLM_MODEL_ID") or "claude-v1"
            self.temperature = kwargs.get('temperature', 0.7)
            self.max_tokens = kwargs.get('max_tokens')
            self.timeout = kwargs.get('timeout', 60)

            # 使用获取的参数创建OpenAI客户端实例
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        else:
            # 如果不是 上述, 则完全使用父类的原始逻辑来处理
            super().__init__(model=model, api_key=api_key, base_url=base_url, provider=provider, **kwargs)
