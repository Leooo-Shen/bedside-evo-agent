# basic calling for llms, for example with openai api
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import openai
from anthropic import Anthropic

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv

    # Look for .env file in project root
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # python-dotenv not installed, will use system environment variables
    pass


class LLMClient:
    """
    Unified LLM client supporting multiple providers (OpenAI, Anthropic).
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = None,
        api_key: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        """
        Initialize LLM client.

        Args:
            provider: "openai" or "anthropic"
            model: Model name (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
            api_key: API key (if None, will use environment variable)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens

        if self.provider == "anthropic":
            self.model = model or "claude-3-5-sonnet-20241022"
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")
            self.client = Anthropic(api_key=api_key)

        elif self.provider == "openai":
            self.model = model or "gpt-4o-mini"
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            openai.api_key = api_key
            self.client = openai

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def chat(
        self, prompt: str, system_prompt: Optional[str] = None, response_format: str = "text", **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat completion request to the LLM.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            response_format: "text" or "json" for structured output
            **kwargs: Additional provider-specific parameters

        Returns:
            Dictionary with 'content' and optionally 'parsed' (for JSON responses)
        """
        if self.provider == "anthropic":
            return self._chat_anthropic(prompt, system_prompt, response_format, **kwargs)
        elif self.provider == "openai":
            return self._chat_openai(prompt, system_prompt, response_format, **kwargs)

    def _chat_anthropic(
        self, prompt: str, system_prompt: Optional[str], response_format: str, **kwargs
    ) -> Dict[str, Any]:
        """Anthropic-specific chat implementation."""

        raise NotImplementedError("Anthropic chat not implemented yet.")

    def _chat_openai(
        self, prompt: str, system_prompt: Optional[str], response_format: str, **kwargs
    ) -> Dict[str, Any]:
        """OpenAI-specific chat implementation."""

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Build request parameters
        request_params = {
            "model": self.model,
            "messages": messages,
        }

        # Handle temperature parameter
        # Some models (gpt-5*, o1*) only support default temperature (1.0)
        if not (self.model.startswith("gpt-5") or self.model.startswith("o1")):
            request_params["temperature"] = kwargs.get("temperature", self.temperature)

        # Determine which token parameter to use based on model
        # Newer models (gpt-5*, o1*) use max_completion_tokens
        # Older models use max_tokens
        max_tokens_value = kwargs.get("max_tokens", self.max_tokens)
        if self.model.startswith("gpt-5") or self.model.startswith("o1"):
            request_params["max_completion_tokens"] = max_tokens_value
        else:
            request_params["max_tokens"] = max_tokens_value

        # Add JSON mode if requested
        if response_format == "json":
            request_params["response_format"] = {"type": "json_object"}

        # Send request
        response = self.client.chat.completions.create(**request_params)

        result = {
            "content": response.choices[0].message.content,
            "model": response.model,
            "usage": {"input_tokens": response.usage.prompt_tokens, "output_tokens": response.usage.completion_tokens},
        }

        # Parse JSON if requested
        if response_format == "json":
            try:
                result["parsed"] = json.loads(result["content"])
            except json.JSONDecodeError as e:
                result["parse_error"] = str(e)
                result["parsed"] = None

        return result
