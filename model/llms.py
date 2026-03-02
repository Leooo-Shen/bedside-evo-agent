# basic calling for llms, for example with openai api
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import openai
from anthropic import Anthropic

try:
    from google import genai as google_genai_sdk
except ImportError:
    google_genai_sdk = None

try:
    import google.generativeai as google_generativeai_sdk
except ImportError:
    google_generativeai_sdk = None

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
    Unified LLM client supporting multiple providers (OpenAI, Anthropic, Gemini).
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
            provider: "openai", "anthropic", "google", or "gemini"
            model: Model name (e.g., "gpt-4o", "claude-3-5-sonnet-20241022", "gemini-1.5-flash")
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

        elif self.provider in {"google", "gemini"}:
            self.model = model or "gemini-1.5-flash"
            api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY (or GEMINI_API_KEY) not found in environment")
            if google_genai_sdk is not None:
                self.google_sdk = "google_genai"
                self.client = google_genai_sdk.Client(api_key=api_key)
            elif google_generativeai_sdk is not None:
                self.google_sdk = "google_generativeai"
                google_generativeai_sdk.configure(api_key=api_key)
                self.client = google_generativeai_sdk
            else:
                raise ImportError(
                    "No Gemini SDK found. Install one of: pip install google-genai OR pip install google-generativeai"
                )

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
        elif self.provider in {"google", "gemini"}:
            return self._chat_gemini(prompt, system_prompt, response_format, **kwargs)

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

    def _chat_gemini(
        self, prompt: str, system_prompt: Optional[str], response_format: str, **kwargs
    ) -> Dict[str, Any]:
        """Google Gemini-specific chat implementation."""
        try:
            if getattr(self, "google_sdk", None) == "google_genai":
                return self._chat_gemini_google_genai(prompt, system_prompt, response_format, **kwargs)
            return self._chat_gemini_google_generativeai(prompt, system_prompt, response_format, **kwargs)
        except Exception as e:
            err = str(e)
            if "NOT_FOUND" in err and "models/" in err:
                raise ValueError(
                    f"Gemini model '{self.model}' is not available for generateContent. "
                    "Try 'gemini-2.0-flash' or 'gemini-1.5-flash', or choose a model returned by the Gemini ListModels API."
                ) from e
            raise

    def _chat_gemini_google_genai(
        self, prompt: str, system_prompt: Optional[str], response_format: str, **kwargs
    ) -> Dict[str, Any]:
        """Gemini implementation using the google-genai SDK."""
        config_kwargs = {
            "temperature": 1.0,  # recommended by Google
            "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        if response_format == "json":
            config_kwargs["response_mime_type"] = "application/json"
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt

        config = config_kwargs
        if google_genai_sdk is not None and hasattr(google_genai_sdk, "types"):
            config = google_genai_sdk.types.GenerateContentConfig(**config_kwargs)

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )

        content = getattr(response, "text", "") or ""
        usage_metadata = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage_metadata, "prompt_token_count", 0) if usage_metadata else 0
        output_tokens = getattr(usage_metadata, "candidates_token_count", 0) if usage_metadata else 0
        if usage_metadata and output_tokens == 0:
            total_tokens = getattr(usage_metadata, "total_token_count", 0) or 0
            if total_tokens >= input_tokens:
                output_tokens = total_tokens - input_tokens

        result = {
            "content": content,
            "model": getattr(response, "model_version", self.model),
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }

        if response_format == "json":
            try:
                result["parsed"] = json.loads(result["content"])
            except json.JSONDecodeError as e:
                result["parse_error"] = str(e)
                result["parsed"] = None

        return result

    def _chat_gemini_google_generativeai(
        self, prompt: str, system_prompt: Optional[str], response_format: str, **kwargs
    ) -> Dict[str, Any]:
        """Gemini implementation using the google-generativeai SDK."""
        generation_config = {
            "temperature": kwargs.get("temperature", self.temperature),
            "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        if response_format == "json":
            generation_config["response_mime_type"] = "application/json"

        if system_prompt:
            model = self.client.GenerativeModel(
                model_name=self.model,
                system_instruction=system_prompt,
            )
        else:
            model = self.client.GenerativeModel(model_name=self.model)

        response = model.generate_content(
            prompt,
            generation_config=generation_config,
        )

        content = ""
        if getattr(response, "text", None):
            content = response.text
        else:
            candidates = getattr(response, "candidates", []) or []
            for candidate in candidates:
                candidate_content = getattr(candidate, "content", None)
                parts = getattr(candidate_content, "parts", []) if candidate_content is not None else []
                for part in parts:
                    part_text = getattr(part, "text", None)
                    if part_text:
                        content += part_text

        usage_metadata = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage_metadata, "prompt_token_count", 0) if usage_metadata else 0
        output_tokens = getattr(usage_metadata, "candidates_token_count", 0) if usage_metadata else 0
        if usage_metadata and output_tokens == 0:
            total_tokens = getattr(usage_metadata, "total_token_count", 0) or 0
            if total_tokens >= input_tokens:
                output_tokens = total_tokens - input_tokens

        result = {
            "content": content,
            "model": getattr(response, "model_version", self.model),
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }

        if response_format == "json":
            try:
                result["parsed"] = json.loads(result["content"])
            except json.JSONDecodeError as e:
                result["parse_error"] = str(e)
                result["parsed"] = None

        return result
