# basic calling for llms, for example with openai api
import json
import os
import time
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
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        max_retries: int = 3,
        retry_base_delay_seconds: float = 1.0,
        retry_max_delay_seconds: float = 30.0,
        request_timeout_seconds: float = 300.0,
    ):
        """
        Initialize LLM client.

        Args:
            provider: "openai", "anthropic", "google", or "gemini"
            model: Model name (e.g., "gpt-4o", "claude-3-5-sonnet-20241022", "gemini-1.5-flash")
            api_key: API key (if None, will use environment variable)
            temperature: Optional sampling temperature override. If None, provider/model default is used.
            max_tokens: Maximum tokens in response
            max_retries: Maximum number of retries for transient API errors
            retry_base_delay_seconds: Base backoff delay in seconds
            retry_max_delay_seconds: Maximum backoff delay in seconds
            request_timeout_seconds: Per-request timeout for provider API calls
        """
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_base_delay_seconds = retry_base_delay_seconds
        self.retry_max_delay_seconds = retry_max_delay_seconds
        self.request_timeout_seconds = request_timeout_seconds

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
            use_vertex_ai = self._resolve_google_vertexai_mode()

            if use_vertex_ai:
                if google_genai_sdk is None:
                    raise ImportError("Vertex AI mode requires google-genai. Install with: pip install google-genai")
                project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GOOGLE_PROJECT_ID")
                location = os.getenv("GOOGLE_CLOUD_LOCATION") or os.getenv("GOOGLE_LOCATION") or "us-central1"
                if not project:
                    raise ValueError("GOOGLE_CLOUD_PROJECT (or GOOGLE_PROJECT_ID) not found in environment")

                self.google_sdk = "google_genai"
                client_kwargs = {
                    "vertexai": True,
                    "project": project,
                    "location": location,
                }
                timeout_seconds = self._resolve_timeout_seconds()
                if timeout_seconds is not None and hasattr(google_genai_sdk, "types"):
                    # google-genai HttpOptions.timeout is in milliseconds.
                    timeout_millis = max(int(timeout_seconds * 1000.0), 10_000)
                    client_kwargs["http_options"] = google_genai_sdk.types.HttpOptions(timeout=timeout_millis)
                self.client = google_genai_sdk.Client(**client_kwargs)
            else:
                if not api_key:
                    raise ValueError(
                        "GOOGLE_API_KEY (or GEMINI_API_KEY) not found in environment. "
                        "For Vertex AI, set GOOGLE_GENAI_USE_VERTEXAI=true and configure GOOGLE_CLOUD_PROJECT."
                    )
                if google_genai_sdk is not None:
                    self.google_sdk = "google_genai"
                    client_kwargs = {"api_key": api_key}
                    timeout_seconds = self._resolve_timeout_seconds()
                    if timeout_seconds is not None and hasattr(google_genai_sdk, "types"):
                        # google-genai HttpOptions.timeout is in milliseconds.
                        timeout_millis = max(int(timeout_seconds * 1000.0), 10_000)
                        client_kwargs["http_options"] = google_genai_sdk.types.HttpOptions(timeout=timeout_millis)
                    self.client = google_genai_sdk.Client(**client_kwargs)
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

        def _single_attempt() -> Dict[str, Any]:
            if self.provider == "anthropic":
                return self._chat_anthropic(prompt, system_prompt, response_format, **kwargs)
            if self.provider == "openai":
                return self._chat_openai(prompt, system_prompt, response_format, **kwargs)
            if self.provider in {"google", "gemini"}:
                return self._chat_gemini(prompt, system_prompt, response_format, **kwargs)
            raise ValueError(f"Unsupported provider: {self.provider}")

        return self._chat_with_retries(_single_attempt)

    def _chat_with_retries(self, chat_call) -> Dict[str, Any]:
        """Retry transient provider failures with exponential backoff."""
        for attempt in range(self.max_retries + 1):
            try:
                return chat_call()
            except Exception as e:
                if attempt >= self.max_retries or not self._is_retryable_error(e):
                    raise
                delay_seconds = self._compute_retry_delay_seconds(e, attempt)
                print(
                    f"[LLMClient] transient API error from {self.provider} "
                    f"(attempt {attempt + 1}/{self.max_retries + 1}), retrying in {delay_seconds:.1f}s: {e}"
                )
                time.sleep(delay_seconds)

        # Unreachable because loop either returns or raises.
        raise RuntimeError("LLM retry loop terminated unexpectedly.")

    def _is_retryable_error(self, error: Exception) -> bool:
        """Detect provider errors that should be retried."""
        status_code = self._extract_status_code(error)
        if status_code in {408, 409, 429, 500, 502, 503, 504}:
            return True

        message = str(error).lower()
        retry_markers = (
            "rate limit",
            "too many requests",
            "resource_exhausted",
            "high demand",
            "deadline exceeded",
            "timed out",
            "timeout",
            "please try again later",
            "temporarily unavailable",
            "service unavailable",
            "status': 'unavailable'",
            '"status": "unavailable"',
        )
        return any(marker in message for marker in retry_markers)

    def _resolve_timeout_seconds(self, **kwargs) -> Optional[float]:
        timeout_value = kwargs.get("timeout_seconds", kwargs.get("timeout", self.request_timeout_seconds))
        if timeout_value is None:
            return None
        try:
            timeout_seconds = float(timeout_value)
        except (TypeError, ValueError):
            return None
        if timeout_seconds <= 0:
            return None
        return timeout_seconds

    def _resolve_temperature(self, **kwargs) -> Optional[float]:
        if "temperature" in kwargs:
            temperature_value = kwargs.get("temperature")
        else:
            temperature_value = self.temperature

        if temperature_value is None:
            return None
        try:
            parsed = float(temperature_value)
        except (TypeError, ValueError):
            return None
        return parsed

    def _resolve_google_vertexai_mode(self) -> bool:
        """Whether Gemini client should use Vertex AI credentials flow."""
        env_value = os.getenv("GOOGLE_GENAI_USE_VERTEXAI")
        if env_value is not None:
            return env_value.strip().lower() in {"1", "true", "yes", "on"}

        # Auto-enable Vertex mode when a project is configured and no API key is present.
        project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GOOGLE_PROJECT_ID")
        has_api_key = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
        return bool(project) and not has_api_key

    def _extract_status_code(self, error: Exception) -> Optional[int]:
        """Extract HTTP status code from SDK exceptions when available."""
        status_code = getattr(error, "status_code", None)
        if isinstance(status_code, int):
            return status_code

        response = getattr(error, "response", None)
        if response is not None:
            response_status = getattr(response, "status_code", None)
            if isinstance(response_status, int):
                return response_status

        return None

    def _extract_retry_after_seconds(self, error: Exception) -> Optional[float]:
        """Extract Retry-After hints from exception response headers when available."""
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        if headers is None:
            return None

        retry_after_ms = headers.get("retry-after-ms")
        if retry_after_ms is not None:
            try:
                parsed = float(retry_after_ms) / 1000.0
                if parsed > 0:
                    return parsed
            except (TypeError, ValueError):
                pass

        retry_after = headers.get("retry-after")
        if retry_after is not None:
            try:
                parsed = float(retry_after)
                if parsed > 0:
                    return parsed
            except (TypeError, ValueError):
                pass

        return None

    def _compute_retry_delay_seconds(self, error: Exception, attempt: int) -> float:
        """Use Retry-After when present, otherwise exponential backoff."""
        retry_after_seconds = self._extract_retry_after_seconds(error)
        if retry_after_seconds is not None:
            return min(retry_after_seconds, self.retry_max_delay_seconds)

        delay = self.retry_base_delay_seconds * (2**attempt)
        return min(delay, self.retry_max_delay_seconds)

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

        # Some models (gpt-5*, o1*) only support default temperature.
        # For other models, only set temperature when explicitly provided.
        temperature = self._resolve_temperature(**kwargs)
        if temperature is not None and not (self.model.startswith("gpt-5") or self.model.startswith("o1")):
            request_params["temperature"] = temperature

        # Determine which token parameter to use based on model
        # Newer models (gpt-5*, o1*) use max_completion_tokens
        # Older models use max_tokens
        max_tokens_value = kwargs.get("max_tokens", self.max_tokens)
        if self.model.startswith("gpt-5") or self.model.startswith("o1"):
            request_params["max_completion_tokens"] = max_tokens_value
        else:
            request_params["max_tokens"] = max_tokens_value

        timeout_seconds = self._resolve_timeout_seconds(**kwargs)
        if timeout_seconds is not None:
            request_params["timeout"] = timeout_seconds

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
            "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        temperature = self._resolve_temperature(**kwargs)
        if temperature is not None:
            config_kwargs["temperature"] = temperature
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
            "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        temperature = self._resolve_temperature(**kwargs)
        if temperature is not None:
            generation_config["temperature"] = temperature
        if response_format == "json":
            generation_config["response_mime_type"] = "application/json"

        if system_prompt:
            model = self.client.GenerativeModel(
                model_name=self.model,
                system_instruction=system_prompt,
            )
        else:
            model = self.client.GenerativeModel(model_name=self.model)

        generate_content_kwargs: Dict[str, Any] = {
            "generation_config": generation_config,
        }
        timeout_seconds = self._resolve_timeout_seconds(**kwargs)
        if timeout_seconds is not None:
            generate_content_kwargs["request_options"] = {"timeout": timeout_seconds}

        response = model.generate_content(
            prompt,
            **generate_content_kwargs,
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
