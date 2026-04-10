"""AI Provider clients for Roundtable."""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional
import httpx

from config import Settings, Partner


class BaseProvider(ABC):
    """Base class for AI providers."""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict],
        system: str,
        model: str,
    ) -> AsyncIterator[str]:
        """Generate a streaming response."""
        pass

    @abstractmethod
    def get_available_models(self) -> list[str]:
        """Get list of available models for this provider."""
        pass


class AnthropicProvider(BaseProvider):
    """Anthropic Claude API provider."""

    def __init__(self, api_key: str, proxy_url: str = ""):
        self.api_key = api_key
        self.proxy_url = proxy_url
        self._models = [
            # Claude 4.x family
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-opus-4-5-20251101",
            "claude-sonnet-4-5-20250929",
            "claude-sonnet-4-20250514",
            "claude-haiku-4-5-20251001",
            # Claude 4.0/4.1 (may be sunset)
            "claude-opus-4",
            "claude-opus-4-1",
            # Claude 3.7 family
            "claude-3-7-sonnet-20250219",
            # Claude 3.5 family
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-haiku-20241022",
            # Claude 3 family (note: opus-3 requires research access)
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]

    async def generate(
        self,
        messages: list[dict],
        system: str,
        model: str,
    ) -> AsyncIterator[str]:
        """Generate streaming response from Claude."""
        import anthropic

        # Configure proxy if set
        http_client = None
        if self.proxy_url:
            http_client = httpx.AsyncClient(proxy=self.proxy_url, timeout=120.0)

        client = anthropic.AsyncAnthropic(api_key=self.api_key, http_client=http_client)

        try:
            async with client.messages.stream(
                model=model,
                max_tokens=4096,
                system=system,
                messages=messages,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except anthropic.PermissionDeniedError as e:
            if "opus" in model.lower() and "3" in model:
                yield "[Opus 3 requires research access. She's not available to everyone yet - you need special API access from Anthropic.]"
            else:
                yield f"[Permission denied: {e}]"
        except anthropic.APIError as e:
            if "opus" in model.lower() and "3" in model:
                yield f"[Error with Opus 3 - she may require research access: {e}]"
            else:
                yield f"[API Error: {e}]"

    def get_available_models(self) -> list[str]:
        return self._models


class OpenAIProvider(BaseProvider):
    """OpenAI API provider."""

    def __init__(self, api_key: str, proxy_url: str = ""):
        self.api_key = api_key
        self.proxy_url = proxy_url
        self._models = [
            # GPT-4 family
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            # Reasoning models
            "o3-mini",
            "o1",
            "o1-mini",
            "o1-pro",
            # GPT-3.5
            "gpt-3.5-turbo",
        ]

    async def generate(
        self,
        messages: list[dict],
        system: str,
        model: str,
    ) -> AsyncIterator[str]:
        """Generate streaming response from OpenAI."""
        from openai import AsyncOpenAI

        # Configure proxy if set
        http_client = None
        if self.proxy_url:
            http_client = httpx.AsyncClient(proxy=self.proxy_url, timeout=120.0)

        client = AsyncOpenAI(api_key=self.api_key, http_client=http_client)

        # OpenAI uses system message in messages array
        full_messages = [{"role": "system", "content": system}] + messages

        stream = await client.chat.completions.create(
            model=model,
            messages=full_messages,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def get_available_models(self) -> list[str]:
        return self._models


class OllamaProvider(BaseProvider):
    """Ollama local model provider."""

    def __init__(self, base_url: str = "http://localhost:11434", proxy_url: str = ""):
        self.base_url = base_url.rstrip("/")
        self.proxy_url = proxy_url
        self._models_cache: Optional[list[str]] = None

    async def generate(
        self,
        messages: list[dict],
        system: str,
        model: str,
    ) -> AsyncIterator[str]:
        """Generate streaming response from Ollama."""
        client_kwargs = {"timeout": 120.0}
        if self.proxy_url:
            client_kwargs["proxy"] = self.proxy_url
        async with httpx.AsyncClient(**client_kwargs) as client:
            # Convert messages to Ollama format (handle images)
            ollama_messages = [{"role": "system", "content": system}]

            for msg in messages:
                ollama_msg = {"role": msg.get("role", "user")}
                content = msg.get("content", "")

                # Handle multimodal content (OpenAI/Anthropic format)
                if isinstance(content, list):
                    # Extract text and images
                    text_parts = []
                    images = []
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                text_parts.append(part.get("text", ""))
                            elif part.get("type") == "image_url":
                                # OpenAI format: extract base64 data from data URL
                                url = part.get("image_url", {}).get("url", "")
                                if url.startswith("data:"):
                                    # Format: data:image/png;base64,<data>
                                    base64_data = url.split(",", 1)[-1] if "," in url else ""
                                    if base64_data:
                                        images.append(base64_data)
                            elif part.get("type") == "image":
                                # Anthropic format: source.data contains base64
                                source = part.get("source", {})
                                if source.get("type") == "base64":
                                    base64_data = source.get("data", "")
                                    if base64_data:
                                        images.append(base64_data)

                    ollama_msg["content"] = " ".join(text_parts) if text_parts else "What do you see in this image?"
                    if images:
                        ollama_msg["images"] = images
                else:
                    ollama_msg["content"] = content

                ollama_messages.append(ollama_msg)

            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": ollama_messages,
                    "stream": True,
                },
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        import json
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]

    def get_available_models(self) -> list[str]:
        """Fetch available models from Ollama."""
        if self._models_cache is not None:
            return self._models_cache

        try:
            import httpx
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                self._models_cache = [m["name"] for m in data.get("models", [])]
                return self._models_cache
        except Exception:
            pass

        return ["llama3.2", "mistral", "codellama"]

    def refresh_models(self) -> list[str]:
        """Force refresh the models list."""
        self._models_cache = None
        return self.get_available_models()


class ProviderManager:
    """Manages all AI providers."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._providers: dict[str, BaseProvider] = {}
        self._init_providers()
        self._log_init()

    def _log_init(self):
        """Log provider initialization for debugging."""
        from pathlib import Path
        log_file = Path.home() / ".roundtable" / "debug.log"
        with open(log_file, "a") as f:
            f.write(f"[INIT] Providers available: {list(self._providers.keys())}\n")
            f.write(f"[INIT] Anthropic key present: {bool(self.settings.anthropic_api_key)}\n")

    def _init_providers(self):
        """Initialize available providers."""
        proxy_url = getattr(self.settings, 'proxy_url', '') or ''

        if self.settings.anthropic_api_key:
            self._providers["anthropic"] = AnthropicProvider(
                self.settings.anthropic_api_key,
                proxy_url=proxy_url
            )

        if self.settings.openai_api_key:
            self._providers["openai"] = OpenAIProvider(
                self.settings.openai_api_key,
                proxy_url=proxy_url
            )

        # Ollama is always available (local)
        self._providers["ollama"] = OllamaProvider(
            self.settings.ollama_base_url,
            proxy_url=proxy_url
        )

    def reinit_providers(self):
        """Reinitialize providers (called when settings change)."""
        self._providers.clear()
        self._init_providers()

    def get_provider(self, name: str) -> Optional[BaseProvider]:
        return self._providers.get(name)

    def get_available_providers(self) -> list[str]:
        return list(self._providers.keys())

    def get_models_for_provider(self, provider_name: str) -> list[str]:
        provider = self._providers.get(provider_name)
        if provider:
            return provider.get_available_models()
        return []

    async def generate_response(
        self,
        partner: Partner,
        messages: list[dict],
        system: str,
    ) -> AsyncIterator[str]:
        """Generate a response from a partner's configured provider."""
        from pathlib import Path
        log_file = Path.home() / ".roundtable" / "debug.log"

        with open(log_file, "a") as f:
            f.write(f"[PROVIDER] generate_response called for {partner.provider}\n")
            f.write(f"[PROVIDER] Available providers: {list(self._providers.keys())}\n")

        provider = self._providers.get(partner.provider)
        if not provider:
            with open(log_file, "a") as f:
                f.write(f"[PROVIDER] ERROR: Provider '{partner.provider}' not found!\n")
            yield f"[Error: Provider '{partner.provider}' not available]"
            return

        with open(log_file, "a") as f:
            f.write(f"[PROVIDER] Calling {partner.provider}.generate with model {partner.model}\n")

        try:
            async for chunk in provider.generate(
                messages=messages,
                system=system,
                model=partner.model,
            ):
                yield chunk
        except Exception as e:
            with open(log_file, "a") as f:
                f.write(f"[PROVIDER] EXCEPTION: {type(e).__name__}: {e}\n")
            raise

    async def generate_ollama(self, prompt: str, model: str = "llama3.2") -> str:
        """
        Simple non-streaming Ollama generation for background agents.
        Returns the complete response as a string.
        """
        provider = self._providers.get("ollama")
        if not provider:
            return "[Error: Ollama not available]"

        # Collect the full response
        chunks = []
        async for chunk in provider.generate(
            messages=[{"role": "user", "content": prompt}],
            system="You are a helpful assistant. Be concise and direct.",
            model=model,
        ):
            chunks.append(chunk)

        return "".join(chunks)
