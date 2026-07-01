"""AI Provider clients for Roundtable."""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional
import httpx

from config import Settings, Partner
import debug_logger as dbg
import cache_monitor


def _normalize_temperature(value: float | int | str | None) -> float:
    """Clamp partner temperature to the supported 0.0-1.0 range."""
    try:
        temperature = float(value)
    except (TypeError, ValueError):
        temperature = 0.7
    return max(0.0, min(1.0, round(temperature, 1)))


def _anthropic_model_supports_temperature(model: str) -> bool:
    """Claude Opus 4.7+ rejects sampling parameters; omit them entirely."""
    parts = (model or "").lower().split("-")
    if len(parts) < 4 or parts[:3] != ["claude", "opus", "4"]:
        return True

    try:
        minor = int(parts[3])
    except ValueError:
        return True

    return minor < 7


def _ensure_user_final_message(messages: list[dict]) -> list[dict]:
    """Anthropic treats assistant-final history as prefill; add a real user turn."""
    if messages and messages[-1].get("role") == "user":
        return messages

    fixed_messages = list(messages)
    fixed_messages.append({
        "role": "user",
        "content": "Continue naturally from the conversation so far.",
    })
    return fixed_messages


def _openai_model_supports_temperature(model: str) -> bool:
    """OpenAI reasoning models may reject custom temperature."""
    return not (model or "").lower().startswith(("o1", "o3"))


class BaseProvider(ABC):
    """Base class for AI providers."""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict],
        system: str,
        model: str,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Generate a streaming response."""
        pass

    @abstractmethod
    def get_available_models(self) -> list[str]:
        """Get list of available models for this provider."""
        pass


class AnthropicProvider(BaseProvider):
    """Anthropic Claude API provider."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._models = [
            # Claude 4.x family
            "claude-opus-4-8",
            "claude-opus-4-7",
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-opus-4-5-20251101",
            "claude-sonnet-4-5-20250929",
            "claude-sonnet-4-20250514",
            "claude-haiku-4-5-20251001",
            # Claude 4.0/4.1 (may be sunset)
            "claude-opus-4-0",
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
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Generate streaming response from Claude with optimized prompt caching."""
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self.api_key)

        # Dynamic TTL based on conversation length
        # Short sessions (1-2 turns): 5min TTL (cheaper 1.25x write, breaks even at 2 requests)
        # Long sessions (3+ turns): 1hr TTL (committed session, worth the 2x write premium)
        messages = _ensure_user_final_message(messages)
        conversation_turns = len(messages)
        ttl = "1h" if conversation_turns >= 3 else "5m"

        dbg.provider(f"Cache TTL: {ttl} (conversation has {conversation_turns} turns)")

        # Cache system prompt with dynamic TTL
        # This caches the entire system context (character, memory, scenario, etc)
        system_with_cache = [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral", "ttl": ttl}
            }
        ]

        # OPTIMIZED: Cache the LAST message for multi-turn conversations
        # This follows best practices - cache grows incrementally with each turn
        cached_messages = []
        for i, msg in enumerate(messages):
            is_last = (i == len(messages) - 1)

            # Last message gets cache control
            if is_last and len(messages) >= 2:  # Only cache if we have a conversation going
                content = msg["content"]

                # Convert to array format if needed
                if isinstance(content, str):
                    cached_messages.append({
                        "role": msg["role"],
                        "content": [
                            {
                                "type": "text",
                                "text": content,
                                "cache_control": {"type": "ephemeral"}
                            }
                        ]
                    })
                else:
                    # Already array format, add cache to last block
                    content_copy = content.copy()
                    if content_copy:
                        content_copy[-1] = {
                            **content_copy[-1],
                            "cache_control": {"type": "ephemeral"}
                        }
                    cached_messages.append({
                        "role": msg["role"],
                        "content": content_copy
                    })
            else:
                # Keep original format for non-cached messages
                cached_messages.append(msg)

        request_args = {
            "model": model,
            "max_tokens": 4096,
            "system": system_with_cache,
            "messages": cached_messages,
        }
        if _anthropic_model_supports_temperature(model):
            request_args["temperature"] = temperature

        try:
            async with client.messages.stream(**request_args) as stream:
                async for text in stream.text_stream:
                    yield text

                # Get final message to log cache stats
                final_message = await stream.get_final_message()
                usage = final_message.usage

                # Log cache performance
                cache_read = usage.cache_read_input_tokens
                cache_write = usage.cache_creation_input_tokens
                uncached = usage.input_tokens
                output_tokens = usage.output_tokens
                total = cache_read + cache_write + uncached

                if total > 0:
                    # Record in cache monitor for long-term tracking
                    monitor = cache_monitor.get_monitor()
                    stats = monitor.record_request(
                        model=model,
                        cache_read_tokens=cache_read,
                        cache_write_tokens=cache_write,
                        uncached_tokens=uncached,
                        output_tokens=output_tokens
                    )

                    dbg.provider(
                        f"💰 Cache: {cache_read:,} read / {cache_write:,} write / {uncached:,} uncached | "
                        f"Hit rate: {stats['cache_hit_rate']:.1f}% | "
                        f"Cost: ${stats['cost']:.4f} (saved ${stats['savings']:.4f})"
                    )

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

    def __init__(self, api_key: str):
        self.api_key = api_key
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
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Generate streaming response from OpenAI."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.api_key)

        # OpenAI uses system message in messages array
        full_messages = [{"role": "system", "content": system}] + messages

        request_args = {
            "model": model,
            "messages": full_messages,
            "stream": True,
        }
        if _openai_model_supports_temperature(model):
            request_args["temperature"] = temperature

        stream = await client.chat.completions.create(**request_args)

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def get_available_models(self) -> list[str]:
        return self._models


class OllamaProvider(BaseProvider):
    """Ollama local model provider."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
        self._models_cache: Optional[list[str]] = None

    async def generate(
        self,
        messages: list[dict],
        system: str,
        model: str,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Generate streaming response from Ollama."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Ollama chat format
            full_messages = [{"role": "system", "content": system}] + messages

            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json={
                        "model": model,
                        "messages": full_messages,
                        "stream": True,
                        "options": {
                            "temperature": temperature,
                        },
                    },
                ) as response:
                    # Check for HTTP errors
                    if response.status_code != 200:
                        error_text = await response.aread()
                        raise Exception(f"Ollama HTTP {response.status_code}: {error_text.decode()[:200]}")

                    got_content = False
                    async for line in response.aiter_lines():
                        if line:
                            import json
                            data = json.loads(line)
                            # Check for Ollama error response
                            if "error" in data:
                                raise Exception(f"Ollama error: {data['error']}")
                            if "message" in data and "content" in data["message"]:
                                content = data["message"]["content"]
                                if content:
                                    got_content = True
                                    yield content

                    # If we got no content at all, that's an error
                    if not got_content:
                        raise Exception("Ollama returned empty response - model may not be loaded")
            except httpx.ConnectError:
                raise Exception(f"Cannot connect to Ollama at {self.base_url} - is it running?")
            except httpx.ReadTimeout:
                raise Exception("Ollama request timed out - model may be loading or overloaded")

    def get_available_models(self) -> list[str]:
        """Fetch available models from Ollama."""
        if self._models_cache is not None:
            return self._models_cache

        try:
            import httpx
            print(f"[ollama] Fetching models from {self.base_url}/api/tags...")
            response = httpx.get(f"{self.base_url}/api/tags", timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                self._models_cache = [m["name"] for m in data.get("models", [])]
                print(f"[ollama] Found {len(self._models_cache)} models: {self._models_cache[:5]}{'...' if len(self._models_cache) > 5 else ''}")
                return self._models_cache
            else:
                print(f"[ollama] Failed to fetch models: HTTP {response.status_code}")
        except Exception as e:
            print(f"[ollama] Failed to fetch models: {e}")

        print("[ollama] Using fallback model list")
        return ["deepseek-v3.1:671b-cloud", "llama3.2", "mistral", "qwen2.5:7b"]

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
        if self.settings.anthropic_api_key:
            self._providers["anthropic"] = AnthropicProvider(
                self.settings.anthropic_api_key
            )

        if self.settings.openai_api_key:
            self._providers["openai"] = OpenAIProvider(
                self.settings.openai_api_key
            )

        # Ollama is always available (local)
        self._providers["ollama"] = OllamaProvider(
            self.settings.ollama_base_url
        )

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
        temperature = _normalize_temperature(getattr(partner, "temperature", 0.7))
        dbg.provider(f"▶ generate_response: {partner.name} via {partner.provider}/{partner.model} temp={temperature:.1f}")

        provider = self._providers.get(partner.provider)
        if not provider:
            dbg.provider(f"✗ Provider '{partner.provider}' not found!")
            yield f"[Error: Provider '{partner.provider}' not available]"
            return

        try:
            chunk_count = 0
            async for chunk in provider.generate(
                messages=messages,
                system=system,
                model=partner.model,
                temperature=temperature,
            ):
                chunk_count += 1
                yield chunk
            dbg.provider(f"✓ Response complete: {chunk_count} chunks")
        except Exception as e:
            dbg.provider(f"✗ EXCEPTION: {type(e).__name__}: {e}")
            raise

    async def generate_ollama(self, prompt: str, model: str = None) -> str:
        """
        Simple non-streaming Ollama generation for background agents.
        Returns the complete response as a string.
        Uses settings.default_ollama_model if no model specified.
        """
        provider = self._providers.get("ollama")
        if not provider:
            dbg.provider("✗ Ollama not available")
            return "[Error: Ollama not available]"

        # Use configured default if no model specified
        if model is None:
            model = self.settings.default_ollama_model

        prompt_preview = prompt[:80] + "..." if len(prompt) > 80 else prompt
        dbg.provider(f"▶ Ollama generate: {model} | {prompt_preview}")

        # Collect the full response
        chunks = []
        async for chunk in provider.generate(
            messages=[{"role": "user", "content": prompt}],
            system="You are a helpful assistant. Be concise and direct.",
            model=model,
        ):
            chunks.append(chunk)

        result = "".join(chunks)
        dbg.provider(f"✓ Ollama response: {len(result)} chars")
        return result
