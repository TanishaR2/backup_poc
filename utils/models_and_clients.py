from __future__ import annotations

import base64
import os
from typing import Any, Optional

from google import genai
from google.genai import types
from groq import Groq
from openai import AzureOpenAI, OpenAI
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from utils.logger_config import logger
from utils.settings import (
    AZURE_OPENAI_API_KEY_MINI,
    AZURE_OPENAI_API_KEY2,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_MODEL_NAME,
    AZURE_OPENAI_MODEL_API_KEY_5_4,
    AZURE_OPENAI_MODEL_NAME_5_4,
    AZURE_OPENAI_MODEL_VERSION_5_4,
    EMBEDDING_MODEL_NAME,
    GOOGLE_API_KEY,
    GROQ_API_KEY,
    MODEL_FALLBACK_CHAIN,
    OPENAI_API_KEY,
    QDRANT_API_KEY,
    QDRANT_ENDPOINT,
)

# --- BACKWARDS COMPATIBLE SINGLETON CLIENTS ---
try:
    openai_client_mini = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY_MINI,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )
except Exception:
    openai_client = None

try:
    openai_client = AzureOpenAI(
        api_key=AZURE_OPENAI_MODEL_API_KEY_5_4 or AZURE_OPENAI_API_KEY_MINI,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )
except Exception:
    openai_client = None

try:
    google_client = genai.Client(api_key=GOOGLE_API_KEY)
except Exception:
    google_client = None

try:
    groq_client = Groq(api_key=GROQ_API_KEY)
except Exception:
    groq_client = None

# Embedding & Reranking Model (BGE-M3 Native Multi-Vector)
embedding_model = BGEM3FlagModel(EMBEDDING_MODEL_NAME, use_fp16=True)

# Vector DB Client
qdrant_client = QdrantClient(api_key=QDRANT_API_KEY, url=QDRANT_ENDPOINT, timeout=60.0)


# --- DYNAMIC CLIENT CACHE ---
_client_cache: dict[str, Any] = {}


def _get_provider_client(provider: str, key_env: str) -> Any | None:
    """Get or create client dynamically based on environment variable key."""
    api_key = os.getenv(key_env)
    if not api_key:
        return None

    cache_key = f"{provider}:{key_env}:{api_key[:8]}"
    if cache_key in _client_cache:
        return _client_cache[cache_key]

    try:
        if provider == "groq":
            client = Groq(api_key=api_key)
        elif provider == "google":
            client = genai.Client(api_key=api_key)
        elif provider == "azure_openai":
            client = AzureOpenAI(
                api_key=api_key,
                api_version=AZURE_OPENAI_API_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
            )
        elif provider == "openai":
            client = OpenAI(api_key=api_key)
        else:
            client = None

        if client:
            _client_cache[cache_key] = client
        return client
    except Exception as exc:
        logger.warning(f"[LLM Failover] Failed to instantiate {provider} client with {key_env}: {exc}")
        return None


def run_llm_completion(
    prompt: str,
    image_base64: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> str | None:
    """Execute LLM text/vision completion through configured MODEL_FALLBACK_CHAIN.

    Sequence is strictly controlled by MODEL_FALLBACK_CHAIN in utils/settings.py.
    When a key or model hits rate limits or errors, it logs a warning and
    automatically cascades to the next entry in the chain.
    """
    for index, entry in enumerate(MODEL_FALLBACK_CHAIN, start=1):
        name = entry.get("name", f"step_{index}")
        provider = entry.get("provider")
        key_env = entry.get("key_env")
        model = entry.get("model")
        supports_vision = entry.get("supports_vision", False)

        # Skip text-only models when image is attached
        if image_base64 and not supports_vision:
            logger.debug(f"[LLM Failover] Skipping {name} ({model}): does not support vision input")
            continue

        api_key = os.getenv(key_env)
        if not api_key:
            logger.debug(f"[LLM Failover] Skipping {name}: Environment variable {key_env} not set")
            continue

        client = _get_provider_client(provider, key_env)
        if client is None:
            continue

        try:
            if index == 1:
                logger.info(f"[LLM Service] Starting completion via primary model {name} ({provider}:{model})")
            else:
                logger.info(f"[LLM Failover] Primary model failed. Initiating fallback attempt {index - 1} via backup provider {name} ({provider}:{model})")
            response_text = None

            if provider == "groq":
                params = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                }
                if max_tokens:
                    params["max_tokens"] = max_tokens
                resp = client.chat.completions.create(**params)
                response_text = resp.choices[0].message.content

            elif provider == "google":
                if image_base64:
                    img_bytes = base64.b64decode(image_base64)
                    contents = [
                        prompt,
                        types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                    ]
                else:
                    contents = prompt

                resp = client.models.generate_content(model=model, contents=contents)
                response_text = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else None)

            elif provider in {"azure_openai", "openai"}:
                if image_base64:
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                        ],
                    }]
                else:
                    messages = [{"role": "user", "content": prompt}]

                params = {"model": model, "messages": messages, "timeout": 30.0}
                if max_tokens:
                    params["max_tokens"] = max_tokens
                if temperature is not None:
                    params["temperature"] = temperature

                resp = client.chat.completions.create(**params)
                response_text = resp.choices[0].message.content

            if response_text and response_text.strip():
                logger.success(f"[LLM Failover] Success via {name} ({provider}:{model})")
                return response_text.strip()

        except Exception as exc:
            logger.warning(f"[LLM Failover] Step {index} failed for {name} ({provider}:{model}): {exc}")

    logger.error("[LLM Failover] All models/keys in MODEL_FALLBACK_CHAIN failed!")
    return None