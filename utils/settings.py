from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

# --- API KEYS ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY2 = os.getenv("GOOGLE_API_KEY2")
GOOGLE_API_KEY3 = os.getenv("GOOGLE_API_KEY3")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_KEY2 = os.getenv("GROQ_API_KEY_2") or os.getenv("GROQ_API_KEY2")

QDRANT_ENDPOINT = os.getenv("QDRANT_ENDPOINT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

AZURE_OPENAI_API_KEY_MINI = os.getenv("AZURE_OPENAI_API_KEY_MINI")
AZURE_OPENAI_API_KEY2 = os.getenv("AZURE_OPENAI_API_KEY2")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_REGION = os.getenv("AZURE_OPENAI_REGION")
AZURE_OPENAI_MODEL_VERSION = os.getenv("AZURE_OPENAI_MODEL_VERSION")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_MODEL_NAME = os.getenv("AZURE_OPENAI_MODEL_NAME") or "gpt-5-mini"
AZURE_OPENAI_MODEL_ENDPOINT = os.getenv("AZURE_OPENAI_MODEL_ENDPOINT")
AZURE_OPENAI_MODEL_API_KEY = os.getenv("AZURE_OPENAI_MODEL_API_KEY")


AZURE_OPENAI_MODEL_VERSION_5_4 = os.getenv("AZURE_OPENAI_MODEL_VERSION_5_4") 
AZURE_OPENAI_MODEL_NAME_5_4 = os.getenv("AZURE_OPENAI_MODEL_NAME_5_4") or "gpt-5.4"
AZURE_OPENAI_MODEL_API_KEY_5_4 = (
    os.getenv("AZURE_OPENAI_MODEL_API_KEY_5_4")
    or os.getenv("AZURE_OPENAI_API_KEY_5_4")
)
AZURE_OPENAI_5_4_KEY_ENV = (
    "AZURE_OPENAI_MODEL_API_KEY_5_4"
    if os.getenv("AZURE_OPENAI_MODEL_API_KEY_5_4")
    else "AZURE_OPENAI_API_KEY_5_4"
)

# --- GENERAL CONFIG ---
COLLECTION_NAME = "InsightDocs"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_NEEDED = True
vlm_provider = "azure_openai"

# Minimal routing and confidence thresholds
RETRIEVAL_CONFIDENCE_THRESHOLD = 0.50
RETRIEVAL_SKIP_VALIDATION_THRESHOLD = 0.70
VALIDATION_CONFIDENCE_THRESHOLD = 0.70

# --- MODEL FALLBACK DICTIONARY CHAIN ---
# Simply reorder these entries to manually control the fallback sequence!
MODEL_FALLBACK_CHAIN = [
    # 1. Azure OpenAI Key 1 (Primary for fast latency & high accuracy)
    {"name": "azure_openai_key_5_4", "provider": "azure_openai", "key_env": AZURE_OPENAI_5_4_KEY_ENV, "model": AZURE_OPENAI_MODEL_NAME_5_4, "supports_vision": True},
    # 2. Azure OpenAI Key 2
    {"name": "azure_openai_key_5_4_2", "provider": "azure_openai", "key_env": AZURE_OPENAI_5_4_KEY_ENV, "model": AZURE_OPENAI_MODEL_NAME_5_4, "supports_vision": True},

    # 3. OpenAI Primary
    {"name": "openai_key1", "provider": "openai", "key_env": "OPENAI_API_KEY", "model": "gpt-4o-mini", "supports_vision": True},

    # 4. Groq Client 1 - Model 1 (Llama 3.3 70B)
    {"name": "groq_key1_70b", "provider": "groq", "key_env": "GROQ_API_KEY", "model": "llama-3.3-70b-versatile", "supports_vision": False},
    # 5. Groq Client 1 - Model 2 (Llama 3.1 8B)
    {"name": "groq_key1_8b", "provider": "groq", "key_env": "GROQ_API_KEY", "model": "llama-3.1-8b-instant", "supports_vision": False},

    # 6. Groq Client 2 - Model 1 (Llama 3.3 70B)
    {"name": "groq_key2_70b", "provider": "groq", "key_env": "GROQ_API_KEY_2", "model": "llama-3.3-70b-versatile", "supports_vision": False},
    # 7. Groq Client 2 - Model 2 (Llama 3.1 8B)
    {"name": "groq_key2_8b", "provider": "groq", "key_env": "GROQ_API_KEY_2", "model": "llama-3.1-8b-instant", "supports_vision": False},

    # 8. Gemini Key 1
    {"name": "gemini_key1", "provider": "google", "key_env": "GOOGLE_API_KEY", "model": "gemini-2.5-flash", "supports_vision": True},
    # 9. Gemini Key 2
    {"name": "gemini_key2", "provider": "google", "key_env": "GOOGLE_API_KEY2", "model": "gemini-2.5-flash", "supports_vision": True},
    # 10. Gemini Key 3
    {"name": "gemini_key3", "provider": "google", "key_env": "GOOGLE_API_KEY3", "model": "gemini-2.5-flash", "supports_vision": True},
]