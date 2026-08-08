"""Environment configuration.

Loaded lazily so that importing the app (or starting the server for /health)
never crashes when API keys are not configured yet. Endpoints that need keys
raise ConfigError with an actionable message instead.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


class ConfigError(RuntimeError):
    """Raised when a required environment variable is missing."""


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value or value.startswith("your-"):
        raise ConfigError(
            f"Missing required environment variable {name}. "
            "Copy .env.example to .env and fill in your keys (see README)."
        )
    return value


@dataclass(frozen=True)
class Settings:
    google_api_key: str
    pinecone_api_key: str
    index_name: str
    namespace: str
    cloud: str
    region: str
    chat_model: str
    embed_model: str
    embed_dim: int = 768
    top_k: int = 4
    max_retrieval_loops: int = 2  # max query rewrites before giving up


def get_settings() -> Settings:
    return Settings(
        google_api_key=_require("GOOGLE_API_KEY"),
        pinecone_api_key=_require("PINECONE_API_KEY"),
        index_name=os.getenv("PINECONE_INDEX_NAME", "legixo-takehome"),
        namespace=os.getenv("PINECONE_NAMESPACE", "corpus"),
        cloud=os.getenv("PINECONE_CLOUD", "aws"),
        region=os.getenv("PINECONE_REGION", "us-east-1"),
        chat_model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash"),
        embed_model=os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001"),
    )
