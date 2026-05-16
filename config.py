"""
Central configuration using pydantic-settings.
All values come from environment variables (or .env file).
Import `settings` anywhere in the app — it's a singleton.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Every field here maps to an environment variable with the same name
    (case-insensitive). Defaults are safe values for local development.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",          # Silently ignore extra env vars
        case_sensitive=False,
    )

    # ── LLM ───────────────────────────────────────────────────────────────
    llm_provider: str = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1500

    groq_api_key: str = ""
    anthropic_api_key: str = ""

    # ── Embeddings ────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── Catalog ───────────────────────────────────────────────────────────
    catalog_json_url: str = (
        "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
    )
    catalog_fallback_url: str = (
        "https://www.shl.com/solutions/products/product-catalog/"
    )

    # ── Storage paths ─────────────────────────────────────────────────────
    faiss_index_path: str = "data/catalog.faiss"
    catalog_metadata_path: str = "data/catalog_metadata.json"

    # ── Retrieval ─────────────────────────────────────────────────────────
    retrieval_top_k: int = 20

    # ── App ───────────────────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_log_level: str = "info"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns the singleton Settings instance.
    Cached so the .env file is only read once, at first call.
    """
    return Settings()


# Module-level alias so you can do: from app.config import settings
settings = get_settings()