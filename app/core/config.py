import os
import logging
from pydantic_settings import BaseSettings
from pydantic import model_validator, ConfigDict

logger = logging.getLogger(__name__)

INSECURE_DEFAULT_SECRETS = {
    "super_secret_key_in_production_from_env",
    "replace_with_a_secure_random_secret_at_least_32_characters",
    "secret",
    "changeme",
    "your_session_secret_key_here",
    "your_session_secret_key_minimum_32_characters",
}

DEV_FALLBACK_SECRET = "dev_insecure_session_secret_key_only_for_local_development_min32chars"

class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    session_secret_key: str | None = None
    session_max_age: int = 86400  # 24 hours
    session_cookie_same_site: str = "lax"
    session_cookie_secure: bool | None = None  # None -> auto-detected based on app_env

    database_url: str = "sqlite:///./adaptive_rag.db"
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 5
    dense_weight: float = 0.65
    bm25_weight: float = 0.35
    enable_reranker: bool = False

    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    gemini_api_key: str | None = None
    groq_api_key: str | None = None

    evaluation_threshold: float = 0.65

    @model_validator(mode="after")
    def validate_session_secret(self):
        env = (self.app_env or "development").strip().lower()
        key = (self.session_secret_key or "").strip()
        is_production = env == "production" or env not in ("development", "test", "testing")

        if is_production:
            if not key:
                raise ValueError(
                    "CRITICAL CONFIGURATION ERROR: SESSION_SECRET_KEY is required in production. "
                    "Set SESSION_SECRET_KEY in your environment to a random string of at least 32 characters."
                )
            if key in INSECURE_DEFAULT_SECRETS:
                raise ValueError(
                    "CRITICAL CONFIGURATION ERROR: The configured SESSION_SECRET_KEY is a known insecure default "
                    "or placeholder. Generate a unique, cryptographically random key of at least 32 characters."
                )
            if len(key) < 32:
                raise ValueError(
                    "CRITICAL CONFIGURATION ERROR: SESSION_SECRET_KEY in production must be at least 32 characters long."
                )
        else:
            if not key or key in INSECURE_DEFAULT_SECRETS:
                # Use explicit dev fallback in local development/testing
                self.session_secret_key = DEV_FALLBACK_SECRET

        return self

    @property
    def is_cookie_secure(self) -> bool:
        if self.session_cookie_secure is not None:
            return self.session_cookie_secure
        return (self.app_env or "development").strip().lower() == "production"

settings = Settings()
