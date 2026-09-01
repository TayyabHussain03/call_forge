"""Application settings loaded from environment variables.

Ye module pure configuration ka single source of truth hai. Koi bhi module
hardcoded value use nahi karega — sab kuch yahan se `get_settings()` ke through
aata hai. Values `.env` file (ya real environment) se padhi jaati hain aur
Pydantic dwara type-check + validate hoti hain, taake galat config app start
hote hi pakdi jaaye, runtime par nahi.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application configuration.

    Har field ek environment variable se map hoti hai (case-insensitive).
    Defaults local development ke liye sane rakhe gaye hain; production values
    `.env` ya deployment environment se override hoti hain.

    Attributes:
        app_env: Deployment environment name (local/staging/production).
        app_debug: Debug mode flag; production mein False hona chahiye.
        log_level: Root logging level.
        database_url: SQLAlchemy async connection string. Default SQLite;
            Postgres par switch sirf is URL ko badalne se hota hai.
        db_echo: Agar True to SQLAlchemy saari SQL queries log karega.
        llm_provider: Kaunsa LLM provider use karna hai (factory isse padhta hai).
        gemini_api_key: Gemini API key (secret).
        llm_model: Model identifier string.
        llm_timeout_seconds: Ek LLM call ke liye max wait time.
        vapi_api_key: Vapi platform key (voice; abhi optional).
        vapi_phone_number_id: Vapi outbound number id (abhi optional).
        redis_url: Redis connection string (queue/workers ke liye).
        conversation_config_path: State-machine YAML config ka path.
        prompts_dir: LLM prompt files ka directory.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──
    app_env: str = "local"
    app_debug: bool = True
    log_level: str = "INFO"

    # ── Database ──
    database_url: str = "sqlite+aiosqlite:///./data/app.db"
    db_echo: bool = False

    # ── LLM ──
    llm_provider: str = "gemini"
    gemini_api_key: str = Field(default="", description="Gemini API key")
    llm_model: str = "gemini-flash-latest"
    llm_timeout_seconds: int = 15

    # ── Voice (optional for now) ──
    vapi_api_key: str = ""
    vapi_phone_number_id: str = ""

    # ── Queue ──
    redis_url: str = "redis://localhost:6379/0"

    # ── Config file locations ──
    conversation_config_path: str = "app/config/conversation_config.yaml"
    prompts_dir: str = "app/config/prompts"

    @property
    def is_production(self) -> bool:
        """Return True agar app production environment mein chal rahi hai.

        Returns:
            bool: True jab app_env 'production' ho.
        """
        return self.app_env.lower() == "production"

    @property
    def prompts_path(self) -> Path:
        """Prompts directory ko Path object ke roop mein deta hai.

        Returns:
            Path: Resolved prompts directory path.
        """
        return Path(self.prompts_dir)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    `lru_cache` ki wajah se `.env` sirf ek baar padha jaata hai aur wahi
    instance poori app mein reuse hota hai. FastAPI dependency injection aur
    kisi bhi module se isi function ko call karna chahiye — direct
    `Settings()` construct karne ke bajaye.

    Returns:
        Settings: The singleton, validated settings object.
    """
    return Settings()