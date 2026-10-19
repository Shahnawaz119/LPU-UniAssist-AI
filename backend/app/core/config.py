import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "meta-llama/llama-3.3-70b-instruct"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_SITE_URL: str = "http://localhost:5173"
    OPENROUTER_APP_NAME: str = "University Helpdesk AI"
    TOP_K: int = 4
    RELEVANCE_THRESHOLD: float = 0.35
    CHROMA_PERSIST_DIR: str = str(BASE_DIR / "data" / "chroma_db")
    KNOWLEDGE_BASE_DIR: str = str(BASE_DIR / "data" / "knowledge_base")

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
