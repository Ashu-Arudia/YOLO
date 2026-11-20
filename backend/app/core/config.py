"""
Backend Configuration
Uses pydantic-settings to load from .env
"""
from __future__ import annotations
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_ENV: str   = "development"
    APP_HOST: str  = "0.0.0.0"
    APP_PORT: int  = 8000
    DEBUG: bool    = True

    DATABASE_URL: str = "sqlite+aiosqlite:///./sportsvision.db"

    STORAGE_DIR: str   = "./storage"
    VIDEOS_DIR: str    = "./storage/videos"
    PROCESSED_DIR: str = "./storage/processed"
    FRAMES_DIR: str    = "./storage/frames"

    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:3001"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",")]

    @property
    def videos_path(self) -> Path:
        return Path(self.VIDEOS_DIR)

    @property
    def processed_path(self) -> Path:
        return Path(self.PROCESSED_DIR)


settings = Settings()
