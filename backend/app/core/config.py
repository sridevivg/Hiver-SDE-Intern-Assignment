"""
SupportGraph AI — Core Configuration

Environment-based settings using Pydantic Settings.
All configuration values are read from environment variables or .env file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Resolve project root (two levels up from this file: backend/app/core/ → root)
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[3]  # SupportGraph-AI/


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Priority order (highest → lowest):
      1. Actual environment variables
      2. backend/.env file
      3. Root .env file
      4. Default values defined here
    """

    model_config = SettingsConfigDict(
        # Look for .env in backend/ first, then project root
        env_file=(
            str(PROJECT_ROOT / "backend" / ".env"),
            str(PROJECT_ROOT / ".env"),
        ),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -----------------------------------------------------------------------
    # Application metadata
    # -----------------------------------------------------------------------
    app_name: str = "SupportGraph AI"
    app_version: str = "0.1.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = True

    # -----------------------------------------------------------------------
    # Server
    # -----------------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000

    # -----------------------------------------------------------------------
    # Data paths — relative to project root
    # -----------------------------------------------------------------------
    data_raw_dir: str = "Dataset/Raw/twcs"
    data_interim_dir: str = "data/interim"
    data_processed_dir: str = "data/processed"
    data_golden_dir: str = "data/golden"
    artifacts_dir: str = "artifacts"

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["text", "json"] = "text"

    # -----------------------------------------------------------------------
    # LLM Configuration (Groq & Ollama Factory)
    # -----------------------------------------------------------------------
    llm_provider: Literal["groq", "ollama"] = "groq"
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:latest"

    # -----------------------------------------------------------------------
    # Computed properties
    # -----------------------------------------------------------------------

    @property
    def raw_data_path(self) -> Path:
        """Absolute path to the raw data directory."""
        p = Path(self.data_raw_dir)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p

    @property
    def interim_data_path(self) -> Path:
        """Absolute path to the interim data directory."""
        p = Path(self.data_interim_dir)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p

    @property
    def processed_data_path(self) -> Path:
        """Absolute path to the processed data directory."""
        p = Path(self.data_processed_dir)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p

    @property
    def artifacts_path(self) -> Path:
        """Absolute path to the artifacts directory."""
        p = Path(self.artifacts_dir)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p

    @property
    def figures_path(self) -> Path:
        """Absolute path to the figures directory."""
        return self.artifacts_path / "figures"

    @property
    def reports_path(self) -> Path:
        """Absolute path to the reports directory."""
        return self.artifacts_path / "reports"


# ---------------------------------------------------------------------------
# Global singleton — import this throughout the app
# ---------------------------------------------------------------------------
settings = Settings()
