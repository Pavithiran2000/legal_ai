"""
Configuration for Ollama-based Model Server.

This is a STANDALONE server — no .env file required.
All defaults are hardcoded. Override via environment variables if needed.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
import os


class ModelServerSettings(BaseSettings):
    """Model server configuration — fully standalone, no .env needed."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=5006)

    # Ollama settings
    ollama_host: str = Field(default="http://localhost:11434")
    default_model: str = Field(default="sri-legal-8b")
    available_models: list = Field(default=["sri-legal-8b", "sri-legal-4b"])

    # GGUF paths (relative to model-server directory)
    model_4b_path: str = Field(default="./models/qwen3-4b.gguf")
    model_8b_path: str = Field(default="./models/qwen3_8b.gguf")

    # Generation defaults
    temperature: float = Field(default=0.1)
    top_p: float = Field(default=0.95)
    num_ctx: int = Field(default=8192)
    max_tokens: int = Field(default=6000)

    # Logging
    log_level: str = Field(default="INFO")

    class Config:
        # No .env file — this server is fully standalone
        extra = "ignore"
        protected_namespaces = ()

    @property
    def model_4b_exists(self) -> bool:
        return os.path.exists(self.model_4b_path)

    @property
    def model_8b_exists(self) -> bool:
        return os.path.exists(self.model_8b_path)


settings = ModelServerSettings()
