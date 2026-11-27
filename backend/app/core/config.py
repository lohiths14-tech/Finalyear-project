from pathlib import Path
import secrets

from pydantic_settings import BaseSettings, SettingsConfigDict
import torch


class Settings(BaseSettings):
    PROJECT_NAME: str = "AgriVision AI"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"
    DATABASE_URL: str | None = None

    # CORS — comma-separated in env:
    BACKEND_CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "https://finalyear-project.vercel.app",
        "https://finalyear-project-bcxuut7q0-lohiths-projects-b447afea.vercel.app",
    ]

    # API Key authentication (leave empty to disable)
    API_KEY: str | None = None

    # ── External Public APIs ────────────────────────────────────────────────
    # GNews — Free Indian agriculture news (get key at https://gnews.io)
    GNEWS_API_KEY: str | None = None
    # Open-Meteo base URL (free, no key required)
    OPEN_METEO_BASE_URL: str = "https://api.open-meteo.com/v1/forecast"

    # ── Grounded Agronomic LLM APIs (Optional) ──────────────────────────────
    GEMINI_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None

    # File upload limits (20MB maximum)
    MAX_FILE_SIZE_MB: int = 20
    MAX_FILE_SIZE_BYTES: int = 20 * 1024 * 1024

    # ML device override: auto-detect CUDA > MPS > CPU
    DEVICE: str = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")

    # Security — MUST be set via SECRET_KEY env variable in production.
    # If not set, a random 64-char hex token is generated at startup (safe for
    # development only; restarts will invalidate sessions).
    SECRET_KEY: str = secrets.token_hex(32)

    # Model weights directory
    MODEL_BASE_PATH: str = "./model_weights"

    # Logging
    LOG_DIR: str = "./logs"
    LOG_LEVEL: str = "INFO"

    thresholds: dict[str, dict[str, float]] = {
        "quality": {
            "blur_min": 0.4,
            "brightness_min": 0.3,
            "brightness_max": 0.9,
            "noise_min": 0.4,
            "contrast_min": 0.3,
            "resolution_min": 0.5,
            "shadow_max": 0.5,
            "vegetation_min": 0.15
        }
    }

    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=(
            str(Path(__file__).resolve().parent.parent.parent / ".env"),
            ".env",
            "backend/.env",
        ),
        extra="allow",
    )

    def resolve_model_base_dir(self) -> Path:
        """Dynamically search common locations for model_weights directory."""
        candidates = [
            Path(self.MODEL_BASE_PATH),
            Path(__file__).resolve().parent.parent.parent / "model_weights",
            Path(__file__).resolve().parent.parent.parent.parent / "model_weights",
            Path("../model_weights"),
            Path("./backend/model_weights"),
        ]
        for c in candidates:
            if c.exists() and (c / "ensemble.pt").exists():
                return c.resolve()
        for c in candidates:
            if c.exists():
                return c.resolve()
        return Path(self.MODEL_BASE_PATH).resolve()

    def get_model_paths(self) -> dict[str, str] | None:
        """Return resolved model weight paths if .pt files exist, else None."""
        base = self.resolve_model_base_dir()
        paths = {
            "spectral": base / "spectral_net.pt",
            "spatial": base / "spatial_net.pt",
            "temporal": base / "temporal_net.pt",
            "ensemble": base / "ensemble.pt",
            "mobilenet": base / "mobilenet_v2_paddy.pt",
        }
        found = {k: str(v) for k, v in paths.items() if v.exists()}
        return found if found else None


settings = Settings()
