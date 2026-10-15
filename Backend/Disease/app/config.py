import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Settings:
    """Application configuration loaded from environment variables."""

    PROJECT_NAME: str = "TreeTrack AI - Disease Detection Module"
    VERSION: str = "0.1.0"
    API_V1_PREFIX: str = "/api/v1"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

    # Allowed CORS Origins (comma-separated in env)
    ALLOWED_ORIGINS: List[str] = field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.getenv(
                "ALLOWED_ORIGINS",
                "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173,*",
            ).split(",")
            if origin.strip()
        ]
    )

    # Image upload settings
    MAX_IMAGE_SIZE_MB: int = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
    ALLOWED_IMAGE_MIME_TYPES: List[str] = field(
        default_factory=lambda: ["image/jpeg", "image/png", "image/webp"]
    )

    # Model settings
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "")
    DEVICE: str = os.getenv("DEVICE", "cpu")


settings = Settings()
