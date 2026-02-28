from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    APP_ENV: str = "development"
    PROJECT_NAME: str = "PlaygroundAgent"
    DEBUG: bool = True
    VERSION: str = "0.1.0"

    # API
    API_V1_PREFIX: str = "/api/v1"

    # LLM — provider selection
    LLM_PROVIDER: str = "anthropic"  # "anthropic" | "openai" | "google"

    # Anthropic / Claude
    ANTHROPIC_API_KEY: str = ""
    DEFAULT_LLM_MODEL: str = "claude-sonnet-4-20250514"

    # OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o"

    # Google Gemini
    GOOGLE_API_KEY: str = ""
    GOOGLE_MODEL: str = "gemini-2.0-flash"

    DEFAULT_LLM_TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 2048

    # Security
    SECRET_KEY: str = "change-me-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Rate limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 60
    RATE_LIMIT_WINDOW: int = 60  # seconds

    # PostgreSQL
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "playground_agent"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
