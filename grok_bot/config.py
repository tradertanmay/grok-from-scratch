import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader: KEY=VALUE lines; existing env vars win."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        os.environ.setdefault(key, value)


def _flag(value: str | None, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "gpt-oss:latest"
    llm_api_key: str = "not-needed"
    db_path: str = "grok_bot.sqlite3"
    max_tool_steps: int = 8
    searxng_url: str = ""
    host: str = "127.0.0.1"
    port: int = 8000
    providers_file: str = "providers.json"
    fallback_models: tuple[str, ...] = ()
    auth_enabled: bool = True
    allow_signup: bool = True
    signup: str = "open"  # open | invite | closed
    admin_username: str = ""
    admin_password: str = ""
    cookie_secure: bool = False
    chat_rate_limit: int = 30
    auth_rate_limit: int = 10
    memory_enabled: bool = True
    embed_model: str = "nomic-embed-text"
    vision_model: str = ""
    think_model: str = ""
    daily_messages: int = 0  # per user per UTC day; 0 = unlimited, admins exempt
    metrics_token: str = ""
    log_json: bool = False
    tasks_enabled: bool = True
    max_upload_mb: int = 20
    stt_url: str = ""
    connect_timeout: float = 5.0
    max_concurrent_primary: int = 0  # 0 = no spillover
    stt_model: str = "whisper-1"
    stt_api_key: str = ""
    mcp_file: str = "mcp.json"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        env = os.environ
        return cls(
            llm_base_url=env.get("LLM_BASE_URL", cls.llm_base_url).rstrip("/"),
            llm_model=env.get("LLM_MODEL", cls.llm_model),
            llm_api_key=env.get("LLM_API_KEY", cls.llm_api_key),
            db_path=env.get("GROK_DB_PATH", cls.db_path),
            max_tool_steps=int(env.get("GROK_MAX_TOOL_STEPS", cls.max_tool_steps)),
            searxng_url=env.get("SEARXNG_URL", cls.searxng_url).rstrip("/"),
            host=env.get("HOST", cls.host),
            port=int(env.get("PORT", cls.port)),
            providers_file=env.get("GROK_PROVIDERS_FILE", cls.providers_file),
            mcp_file=env.get("GROK_MCP_FILE", cls.mcp_file),
            fallback_models=tuple(
                m.strip() for m in env.get("GROK_FALLBACK_MODELS", "").split(",") if m.strip()
            ),
            auth_enabled=_flag(env.get("GROK_AUTH"), cls.auth_enabled),
            allow_signup=_flag(env.get("GROK_ALLOW_SIGNUP"), cls.allow_signup),
            signup=_signup_mode(env),
            admin_username=env.get("GROK_ADMIN_USER", ""),
            admin_password=env.get("GROK_ADMIN_PASSWORD", ""),
            cookie_secure=_flag(env.get("GROK_COOKIE_SECURE"), cls.cookie_secure),
            chat_rate_limit=int(env.get("GROK_CHAT_RATE_LIMIT", cls.chat_rate_limit)),
            auth_rate_limit=int(env.get("GROK_AUTH_RATE_LIMIT", cls.auth_rate_limit)),
            memory_enabled=_flag(env.get("GROK_MEMORY"), cls.memory_enabled),
            embed_model=env.get("GROK_EMBED_MODEL", cls.embed_model).strip(),
            vision_model=env.get("GROK_VISION_MODEL", cls.vision_model).strip(),
            think_model=env.get("GROK_THINK_MODEL", cls.think_model).strip(),
            daily_messages=int(env.get("GROK_DAILY_MESSAGES") or cls.daily_messages),
            metrics_token=env.get("GROK_METRICS_TOKEN", "").strip(),
            log_json=_flag(env.get("GROK_LOG_JSON"), cls.log_json),
            tasks_enabled=_flag(env.get("GROK_TASKS"), cls.tasks_enabled),
            max_upload_mb=int(env.get("GROK_MAX_UPLOAD_MB", cls.max_upload_mb)),
            connect_timeout=float(env.get("GROK_CONNECT_TIMEOUT") or cls.connect_timeout),
            max_concurrent_primary=int(
                env.get("GROK_MAX_CONCURRENT_PRIMARY") or cls.max_concurrent_primary
            ),
            stt_url=env.get("GROK_STT_URL", "").strip().rstrip("/"),
            stt_model=env.get("GROK_STT_MODEL", "").strip() or cls.stt_model,
            stt_api_key=env.get("GROK_STT_KEY", "").strip(),
        )


def _signup_mode(env) -> str:
    mode = env.get("GROK_SIGNUP", "").strip().lower()
    if not mode:
        return "open" if _flag(env.get("GROK_ALLOW_SIGNUP"), True) else "closed"
    if mode not in {"open", "invite", "closed"}:
        raise ValueError(f"GROK_SIGNUP must be open, invite or closed, not {mode!r}")
    return mode
