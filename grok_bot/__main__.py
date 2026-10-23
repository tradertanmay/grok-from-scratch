import uvicorn

from .app import create_app
from .config import Settings
from .observability import setup_json_logging


def main() -> None:
    settings = Settings.from_env()
    if settings.log_json:
        setup_json_logging()
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_config=None if settings.log_json else uvicorn.config.LOGGING_CONFIG,
        access_log=not settings.log_json,
    )


if __name__ == "__main__":
    main()
