"""Aethra entry point: python run.py"""

import uvicorn

from app.config import Settings
from app.logging_setup import setup_logging


def main() -> None:
    settings = Settings.load()
    setup_logging(settings.log_level)
    print(f"Starting {settings.app_name} backend ...")
    print(f"  LLM API : {settings.llm_base_url} (model: {settings.llm_model})")
    print(f"  Data dir: {settings.data_dir}")
    uvicorn.run(
        "app.main:create_app",
        factory=True,
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
