from __future__ import annotations

import os

import uvicorn
from dotenv import load_dotenv

from app.logging_utils import setup_logging


def main() -> None:
    load_dotenv()
    log_path = setup_logging()
    port = int(os.getenv("PORT", "3040"))
    import logging

    logging.getLogger("agent_voice_app.main").info(
        "starting uvicorn host=0.0.0.0 port=%s log_file=%s", port, log_path
    )
    uvicorn.run("server:app", host="0.0.0.0", port=port, log_level="info", reload=False)


if __name__ == "__main__":
    main()
