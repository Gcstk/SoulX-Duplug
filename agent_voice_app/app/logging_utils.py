from __future__ import annotations

import datetime as dt
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

DEFAULT_LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "agent_voice_app.log"


class MillisecondFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):  # noqa: N802
        timestamp = dt.datetime.fromtimestamp(record.created)
        return timestamp.strftime("%Y-%m-%d %H:%M:%S.") + f"{int(record.msecs):03d}"


def setup_logging() -> Path:
    level_name = os.getenv("AGENT_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, level_name, logging.INFO)
    log_path = Path(os.getenv("AGENT_LOG_FILE", str(DEFAULT_LOG_FILE))).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = MillisecondFormatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    root = logging.getLogger()
    root.setLevel(log_level)

    if not any(getattr(h, "_agent_voice_app_handler", False) for h in root.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(log_level)
        stream_handler.setFormatter(formatter)
        stream_handler._agent_voice_app_handler = True  # type: ignore[attr-defined]
        root.addHandler(stream_handler)

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        file_handler._agent_voice_app_handler = True  # type: ignore[attr-defined]
        root.addHandler(file_handler)

    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("agent_voice_app").info("logging initialized log_file=%s", log_path)
    return log_path


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"agent_voice_app.{name}")
