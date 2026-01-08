from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

_DEFAULT_LOG_LEVEL = logging.INFO
_DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
_DEFAULT_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
_LOGGING_INITIALIZED = False


def init_logging(*, level: int = _DEFAULT_LOG_LEVEL, log_path: Optional[Path] = None) -> None:
    global _LOGGING_INITIALIZED
    root = logging.getLogger()
    if not _LOGGING_INITIALIZED:
        root.setLevel(level)
        formatter = logging.Formatter(_DEFAULT_LOG_FORMAT, datefmt=_DEFAULT_LOG_DATEFMT)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)
        _LOGGING_INITIALIZED = True
    if log_path is not None:
        resolved = Path(log_path).expanduser().resolve()
        for handler in root.handlers:
            if isinstance(handler, logging.FileHandler):
                if Path(handler.baseFilename).resolve() == resolved:
                    return
        file_handler = logging.FileHandler(resolved)
        file_handler.setFormatter(logging.Formatter(_DEFAULT_LOG_FORMAT, datefmt=_DEFAULT_LOG_DATEFMT))
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    if not _LOGGING_INITIALIZED:
        init_logging()
    return logging.getLogger(name)


def _format_value(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return "[" + ",".join(_format_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    if value is None:
        return "null"
    return str(value)


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    if not fields:
        logger.info(event)
        return
    parts = [f"{key}={_format_value(fields[key])}" for key in sorted(fields)]
    logger.info("%s %s", event, " ".join(parts))
