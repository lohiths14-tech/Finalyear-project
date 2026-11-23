import logging
import os
import sys


def setup_logging() -> logging.Logger:
    """Configure structured logging with console + file handlers."""
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    # Write to file only when a writable logs dir exists (Docker volume)
    log_dir = os.environ.get("LOG_DIR", "./logs")
    try:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, "agrivision.log")
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    except OSError:
        # Gracefully skip file logging if the path is not writable
        pass

    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

    return logging.getLogger("agrivision")


logger = setup_logging()
