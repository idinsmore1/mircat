import sys
import json

from time import time
from pathlib import Path

from loguru import logger


# Quick timing wrapper
def timer(func):
    """Time a functions runtime
    :param func: the function to time
    :return: the time measured, and the result of the function
    """

    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        # logger.info(f'{func.__name__!r} completed in {(t2-t1):.2f}s')
        return result, round(t2 - t1, 2)

    return wrap_func


# Logging functions
def get_project_root() -> Path:
    return Path(__file__).parent.parent


def serialize(record):
    subset = {
        "level": record["level"].name,
        "timestamp": record["time"].strftime("%Y-%m-%d %H:%M:%S"),
        "message": record["message"],
        # "exception": str(record["exception"]['type']),
        "module": record["module"],
        "line": record["line"],
        **record["extra"],
    }
    return json.dumps(subset)


def formatter(record):
    record["extra"]["serialized"] = serialize(record)
    return "{extra[serialized]}\n"


def configure_logging(log_file_path, verbose: bool):
    # Remove all handlers
    logger.remove()
    # Log all logs except INFO to a file
    logger.add(
        log_file_path,
        format=formatter,
        # serialize=True,
        level="DEBUG",
        rotation="10 GB",
        compression="gz",
        enqueue=True,
        retention=10,
        filter=lambda record: record["level"].name != "INFO",
    )
    if verbose:
        # Print INFO and SUCCESS logs to stdout and WARNING, ERROR, and CRITICAL to stderr
        logger.add(
            sys.stdout,
            colorize=True,
            format="<green>{time: DD-MM-YYYY -> HH:mm:ss}</green> <level>{message}</level>",
            level="INFO",
            filter=lambda record: record["level"].name in ["INFO", "SUCCESS"],
            enqueue=True,
        )
        logger.add(
            sys.stderr,
            colorize=True,
            format="<red>{time: DD-MM-YYYY -> HH:mm:ss}</red> <level>{message}</level>",
            level="WARNING",
            filter=lambda record: record["level"].name
            not in ["DEBUG", "INFO", "SUCCESS"],
            enqueue=True,
        )
    else:
        # Only print INFO logs to stdout
        logger.add(
            sys.stdout,
            colorize=True,
            format="<green>{time: DD-MM-YYYY -> HH:mm:ss}</green> <level>{message}</level>",
            level="INFO",
            filter=lambda record: record["level"].name == "INFO",
            enqueue=True,
        )
    # Always print WARNING, ERROR, and CRITICAL to stderr
