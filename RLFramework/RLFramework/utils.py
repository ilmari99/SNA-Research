import logging
from typing import Any, Dict, List, Tuple


class _NoneLogger(logging.Logger):
    """ A logger that does nothing.
    """
    def __init__(self):
        pass
    def debug(self, *args, **kwargs):
        pass
    def info(self, *args, **kwargs):
        pass
    def warning(self, *args, **kwargs):
        pass
    def error(self, *args, **kwargs):
        pass
    def critical(self, *args, **kwargs):
        pass

def _get_logger(logger_args: Dict[str, Any] = None):
    """ If there are no arguments, return a logger that doesn't do anything.
    Otherwise, construct a logger with the given arguments.
    """
    if logger_args is None:
        return _NoneLogger()
    logger = logging.getLogger(logger_args.get("name", "Game"))
    logger.setLevel(logger_args.get("level", logging.INFO))
    handler = logging.StreamHandler()
    handler.setLevel(logger_args.get("level", logging.INFO))
    formatter = logging.Formatter(logger_args.get("format", '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger