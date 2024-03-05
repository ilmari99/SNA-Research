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
    default_args = {
        "name" : "Game",
        "level" : logging.INFO,
        "format" : '%(name)s - %(levelname)s - %(message)s',
        # stdout
        "log_file" : None,
        "write_mode" : "w",
    }
    logger_args = {**default_args, **logger_args}
    logger = logging.getLogger(logger_args.get("name", "Game"))
    logger.setLevel(logger_args.get("log_level", logging.INFO))
    formatter = logging.Formatter(logger_args.get("format", default_args["format"]))
    if logger_args.get("log_file", None):
        file_handler = logging.FileHandler(logger_args["log_file"], mode = logger_args.get("write_mode", "w"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger