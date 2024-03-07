import logging
from typing import Any, Dict, List, Tuple
import os
import numpy as np
import tensorflow as tf

class TFLiteModel:
    """ A class representing a tensorflow lite model.
    """
    def __init__(self, path : str, expand_input_dims : bool = False):
        """ Initialize the model.
        """
        path = os.path.abspath(path)
        self.expand_input_dims = expand_input_dims
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        self.interpreter = tf.lite.Interpreter(model_path=path)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.interpreter.allocate_tensors()
        
    def validate_input(self, X) -> None:
        """ Validate the input.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError(f"X should be a numpy array, not {type(X)}")
        if X.shape != self.input_details[0]['shape'][1:]:
            raise ValueError(f"X should have shape {self.input_details[0]['shape'][1:]}, not {X.shape}")
        
    def predict(self, X) -> List[float]:
        """ Predict the output of the model.
        The input should be a numpy array with size (batch_size, input_size)
        """
        if self.expand_input_dims:
            X = np.expand_dims(X, axis=-1)
        self.validate_input(X)
        self.interpreter.set_tensor(self.input_details[0]['index'], X)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details[0]['index'])
        out = out[0]
        
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