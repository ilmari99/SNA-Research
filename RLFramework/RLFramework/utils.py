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
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        self.interpreter = tf.lite.Interpreter(model_path=path)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.interpreter.allocate_tensors()
        
    def is_valid_size_input(self, X) -> bool:
        """ Validate the input.
        """
        is_valid = X.shape[1:] == self.input_details[0]['shape'][1:]
        return True if all(is_valid) else False
        
    def predict(self, X) -> List[float]:
        """ Predict the output of the model.
        The input should be a numpy array with size (batch_size, input_size)
        """
        if not self.is_valid_size_input(X):
            # Add a dimension to the input
            X = np.expand_dims(X, axis = -1)
            if not self.is_valid_size_input(X):
                raise ValueError(f"Input shape {X.shape} is not valid for the model. Expected shape {self.input_details[0]['shape']}")
        self.interpreter.resize_tensor_input(self.input_details[0]['index'], X.shape)
        self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(self.input_details[0]['index'], X)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details[0]['index'])
        return list(out)
        
class _NoneLogger(logging.Logger):
    """ A logger that does nothing.
    """
    @property
    def name(self):
        return "NoneLogger"
    
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