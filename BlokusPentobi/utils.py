
import os
from typing import List

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