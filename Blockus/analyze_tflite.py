from collections import deque
from itertools import chain
import os
import sys
import time
from typing import List
import numpy as np
import tensorflow as tf
from RLFramework.read_to_dataset import read_to_dataset
import board_norming
os.environ["CUDA_VISIBLE_DEVICES"] = ""


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

    def __sizeof__(self):
        """ Get the size of the model.
        """
        return sys.getsizeof(self.interpreter, 0)

def convert_model_to_tflite(file_path : str, output_file : str = None) -> None:
    if output_file is None:
        output_file = file_path.replace(".keras", ".tflite")
        
    print("Converting '{}' to '{}'".format(file_path, output_file))

    model = tf.keras.models.load_model(file_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS, # enable TensorFlow ops.
        
    ]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    

    with open(output_file, "wb") as f:
        f.write(tflite_model)
    return output_file


if __name__ == "__main__":
    model_path = "model.keras"
    tflite_path = "model_test.tflite"
    
    # Convert the model to a tflite model
    convert_model_to_tflite(model_path, tflite_path)
    
    tflite_model = TFLiteModel(tflite_path)
    
    # Show size of the tflite model
    model_size = sys.getsizeof(tflite_model)
    model_size_kb = model_size / 1024
    print(f"Size of tflite model: {model_size_kb} KB")
    
    # Load the data
    data_folder = "TestIndividualPlayers"
    ds, _, _ = read_to_dataset(data_folder)
    
    # Test tflite model inference speed
    t_start = time.time()
    num_samples = 0
    mae = 0
    for x, y in ds:
        x = x.numpy()
        x = np.expand_dims(x,axis=0)
        #print(f"Predicting {x.shape}")
        pred = tflite_model.predict(x)
        mae += np.abs(pred[0] - y)
        num_samples += x.shape[0]
    t_end = time.time()
    
    print(f"Mean absolute error: {mae / num_samples}")
    print(f"Time to predict {num_samples} samples: {t_end - t_start}")
    