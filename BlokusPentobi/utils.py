
import multiprocessing
import os
import random
import subprocess
from typing import List

import numpy as np
import tensorflow as tf


def read_to_dataset(paths,
                    frac_test_files=0,
                    add_channel=False,
                    shuffle_files=True,
                    filter_files_fn = None):
    """ Create a tf dataset from a folder of files.
    If split_files_to_test_set is True, then frac_test_files of the files are used for testing.
    
    """
    assert 0 <= frac_test_files <= 1, "frac_test_files must be between 0 and 1"
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    if filter_files_fn is None:
        filter_files_fn = lambda x: True
    
    # Find all files in paths, that fit the filter_files_fn
    file_paths = [os.path.join(path, file) for path in paths for file in os.listdir(path) if filter_files_fn(file)]
    if shuffle_files:
        random.shuffle(file_paths)
        
    # Read one file to get the number of samples in a file
    with open(file_paths[0], "r") as f:
        num_samples = sum(1 for line in f)

    print("Found {} files".format(len(file_paths)))
    def txt_line_to_tensor(x):
        s = tf.strings.split(x, sep=",")
        s = tf.strings.to_number(s, out_type=tf.float32)
        return (s[:-1], s[-1])

    def ds_maker(x):
        ds = tf.data.TextLineDataset(x, num_parallel_reads=tf.data.experimental.AUTOTUNE)
        ds = ds.map(txt_line_to_tensor,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE,
                    deterministic=False)
        return ds
    
    test_files = file_paths[:int(frac_test_files*len(file_paths))]
    train_files = file_paths[int(frac_test_files*len(file_paths)):]
    
    if len(test_files) > 0:
        test_ds = tf.data.Dataset.from_tensor_slices(test_files)
        test_ds = test_ds.interleave(ds_maker,
                                cycle_length=tf.data.experimental.AUTOTUNE,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                deterministic=False)
        if add_channel:
            test_ds = test_ds.map(lambda x, y: (tf.expand_dims(x, axis=-1), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = tf.data.Dataset.from_tensor_slices(train_files)
    train_ds = train_ds.interleave(ds_maker,
                                cycle_length=tf.data.experimental.AUTOTUNE,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                deterministic=False)
    # Add a channel dimension if necessary
    if add_channel:
        train_ds = train_ds.map(lambda x, y: (tf.expand_dims(x, axis=-1), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if len(test_files) > 0:
        return train_ds, test_ds, len(file_paths), num_samples*len(file_paths)
    return train_ds, len(file_paths), num_samples*len(file_paths)

class TFLiteModel:
    """A class representing a tensorflow lite model."""
    
    # Class-level cache for models
    _model_cache = {}
    
    def __new__(cls, path: str, expand_input_dims: bool = False):
        """Check the cache before creating a new instance."""
        path = os.path.abspath(path)
        if path in cls._model_cache:
            return cls._model_cache[path]
        else:
            instance = super(TFLiteModel, cls).__new__(cls)
            cls._model_cache[path] = instance
            return instance
    
    def __init__(self, path: str, expand_input_dims: bool = False):
        """Initialize the model."""
        # Ensure __init__ is only called once per instance
        if not hasattr(self, 'initialized'):
            path = os.path.abspath(path)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found at {path}")
            self.interpreter = tf.lite.Interpreter(model_path=path)
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.interpreter.allocate_tensors()
            self.lock = multiprocessing.Lock()
            self.initialized = True
    
    def is_valid_size_input(self, X) -> bool:
        """Validate the input."""
        is_valid = X.shape[1:] == self.input_details[0]['shape'][1:]
        return True if all(is_valid) else False
        
    def predict(self, X) -> List[float]:
        """Predict the output of the model.
        The input should be a numpy array with size (batch_size, input_size)
        """
        with self.lock:
            if not self.is_valid_size_input(X):
                # Add a dimension to the input
                X = np.expand_dims(X, axis=-1)
                if not self.is_valid_size_input(X):
                    raise ValueError(f"Input shape {X.shape} is not valid for the model. Expected shape {self.input_details[0]['shape']}")
            self.interpreter.resize_tensor_input(self.input_details[0]['index'], X.shape)
            self.interpreter.allocate_tensors()
            self.interpreter.set_tensor(self.input_details[0]['index'], X)
            self.interpreter.invoke()
            out = self.interpreter.get_tensor(self.output_details[0]['index'])
            return list(out)

@tf.keras.utils.register_keras_serializable(name="BlokusPentobiMetric") 
class BlokusPentobiMetric(tf.keras.metrics.Metric):
    def __init__(self, model_tflite_path,
                 name='blokus_pentobi_metric',
                 ret_metric="average_score",
                 num_games=60,
                 num_cpus=10,
                 timeout=60,
                 **kwargs):
        super(BlokusPentobiMetric, self).__init__(name=name)
        self.model_tflite_path = model_tflite_path
        self.timeout = timeout
        assert ret_metric in ["average_score", "win_rate"], "ret_metric must be either 'average_score' or 'win_rate'"
        self.ret_metric = ret_metric
        self.num_games = num_games
        self.num_cpus = num_cpus
        self.win_rate = -1
        self.avg_score = -1
        self.num_calls = 0

    def update_state(self, y_true, y_pred, sample_weight=None):
        pass

    def result(self):
        # Calculate on every oddth call
        if self.num_calls % 2 != 0:
            return self.avg_score if self.ret_metric == "average_score" else self.win_rate
        self.num_calls += 1
        try:
            print(f"Running benchmark for model at {self.model_tflite_path}")
            command = f"python3 BlokusPentobi/benchmark.py --dont_update_win_rate --model_path={self.model_tflite_path} --num_games={self.num_games} --num_cpus={self.num_cpus}"
            output = subprocess.run(command.split(), capture_output=True, text=True, timeout=self.timeout).stdout
            print(output)
            
            # Parse the output for win rate and average score
            for line in output.splitlines():
                if line.startswith("Wins"):
                    wins_dict = eval(line.split("Wins ")[-1])
                    self.win_rate = wins_dict.get("PlayerToTest", -1)
                if line.startswith("Average score"):
                    score_dict = eval(line.split("Average score ")[-1])
                    self.avg_score = score_dict.get("PlayerToTest", -1)

            # Return win rate and average score as a tuple
            return self.avg_score if self.ret_metric == "average_score" else self.win_rate
        except Exception as e:
            print(f"Error running benchmark: {e}", flush=True)
            return self.avg_score if self.ret_metric == "average_score" else self.win_rate
        
    def get_config(self):
        return {"model_tflite_path": self.model_tflite_path,
                "num_games": self.num_games,
                "num_cpus": self.num_cpus,
                "timeout": self.timeout,
                "ret_metric": self.ret_metric}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def convert_model_to_tflite(file_path : str, output_file : str = None) -> None:
    if output_file is None:
        output_file = file_path.replace(".keras", ".tflite")
        
    print("Converting '{}' to '{}'".format(file_path, output_file))

    model = tf.keras.models.load_model(file_path, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    tflite_model = converter.convert()

    with open(output_file, "wb") as f:
        f.write(tflite_model)
    return output_file

if __name__ == "__main__":
    isg = BlokusPentobiMetric("model.tflite",
                              num_games=40,
                              num_cpus=10,
                              ret_metric="average_score")
    print(isg.result())
    
    json_config = isg.get_config()
    print(json_config)
    isg2 = BlokusPentobiMetric.from_config(json_config)
    print(isg2.result())
    exit(0)