import os
import time
import numpy as np
import tensorflow as tf
import multiprocessing
# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def read_to_dataset(file_path: str):
    """ Read the data to a dataset.
    """
    def parse_line(line):
        values = tf.strings.split([line], sep=',').values
        x = tf.strings.to_number(values[:-1], out_type=tf.float32)
        y = tf.strings.to_number(values[-1], out_type=tf.float32)
        return x, y

    ds = tf.data.TextLineDataset(file_path)
    ds = ds.map(parse_line)
    return ds
    

class TFLiteModel:
    def __init__(self, path):
        """ Initialize the model.
        """
        path = os.path.abspath(path)
        self.interpreter = tf.lite.Interpreter(model_path=path)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.interpreter.allocate_tensors()
        
    def predict(self, X : np.ndarray):
        self.interpreter.resize_tensor_input(self.input_details[0]['index'], X.shape)
        self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(self.input_details[0]['index'], X)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details[0]['index'])
        return list(out)

def convert_keras_model_to_tflite(input_path, output_path) -> None:
        
    print("Converting '{}' to '{}'".format(input_path, output_path))

    model = tf.keras.models.load_model(input_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    tflite_model = converter.convert()
    
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    return output_path

def test_model(args):
    """ Test the model.
    """
    #print(args)
    model_path, data_file = args
    # Convert the model to a tflite model
    model_path = model_path.replace(".keras", ".tflite")
    print(f"Testing model {model_path}...")
    tflite_model = TFLiteModel(model_path)
    
    # Load the data
    ds = read_to_dataset(data_file)
    
    # Check that the model works
    num_samples = 0
    mae = 0
    for x, y in ds:
        x = x.numpy()
        x = np.expand_dims(x,axis=0)
        pred = tflite_model.predict(x)
        mae += np.abs(pred[0] - y)
        num_samples += x.shape[0]
    
    print(f"Test done. Mae: {mae / num_samples}")
    return
    
def make_tf_model(model_path = "model_0.keras"):
    """ Create a simple model.
    """
    inp = tf.keras.layers.Input(shape=(402,))
    out = tf.keras.layers.Dense(1)(inp)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mse')
    model.save(model_path)
    return

if __name__ == "__main__":
    
    # IF the multiprocessing is set to "spawn", everything works fine.
    # On Linux, the default is "fork", which causes the program to hang.
    #multiprocessing.set_start_method('spawn')
    
    # Replace keras -> tflite
    data_path = "Data/data_0.csv"
    model_path = "HangTestFolder/model_0.keras"
    tflite_path = model_path.replace(".keras", ".tflite")
    
    # Create the model, and convert it to a tflite model
    make_tf_model(model_path)
    convert_keras_model_to_tflite(model_path, tflite_path)
    
    # Test the model using just the main process.
    # These should work fine.
    print(f"----------------------Testing on main process----------------------")
    #test_model((model_path, data_path))
    #test_model((model_path, data_path))
    print(f"----------------------Test on main process done----------------------")
    
    # Now, when we try to test the model using a different process, the program just hangs.
    # If creating the model and testing on the main thread are commented out, _the program works fine_
    n_proc = 2
    def arg_gen():
        for _ in range(n_proc):
            yield (model_path, data_path)
            
    with multiprocessing.Pool(n_proc) as p:
        p.map(test_model, arg_gen())
        
    