import os
import glob
from utils import convert_model_to_tflite
import argparse
import board_norming

def convert_keras_models_to_tflite(folder_path):
    # Find all *.keras files in the given folder
    keras_files = glob.glob(os.path.join(folder_path, '*.keras'))

    # Convert each keras file to tflite
    for keras_file in keras_files:
        # If the tflite file already exists, rename it to .oldtflite
        tflite_file = keras_file.replace('.keras', '.tflite')
        if os.path.exists(tflite_file):
            os.rename(tflite_file, tflite_file.replace('.tflite', '.oldtflite'))
        convert_model_to_tflite(keras_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert keras models to tflite')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing keras models')
    args = parser.parse_args()
    
    print(f"Converting keras models in folder {args.folder_path} to tflite...")
    convert_keras_models_to_tflite(args.folder_path)