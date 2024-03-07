#!/usr/bin/env python3
import os
import tensorflow as tf
import sys
import argparse

class TieToLoss(tf.keras.losses.Loss):
    """ A custom loss function, that is BCE.
    If y_true is 0.5 (tie), then the y_true will be set to 0
    """
    def __init__(self, name="tie_to_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        
    def call(self, y_true, y_pred):
        y_true = tf.where(y_true == 0.5, 0.0, y_true)
        return self.bce(y_true, y_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert a keras model to tflite")
    parser.add_argument("file_path", help="Path to the keras model", default="model.keras")
    parser.add_argument("--output_file", help="Path to the output file", default=None)
    parser = parser.parse_args()
    output_file = parser.output_file
    file_path = parser.file_path
    if output_file is None:
        output_file = file_path.replace(".keras", ".tflite")
        
    print("Converting '{}' to '{}'".format(file_path, output_file))

    model = tf.keras.models.load_model(file_path, custom_objects={"TieToLoss": TieToLoss})
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(output_file, "wb") as f:
        f.write(tflite_model)
    print("Done.")