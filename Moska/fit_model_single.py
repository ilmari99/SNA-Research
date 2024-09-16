import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from RLFramework.read_to_dataset import read_to_dataset
from RLFramework.utils import convert_model_to_tflite
import argparse

def residual_block(x, filters, kernel_size=(3,3)):
    y = keras.layers.Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
    y = keras.layers.Conv2D(filters, kernel_size, padding='same')(y)
    y = keras.layers.Add()([x, y])
    y = keras.layers.ReLU()(y)
    return y

def get_model(input_shape, tflite_path=None):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(inputs)
    
    x = tf.keras.layers.Dense(400, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(400, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(400, activation="relu")(x)
    x = tf.keras.layers.Dense(400, activation="relu")(x)
    
    output = keras.layers.Dense(1,activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="binary_crossentropy"
    )
    return model

class SaveModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_save_path):
        super(SaveModelCallback, self).__init__()
        self.model_save_path = model_save_path

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.model_save_path)
        convert_model_to_tflite(self.model_save_path)
    
def main(data_folder,
         model_save_path,
         load_model_path = None,
         log_dir = "./logs/",
         num_epochs=25,
         patience=5,
         validation_split=0.2,
         batch_size=64,
         ):
    data_folders = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]
    print(data_folders)
    
    if len(tf.config.experimental.list_physical_devices('GPU')) == 1:
        print("Using single GPU")
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    elif len(tf.config.experimental.list_physical_devices('GPU')) > 1:
        print("Using multiple GPUs")
        strategy = tf.distribute.MirroredStrategy()
    else:
        print("Using CPU")
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    with strategy.scope():
        train_ds, val_ds, num_files, approx_num_samples = read_to_dataset(data_folders, frac_test_files=validation_split)
        
        input_shape = train_ds.take(1).as_numpy_iterator().next()[0].shape
        print(f"Input shape: {input_shape}")
        print(f"Num samples: {approx_num_samples}")
        
        train_ds = train_ds.shuffle(10000).batch(batch_size)
        val_ds = val_ds.batch(batch_size)
        
        train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
        val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)
        
        if load_model_path:
            model = tf.keras.models.load_model(load_model_path)
        else:
            model = get_model(input_shape)
            print(model.summary())

                # Compile the model, keeping optimizer and loss, but adding metrics
        metrics = ['mae',"accuracy","binary_crossentropy"]
        model.compile(optimizer=model.optimizer, loss=model.loss, metrics=metrics)
        
        tb_log = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        save_model_cb = SaveModelCallback(model_save_path)
        model.fit(train_ds, epochs=num_epochs, callbacks=[tb_log, early_stop, save_model_cb], validation_data=val_ds,class_weight={0: 0.75, 1: 0.25})
    model.save(model_save_path)
    convert_model_to_tflite(model_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with given data.')
    parser.add_argument('--data_folder', type=str, required=True,
                        help='Folders containing the data. Provide as a space separated list.')
    parser.add_argument('--load_model_path', type=str, help='Path to load a model from.', default=None)
    parser.add_argument('--model_save_path', type=str, required=True, help='Path to save the trained model.')
    parser.add_argument('--log_dir', type=str, required=False, help='Directory for TensorBoard logs.', default="./moskalogs/")
    parser.add_argument('--num_epochs', type=int, help='Number of epochs to train.', default=25)
    parser.add_argument('--patience', type=int, help='Patience for early stopping.', default=5)
    parser.add_argument('--validation_split', type=float, help='Validation split.', default=0.2)
    parser.add_argument('--batch_size', type=int, help='Batch size.', default=64)
    args = parser.parse_args()
    print(args)
    main(data_folder=args.data_folder,
            model_save_path=args.model_save_path,
            load_model_path=args.load_model_path,
            log_dir=args.log_dir,
            num_epochs=args.num_epochs,
            patience=args.patience,
            validation_split=args.validation_split,
            batch_size=args.batch_size
            )
    convert_model_to_tflite(args.model_save_path)
    exit(0)
    
    
    