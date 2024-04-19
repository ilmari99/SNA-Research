import os
import tensorflow as tf
import numpy as np
from RLFramework.read_to_dataset import read_to_dataset
from RLFramework.utils import convert_model_to_tflite
import argparse

def get_model(input_shape):
    
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Dense(600, activation='relu')(inputs)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(500, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.35)(x)
        x = tf.keras.layers.Dense(500, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.35)(x)
        x = tf.keras.layers.Dense(500, activation='relu')(x)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=output)

        model.compile(optimizer="adam",
                loss='binary_crossentropy',
                metrics=['mae', "accuracy"]
        )
        return model
    
def main(data_folders,
         model_save_path,
         load_model_path = None,
         log_dir = "./logs/",
         num_epochs=25,
         patience=5,
         validation_split=0.2,
         batch_size=64,
         ):
    # Check that all paths are folders
    data_folders = [os.path.abspath(folder) for folder in data_folders]
    assert all([os.path.isdir(folder) for folder in data_folders]), "All data folders must be directories."
    print(data_folders)
    
    ds, num_files, approx_num_samples = read_to_dataset(data_folders)
    
    input_shape = ds.take(1).as_numpy_iterator().next()[0].shape
    print(f"Input shape: {input_shape}")
    print(f"Num samples: {approx_num_samples}")
    
    train_ds = ds.take(int((1-validation_split)*approx_num_samples)).batch(batch_size)
    val_ds = ds.skip(int((1-validation_split)*approx_num_samples)).batch(batch_size)
    
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    
    if load_model_path:
        model = tf.keras.models.load_model(load_model_path)
    else:
        model = get_model(input_shape)
        print(model.summary())
    
    tb_log = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    model.fit(train_ds, epochs=num_epochs, callbacks=[tb_log, early_stop], validation_data=val_ds)
    model.save(model_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with given data.')
    parser.add_argument('--data_folders', type=str, required=True, nargs='+',
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
    main(data_folders=args.data_folders,
            model_save_path=args.model_save_path,
            load_model_path=args.load_model_path,
            log_dir=args.log_dir,
            num_epochs=args.num_epochs,
            patience=args.patience,
            validation_split=args.validation_split,
            batch_size=args.batch_size)
    convert_model_to_tflite(args.model_save_path)
    exit(0)
    
    
    