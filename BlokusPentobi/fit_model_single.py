import os
import sys
import keras
import tensorflow as tf
import numpy as np
from utils import read_to_dataset
from utils import convert_model_to_tflite
from utils import BlokusPentobiMetric
import argparse

from board_norming import NormalizeBoardToPerspectiveLayer, separate_to_patches


class SaveModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_save_path):
        super(SaveModelCallback, self).__init__()
        self.model_save_path = model_save_path

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.model_save_path)
        convert_model_to_tflite(self.model_save_path)

def get_model(input_shape, tflite_path=None):
    inputs = tf.keras.Input(shape=input_shape)
    #input_len = input_shape[1]
    
    # Separate the input into the board and the rest
    # Board is everything except the first 2 elements
    board = inputs[:,2:]
    meta = inputs[:,:2]
    
    meta = tf.keras.layers.Flatten()(meta)
    # This element tells whose perspective of the game we are evaluating.
    perspective_pids = meta[:,0]
    perspective_pids = tf.cast(perspective_pids, tf.int32)
    
    # Reshape the board
    board_side_len = int(np.sqrt(board.shape[1]))
    board = tf.reshape(board, (-1, board_side_len, board_side_len))
    
    # Normalize the board to the perspective of the player
    board = NormalizeBoardToPerspectiveLayer()([board, perspective_pids])
    
    board = tf.reshape(board, (-1, board_side_len, board_side_len, 1))
    
    # Convert the board to a tensor with 5 channels, i.e. one-hot encode the values -1...3
    board = board + 1
    
    # Embed each value (0 ... 4) to 16 dimensions
    board = tf.keras.layers.Embedding(5, 16)(board)
    board = tf.reshape(board, (-1, board_side_len, board_side_len, 16))
    
    # Apply convolutions
    #x = keras.layers.Conv2D(16, (3,3), activation='linear')(board)
    #x = keras.layers.BatchNormalization()(x)
    #x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(32, (3,3), activation='linear')(board)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(64, (3,3), activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(128, (3,3), activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    output = keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
        loss='binary_crossentropy',
    )
    return model
    
def main(data_folder,
         model_save_path,
         load_model_path = None,
         log_dir = "./logs/",
         num_epochs=25,
         patience=5,
         validation_split=0.2,
         batch_size=64,
         divide_y_by=1
         ):
    # Find all folders inside the data_folder
    data_folders = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]
    data_folders += [data_folder]
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
        
        train_ds, val_ds, num_files, approx_num_samples = read_to_dataset(data_folders, frac_test_files=validation_split,filter_files_fn=lambda x: x.endswith(".csv"))
        
        if divide_y_by != 1:
            train_ds = train_ds.map(lambda x, y: (x, y/divide_y_by), num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)
            val_ds = val_ds.map(lambda x, y: (x, y/divide_y_by), num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)
        
        first_sample = train_ds.take(1).as_numpy_iterator().next()
        input_shape = first_sample[0].shape
        print(f"First sample: {first_sample}")
        #input_shape = (20*20 +2,)
        print(f"Input shape: {input_shape}")
        print(f"Num samples: {approx_num_samples}")
        
        train_ds = train_ds.shuffle(1000).batch(batch_size)
        val_ds = val_ds.batch(batch_size)
        
        train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
        val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)
        
        if load_model_path:
            model = tf.keras.models.load_model(load_model_path)
        else:
            model = get_model(input_shape, model_save_path.replace(".keras", ".tflite"))
            print(model.summary())
        
        # Compile the model, keeping optimizer and loss, but adding metrics
        metrics = ['mae',"mse","binary_crossentropy",BlokusPentobiMetric(model_save_path.replace(".keras", ".tflite"))]
        model.compile(optimizer=model.optimizer, loss=model.loss, metrics=metrics)
        
        
        tb_log = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        save_model_cb = SaveModelCallback(model_save_path)
        model.fit(train_ds, epochs=num_epochs, callbacks=[tb_log, early_stop, save_model_cb], validation_data=val_ds)
    model.save(model_save_path)
    convert_model_to_tflite(model_save_path)
    
    # Run benchmark.py to test the model
    model_tflite_path = model_save_path.replace(".keras", ".tflite")
    os.system(f"python3 BlokusPentobi/benchmark.py --model_path={model_tflite_path} --num_games=60 --num_cpus=10")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with given data.')
    parser.add_argument('--data_folder', type=str, required=True,
                        help='Folder containing the data.')
    parser.add_argument('--load_model_path', type=str, help='Path to load a model from.', default=None)
    parser.add_argument('--model_save_path', type=str, required=True, help='Path to save the trained model.')
    parser.add_argument('--log_dir', type=str, required=False, help='Directory for TensorBoard logs.', default="./blokuslogs/")
    parser.add_argument('--num_epochs', type=int, help='Number of epochs to train.', default=25)
    parser.add_argument('--patience', type=int, help='Patience for early stopping.', default=5)
    parser.add_argument('--validation_split', type=float, help='Validation split.', default=0.2)
    parser.add_argument('--batch_size', type=int, help='Batch size.', default=64)
    parser.add_argument('--divide_y_by', type=int, required=False, help='Divide y by this number.', default=159)
    args = parser.parse_args()
    print(args)
    main(data_folder=args.data_folder,
            model_save_path=args.model_save_path,
            load_model_path=args.load_model_path,
            log_dir=args.log_dir,
            num_epochs=args.num_epochs,
            patience=args.patience,
            validation_split=args.validation_split,
            batch_size=args.batch_size,
            divide_y_by=args.divide_y_by)
    exit(0)
    
    
    