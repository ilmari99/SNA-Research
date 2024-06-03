import os
import sys
import keras
import tensorflow as tf
import numpy as np
from RLFramework.read_to_dataset import read_to_dataset
from RLFramework.utils import convert_model_to_tflite
import argparse
#tf.compat.v1.disable_eager_execution()


@tf.keras.saving.register_keras_serializable()
class RandomRotateBoardLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RandomRotateBoardLayer, self).__init__(**kwargs)

    def call(self, board, training=None):
        # Randomly rotate the board only during training
        if training:
            board = tf.image.rot90(board, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
        return board

@tf.keras.saving.register_keras_serializable()
class RandomFlipBoardLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RandomFlipBoardLayer, self).__init__(**kwargs)
        
    def call(self, board, training=None):
        # Randomly flip the board only during training
        if training:
            board = tf.image.random_flip_left_right(board)
            board = tf.image.random_flip_up_down(board)
        return board

@tf.keras.saving.register_keras_serializable()
class RotLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RotLayer, self).__init__(**kwargs)
        
    def call(self, inputs, training=None):
        board = tf.vectorized_map(rotate90, inputs)
        return board

@tf.keras.saving.register_keras_serializable()
def rotate90(x):
    boards = tf.reshape(x[:-1], (20, 20, 1))
    rots = tf.cast(x[-1], tf.int32)
    return tf.image.rot90(boards, k=rots)

@tf.keras.saving.register_keras_serializable()
def rotate_board_to_perspective_tf(board, perspective_pid):
    
    perspective_pid = tf.reshape(perspective_pid, (-1,1))
    
    top_left_pids = tf.reshape(board[:,0,0], (-1,1))
    top_right_pids = tf.reshape(board[:,0,-1], (-1,1))
    bottom_right_pids = tf.reshape(board[:,-1,-1], (-1,1))
    bottom_left_pids = tf.reshape(board[:,-1,0], (-1,1))
    
    corner_pids = tf.concat([top_left_pids, top_right_pids, bottom_right_pids, bottom_left_pids], axis=1)
    corner_pids = tf.reshape(corner_pids, (-1, 4))
    corner_pids = tf.cast(corner_pids, tf.int32)
    
    corner_index = tf.argmax(tf.cast(tf.equal(corner_pids, perspective_pid), tf.float32), axis=1)
    corner_index = tf.reshape(corner_index, (-1,1))
    
    board = tf.reshape(board, (-1, 20*20))
    board = tf.cast(board, tf.float32)
    corner_index = tf.cast(corner_index, tf.float32)
    board_rot_pairs = tf.concat([board, corner_index], axis=1)
    board_rot_pairs = tf.reshape(board_rot_pairs, (-1, 20*20+1))
    #print(f"Board rot pairs: {board_rot_pairs}")
    
    board = RotLayer()(board_rot_pairs)
    
    board = tf.reshape(board, (-1, 20, 20))
    
    return board

@tf.keras.saving.register_keras_serializable()
def normalize_board_to_perspective_tf(board, perspective_pid):

    # We want to make the neural net invariant to whose turn it is.
    # First, we get a matrix P by multiplying each perspective_id to a 20x20 board
    perspective_pid = tf.reshape(4 - perspective_pid, (-1,1))
    perspective_full = tf.reshape(perspective_pid, (-1,1,1))
    perspective_full = tf.cast(perspective_full, tf.float32)
    perspective_full = tf.tile(perspective_full, [1,20,20])
    
    # Then, we need a mask, same shape as board, that is 1 where the board == -1
    mask = tf.equal(board, -1)
    
    # Now, we can add the P matrix to the boards, and take mod 4
    perspective_full = tf.cast(perspective_full, tf.float32)
    board = tf.cast(board, tf.float32)
    board = board + perspective_full
    board = tf.cast(board, tf.int32)
    board = tf.math.mod(board, 4)
    
    # Now, to maintain -1's, we'll set the -1's back to -1
    board = tf.where(mask, -1, board)
    board = tf.reshape(board, (-1, 20, 20))
    
    board = rotate_board_to_perspective_tf(board, 0)
    
    return board

@tf.keras.saving.register_keras_serializable()
class NormalizeBoardToPerspectiveLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NormalizeBoardToPerspectiveLayer, self).__init__(**kwargs)
        
    def call(self, inputs):
        board = inputs[0]
        perspective_pid = inputs[1]
        return normalize_board_to_perspective_tf(board, perspective_pid)

class SaveModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_save_path):
        super(SaveModelCallback, self).__init__()
        self.model_save_path = model_save_path

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.model_save_path)
        convert_model_to_tflite(self.model_save_path)

def get_model(input_shape):
    
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
        
        
        #print(f"Board: {board}")
        board = tf.reshape(board, (-1, board_side_len, board_side_len, 1))
        
        # Apply convolutions
        x = keras.layers.Conv2D(32, (3,3), activation='relu')(board)
        x = keras.layers.Conv2D(64, (3,3), activation='relu')(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(32, activation='relu')(x)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(16, activation='relu')(x)
        output = keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs=inputs, outputs=output)

        model.compile(optimizer="adam",
                loss='binary_crossentropy',
                metrics=['mae'],
                #run_eagerly=True
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
            model = get_model(input_shape)
            print(model.summary())
        
        tb_log = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        #save_model_cb = SaveModelCallback(model_save_path)
        model.fit(train_ds, epochs=num_epochs, callbacks=[tb_log, early_stop], validation_data=val_ds)
    model.save(model_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with given data.')
    parser.add_argument('--data_folder', type=str, required=True,
                        help='Folder containing the data.')
    parser.add_argument('--load_model_path', type=str, help='Path to load a model from.', default=None)
    parser.add_argument('--model_save_path', type=str, required=True, help='Path to save the trained model.')
    parser.add_argument('--log_dir', type=str, required=False, help='Directory for TensorBoard logs.', default="./blockuslogs/")
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
            batch_size=args.batch_size)
    exit(0)
    
    
    