import os
import tensorflow as tf
import numpy as np
from RLFramework.read_to_dataset import read_to_dataset
from RLFramework.utils import convert_model_to_tflite
import argparse

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
        
        # Reshape the board
        board_side_len = int(np.sqrt(board.shape[1]))
        board = tf.keras.layers.Reshape((board_side_len, board_side_len, 1))(board)
        board = RandomRotateBoardLayer()(board)
        board = RandomFlipBoardLayer()(board)
        # Now we have the 20x20 board as a 3D tensor
        # Convert the board to one-hot encoded
        board = board + 1
        board = tf.one_hot(tf.cast(board,tf.int32),5)
        board = tf.reshape(board,(-1,20,20,5))

        # Lets apply convolutions
        board = tf.keras.layers.Conv2D(32, (3,3), activation='linear')(board)
        #board = tf.keras.layers.BatchNormalization()(board)
        board = tf.keras.layers.ReLU()(board)
        board = tf.keras.layers.Conv2D(64, (3,3), activation='linear')(board)
        #board = tf.keras.layers.BatchNormalization()(board)
        board = tf.keras.layers.ReLU()(board)
        #board = tf.keras.layers.Conv2D(128, (3,3), activation='linear')(board)
        #board = tf.keras.layers.PReLU(shared_axes=(1,2))(board)
        board = tf.keras.layers.Flatten()(board)
        
        # Concatenate the board and the meta
        x = tf.keras.layers.Concatenate()([meta, board])
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=output)

        model.compile(optimizer="adam",
                loss='binary_crossentropy',
                metrics=['mae']
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
        
        tb_log = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        save_model_cb = SaveModelCallback(model_save_path)
        model.fit(train_ds, epochs=num_epochs, callbacks=[tb_log, early_stop, save_model_cb], validation_data=val_ds)
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
    
    
    