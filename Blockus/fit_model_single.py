import os
import tensorflow as tf
import numpy as np
from RLFramework.read_to_dataset import read_to_dataset

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
        # Now we have the Blokus board, which is 14x14
        # Lets apply a 3x3 convolution, and then 2x2 convolution
        board = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(board)
        board = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(board)
        board = tf.keras.layers.Conv2D(128, (3,3), activation='relu')(board)
        board = tf.keras.layers.Flatten()(board)
        
        # Concatenate the board and the meta
        x = tf.keras.layers.Concatenate()([meta, board])
        x = tf.keras.layers.Dense(8, activation='relu')(x)
        x = tf.keras.layers.Dense(8, activation='relu')(x)
        output = tf.keras.layers.Dense(1, activation='relu')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=output)

        model.compile(optimizer="adam",
                loss='mse',
                metrics=['mae']
        )
        print(model.summary())
        return model

if __name__ == "__main__":
    # All fodlers in BlockusModelFit
    data_folders = os.listdir("/home/ilmari/python/RLFramework/BlockusModelFitV1/")
    # Filter only folders
    data_folders = [os.path.join("/home/ilmari/python/RLFramework/BlockusModelFitV1/", folder) for folder in data_folders if os.path.isdir(os.path.join("/home/ilmari/python/RLFramework/BlockusModelFit/", folder))]
    print(data_folders)
    
    ds, num_files, approx_num_samples = read_to_dataset(data_folders)
    
    input_shape = ds.take(1).as_numpy_iterator().next()[0].shape
    print(f"Input shape: {input_shape}")
    print(f"Num samples: {approx_num_samples}")
    
    ds = ds.shuffle(2000)
    
    train_ds = ds.take(int(0.7*approx_num_samples)).batch(128)
    val_ds = ds.skip(int(0.7*approx_num_samples)).batch(128)
    
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    
    model = get_model(input_shape)
    print(model.summary())
    
    tb_log = tf.keras.callbacks.TensorBoard(log_dir=f"logs/fit", histogram_freq=1)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(train_ds, epochs=25, callbacks=[tb_log, early_stop], validation_data=val_ds)
    model.save(f"BlockusModels/model_all_data.keras")
    
    
    