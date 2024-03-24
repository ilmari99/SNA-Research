import os
import numpy as np

import tensorflow as tf


from RLFramework.fit_model import fit_model
from PFGame import PFGame
from PFPlayer import PFPlayer
from PFNeuralNetworkPlayer import PFNeuralNetworkPlayer
from PFResult import PFResult

def game_constructor(i):
    return PFGame(board_size=(7,7),
                   logger_args = None,
                   render_mode = "",
                   gather_data = f"gathered_data_{i}.csv",
                   custom_result_class = PFResult,
                   max_num_total_steps = 200,
    )

def players_constructor(i, model_path):
    """ If no model path, use random player.
    """
    if not model_path:
        return [PFPlayer(f"RandomPlayer{i}", logger_args=None)]
    return [PFNeuralNetworkPlayer(f"NeuralNetPlayer{i}",
                                  model_path,
                                  move_selection_temp=1.0,
                                  logger_args=None)]
    
@tf.keras.saving.register_keras_serializable()
class RandomRotateBoardLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RandomRotateBoardLayer, self).__init__(**kwargs)

    def call(self, board, training=None):
        # Randomly rotate the board only during training
        if training:
            board = tf.image.rot90(board, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
        return board

def model_fit(ds, epoch, num_samples):
    """ Fit a neural network model to the dataset.
    If epoch == 0, create a new model. Otherwise, load the model from the previous epoch.
    """
    input_shape = ds.take(1).as_numpy_iterator().next()[0].shape
    print(f"Input shape: {input_shape}")
    print(f"Num samples: {num_samples}")
    
    ds = ds.shuffle(10000)
    
    train_ds = ds.take(int(0.9*num_samples)).batch(128)
    val_ds = ds.skip(int(0.1*num_samples)).batch(128)
    
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    
    if epoch == 0:
        inputs = tf.keras.Input(shape=input_shape)
        
        # Board is everything except the first element
        board = inputs[:,1:]
        meta = inputs[:,0]
        
        meta = tf.keras.layers.Flatten()(meta)
        
        # Reshape the board
        board_side_len = int(np.sqrt(board.shape[1]))
        board = tf.keras.layers.Reshape((board_side_len, board_side_len, 1))(board)
        
        # Randomly rotate the board
        board = RandomRotateBoardLayer()(board)
        
        # Convolutional layers
        x = tf.keras.layers.Conv2D(16, 3, activation='relu')(board)
        x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        
        # Concatenate the board and the meta data
        x = tf.keras.layers.Concatenate()([meta, x])
        x = tf.keras.layers.Dense(10, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='linear')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
    else:
        p = os.path.abspath(f"models/model_{epoch-1}.keras")
        print(f"Loading model: {p}")
        model = tf.keras.models.load_model(p)
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
    tb_log = tf.keras.callbacks.TensorBoard(log_dir=f"logs/fit/{epoch}", histogram_freq=1)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(train_ds, epochs=50, callbacks=[tb_log, early_stop], validation_data=val_ds)
    model.save(f"models/model_{epoch}.keras")
    tf.keras.backend.clear_session()
    return os.path.abspath(f"models/model_{epoch}.keras")

if __name__ == "__main__":
    fit_model(players_constructor,
              game_constructor,
              model_fit,
              starting_model_path="",
              num_epochs=10,
              num_games=1000,
              num_files=-1,
              num_cpus=12,
              folder="PFModelFitDataset",
              )


