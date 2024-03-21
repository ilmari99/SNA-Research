import os
from RLFramework.fit_model import fit_model
import numpy as np
from TTTGame import TTTGame
from TTTPlayerNeuralNet import TTTPlayerNeuralNet
from TTTResult import TTTResult
import tensorflow as tf

from TTTPlayer import TTTPlayer


def game_constructor(i):
    return TTTGame(board_size=(3,3),
                            logger_args = None,
                            render_mode = "",
                            gather_data = f"gathered_data_{i}.csv",
                            custom_result_class = TTTResult,
                            )

def players_constructor(i, model_path):
    if not model_path:
        return [TTTPlayer(name=f"Player{j}_{i}", logger_args=None) for j in range(2)]
    # Get the epoch number from the model path
    model_base_path = model_path.split("/")[-1]
    epoch_num = int(model_base_path.split("_")[1].split(".")[0])
    # The previous models are in the same folder, but with different epoch numbers
    all_model_paths = [os.path.abspath(f"../../models/model_{i}.tflite") for i in range(epoch_num + 1)]
    #print(all_model_paths)
    # In the simulation, we play games with the current and previous models
    # To do that, we'll create a dict of players, where the keys are the model paths, and the values are the weights
    # for picking that player. The weight is the epoch number.
    models_weighted_set = {model_path_ : epoch_num_ + 1 for model_path_, epoch_num_ in zip(all_model_paths, range(epoch_num+1))}
    # Softmax the weights
    model_weights = np.array(list(models_weighted_set.values()))
    model_weights = np.exp(model_weights) / np.sum(np.exp(model_weights))
    
    models_weighted_set = {model_path_ : w for model_path_, w in zip(all_model_paths, model_weights)}
    #print(models_weighted_set)
    players = [TTTPlayerNeuralNet(name=f"Player{j}_{i}",
                                    logger_args=None,
                                    model_path=np.random.choice(list(models_weighted_set.keys()), p=list(models_weighted_set.values())),
                                    move_selection_temp=0.99,
                                    )
                for j in range(2)]
    
    return players

def count_num_samples_in_ds(ds):
    """ Count how many y values of 0, 0.5, 1 there are in the dataset.
    """
    num_samples = {0 : 0, 0.5 : 0, 1 : 0}
    for x, y in ds:
        num_samples[y.numpy()] += 1
    return num_samples

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
    
    # Get the input shape from the first element of the dataset
    input_shape = ds.take(1).as_numpy_iterator().next()[0].shape
    print(f"Input shape: {input_shape}")
    print(f"Num samples: {num_samples}")
    
    ds = ds.shuffle(10000)
    
    train_ds = ds.take(int(0.9*num_samples)).batch(128)
    val_ds = ds.skip(int(0.1*num_samples)).batch(128)
    
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    
    if epoch == 0:
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
        # Apply 1x2 convolution, and then 2x1 convolution
        board = tf.keras.layers.Conv2D(16, (1,2), activation='relu')(board)
        board = tf.keras.layers.Conv2D(32, (2,1), activation='relu')(board)
        board = tf.keras.layers.Flatten()(board)
        
        # Concatenate the board and the meta
        x = tf.keras.layers.Concatenate()([meta, board])
        x = tf.keras.layers.Dense(16, activation='relu')(x)
        x = tf.keras.layers.Dense(8, activation='relu')(x)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=output)

        model.compile(optimizer="adam",
                loss='mse',
                metrics=['mae', 'mse', "accuracy"]
        )
        print(model.summary())
    else:
        model = tf.keras.models.load_model(f"models/model_{epoch-1}.keras")

    tb_log = tf.keras.callbacks.TensorBoard(log_dir=f"logs/fit/{epoch}", histogram_freq=1)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(train_ds, epochs=30, callbacks=[tb_log, early_stop], validation_data=val_ds)
    model.save(f"models/model_{epoch}.keras")
    tf.keras.backend.clear_session()
    return os.path.abspath(f"models/model_{epoch}.keras")

if __name__ == "__main__":
    fit_model(players_constructor,
              game_constructor,
              model_fit,
              starting_model_path="",
              num_epochs=10,
              num_games=5000,
              num_files=-1,
              num_cpus=12,
              folder="TicTacToeModelFitDataset",
              )


