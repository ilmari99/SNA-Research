import os
from RLFramework.fit_model import fit_model
import numpy as np
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
from BlockusGame import BlockusGame
from BlockusResult import BlockusResult
from BlockusPlayer import BlockusPlayer
from BlockusNNPlayer import BlockusNNPlayer


def game_constructor(i):
    model_paths = list(filter(lambda path: path.endswith(".tflite"), os.listdir("/home/ilmari/RLFramework/BlockusModels")))
    model_paths = [os.path.abspath(f"/home/ilmari/RLFramework/BlockusModels/{model_path}") for model_path in model_paths]
    return BlockusGame(
        board_size=(14,14),
        timeout=60,
        logger_args = None,
        render_mode = "",
        gather_data = f"gathered_data_{i}.csv",
        model_paths=model_paths,
        )

def players_constructor(i, model_path):
    if not model_path:
        return [BlockusPlayer(name=f"Player{j}_{i}", logger_args=None) for j in range(4)]
    # Get the epoch number from the model path
    model_base_path = model_path.split("/")[-1]
    epoch_num = int(model_base_path.split("_")[1].split(".")[0])
    # The previous models are in the same folder, but with different epoch numbers
    all_model_paths = [os.path.abspath(f"/home/ilmari/RLFramework/BlockusModels/model_{i}.tflite") for i in range(epoch_num + 1)]
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
    players = [BlockusNNPlayer(name=f"Player{j}_{i}",
                                    logger_args=None,
                                    model_path=np.random.choice(list(models_weighted_set.keys()), p=list(models_weighted_set.values())),
                                    move_selection_temp=1.0,
                                    )
                for j in range(4)]
    
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

def model_fit(ds, epoch, num_samples):

    # Randomly drop 1/3 of samples, where y is 1
    #ds = ds.filter(lambda x, y: tf.logical_or(tf.not_equal(y, 1), tf.random.uniform([]) < 0.66))
    
    # Get the input shape from the first element of the dataset
    input_shape = ds.take(1).as_numpy_iterator().next()[0].shape
    print(f"Input shape: {input_shape}")
    print(f"Num samples: {num_samples}")
    
    ds = ds.shuffle(2000)
    
    train_ds = ds.take(int(0.7*num_samples)).batch(64)
    val_ds = ds.skip(int(0.7*num_samples)).batch(64)
    
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
    else:
        model = tf.keras.models.load_model(f"/home/ilmari/RLFramework/BlockusModels/model_{epoch-1}.keras")

    tb_log = tf.keras.callbacks.TensorBoard(log_dir=f"logs/fit/{epoch}", histogram_freq=1)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(train_ds, epochs=25, callbacks=[tb_log, early_stop], validation_data=val_ds)
    model.save(f"BlockusModels/model_{epoch}.keras")
    tf.keras.backend.clear_session()
    return os.path.abspath(f"/home/ilmari/RLFramework/BlockusModels/model_{epoch}.keras")

if __name__ == "__main__":
    fit_model(players_constructor,
              game_constructor,
              model_fit,
              starting_model_path="",
              num_epochs=20,
              num_games=500,
              num_files=-1,
              num_cpus=10,
              folder="BlockusModelFit",
              starting_epoch=0,
              )
