import os
from RLFramework.fit_model import fit_model
import numpy as np
import argparse
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
from BlockusGame import BlockusGame
from BlockusResult import BlockusResult
from BlockusPlayer import BlockusPlayer
from BlockusNNPlayer import BlockusNNPlayer


def game_constructor(i, model_base_folder):
    model_paths = list(filter(lambda path: path.endswith(".tflite"), os.listdir(model_base_folder)))
    model_paths = [os.path.abspath(os.path.join(model_base_folder,model_path)) for model_path in model_paths]
    return BlockusGame(
        board_size=(20,20),
        timeout=60,
        logger_args = None,
        render_mode = "",
        gather_data = f"gathered_data_{i}.csv",
        model_paths=model_paths,
        )

def players_constructor(i, model_path, model_base_folder):
    if not model_path:
        return [BlockusPlayer(name=f"Player{j}_{i}", logger_args=None) for j in range(4)]
    # Get the epoch number from the model path
    model_base_path = model_path.split("/")[-1]
    epoch_num = int(model_base_path.split("_")[1].split(".")[0])
    # The previous models are in the same folder, but with different epoch numbers
    all_model_paths = [os.path.abspath(os.path.join(model_base_folder,f"model_{i}.tflite")) for i in range(epoch_num + 1)]
    # Filter non-existent
    for path in all_model_paths.copy():
        if not os.path.exists(path):
            all_model_paths.remove(path)
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

def model_fit(ds, epoch, num_samples, model_base_folder):

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
        model = tf.keras.models.load_model(os.path.abspath(os.path.join(model_base_folder,f"model_{epoch-1}.keras")))

    tb_log = tf.keras.callbacks.TensorBoard(log_dir=f"logs/fit/{epoch}", histogram_freq=1)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(train_ds, epochs=25, callbacks=[tb_log, early_stop], validation_data=val_ds)
    model.save(os.path.abspath(os.path.join(model_base_folder,f"model_{epoch}.keras")))
    tf.keras.backend.clear_session()
    return os.path.abspath(os.path.join(model_base_folder,f"model_{epoch}.keras"))

def parse_arguments():
    """Parses command-line arguments for training script.

    Returns:
    A namespace containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a Moska model.")

    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs.")
    parser.add_argument("--num_games", type=int, default=100,
                        help="Number of games to play per epoch.")
    parser.add_argument("--num_files", type=int, default=-1,
                        help="Number of data files to use (-1 for all).")
    parser.add_argument("--num_cpus", type=int, default=10,
                        help="Number of CPUs to use for training.")

    # Path arguments
    parser.add_argument("--starting_epoch", type=int, default=0,
                        help="Epoch to start training from.")
    parser.add_argument("--model_folder_base", type=str, default=os.path.abspath("./BlockusModels/"),
                        help="Base path for model storage.")
    parser.add_argument("--data_folder_base", type=str, default=os.path.abspath("./BlockusModelFit/"),
                        help="Base path for training data.")
    parser.add_argument("--starting_model_path", type=str, default="",
                        help="Path to the starting model (optional).")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    print(args)

    def players_constructor_(i, model_path):
        return players_constructor(i,model_path,args.model_folder_base)

    def game_constructor_(i):
        return game_constructor(i, args.model_folder_base)
    
    def model_fit_(ds, epoch, num_samples):
        return model_fit(ds, epoch, num_samples,args.model_folder_base)
    
    fit_model(players_constructor_,
              game_constructor_,
              model_fit_,
              starting_model_path=args.starting_model_path,
              num_epochs=args.num_epochs,
              num_games=args.num_games,
              num_files=args.num_files,
              num_cpus=args.num_cpus,
              folder=args.data_folder_base,
              starting_epoch=args.starting_epoch,
              )
