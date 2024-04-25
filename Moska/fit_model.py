import os
from RLFramework.fit_model import fit_model
import numpy as np

import tensorflow as tf
from MoskaGame import MoskaGame
from MoskaResult import MoskaResult
from MoskaPlayer import MoskaPlayer
from MoskaNNPlayer import MoskaNNPlayer
import argparse

def game_constructor(i, model_base_folder):
    model_paths = list(filter(lambda path: path.endswith(".tflite"), os.listdir(model_base_folder)))
    model_paths = [os.path.abspath(os.path.join(model_base_folder,model_path)) for model_path in model_paths]
    return MoskaGame(
        timeout=15,
        logger_args = None,
        render_mode = "",
        gather_data = f"gathered_data_{i}.csv",
        model_paths=model_paths,
    )

def players_constructor(i, model_path, model_base_folder):
    if not model_path:
        return [MoskaPlayer(name=f"Player{j}_{i}", logger_args=None) for j in range(4)]
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
    players = [MoskaNNPlayer(name=f"Player{j}_{i}",
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

def get_conv_model(input_shape):
        
        inputs = tf.keras.Input(shape=input_shape)
        
        # First separate the input into misc and card data:
        # The first 15 values are miscellanous
        misc = tf.gather(inputs, [i for i in range(15)], axis=1)
        # The rest are card data
        cards = tf.gather(inputs, [i for i in range(15, input_shape[0])], axis=1)
        # We then reshape the card data to 8x52x1
        # 8 players, 52 cards (1 means the card is in the set, 0 means it is not), 1 channel
        cards = tf.keras.layers.Reshape((8,52,1))(cards)
        # And apply convolutional layers
        x = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(cards)
        x = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(x)
        x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        # Concatenate the misc data to the convolutional layers
        x = tf.keras.layers.Concatenate()([x, misc])
        # And apply dense layers
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', "mae"])
        return model

def get_mlp_model(input_shape):
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

def model_fit(train_ds, val_ds, epoch, num_samples, model_base_folder, model_type="mlp"):

    # Randomly drop 1/3 of samples, where y is 1
    #ds = ds.filter(lambda x, y: tf.logical_or(tf.not_equal(y, 1), tf.random.uniform([]) < 0.66))
    
    # Get the input shape from the first element of the dataset
    input_shape = train_ds.take(1).as_numpy_iterator().next()[0].shape
    print(f"Input shape: {input_shape}")
    print(f"Num samples: {num_samples}")
    
    train_ds = train_ds.batch(4096)
    val_ds = val_ds.batch(4096)
    
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    if epoch == 0:
        if model_type == "mlp":
            model = get_mlp_model(input_shape)
        else:
            model = get_conv_model(input_shape)
        print(model.summary())
        
    else:
        model = tf.keras.models.load_model(os.path.join(model_base_folder,f"model_{epoch-1}.keras"))

    tb_log = tf.keras.callbacks.TensorBoard(log_dir=f"logs/fit/{epoch}", histogram_freq=1)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    # balance the classes using the class_weight parameter: 75% of the samples are 1, 25% are 0
    model.fit(train_ds, epochs=25, callbacks=[tb_log, early_stop], validation_data=val_ds, class_weight={0: 0.75, 1: 0.25})
    model.save(os.path.join(model_base_folder,f"model_{epoch}.keras"))
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
    parser.add_argument("--validation_frac", type=float, default=0.2,
                        help="Fraction of data to use for validation.")
    parser.add_argument("--model_type", type=str, default="mlp",
                        help="Type of model to use (mlp or conv).")

    # Path arguments
    parser.add_argument("--starting_epoch", type=int, default=0,
                        help="Epoch to start training from.")
    parser.add_argument("--model_folder_base", type=str, default=os.path.abspath("./MoskaModels/"),
                        help="Base path for model storage.")
    parser.add_argument("--data_folder_base", type=str, default=os.path.abspath("./MoskaModelFit/"),
                        help="Base path for training data.")
    parser.add_argument("--starting_model_path", type=str, default="",
                        help="Path to the starting model (optional).")
    parser.add_argument("--cumulate_data", action="store_true",
                        help="Whether to keep old datasets or not.", default=False)
    parser.add_argument(f"--delete_data_after_fit", action="store_true", default=False)

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()

    print(args)
    
    os.makedirs(args.model_folder_base, exist_ok=True)
    os.makedirs(args.data_folder_base, exist_ok=True)
    
    model_folder_base = os.path.abspath(args.model_folder_base)

    def players_constructor_(i, model_path):
        return players_constructor(i,model_path,model_folder_base)

    def game_constructor_(i):
        return game_constructor(i, model_folder_base)
    
    def model_fit_(train_ds, val_ds, epoch, num_samples):
        return model_fit(train_ds, val_ds, epoch, num_samples,model_folder_base, model_type=args.model_type)

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
              validation_frac=args.validation_frac,
              cumulate_data=args.cumulate_data,
              delete_data_after_fit=args.delete_data_after_fit
    )
