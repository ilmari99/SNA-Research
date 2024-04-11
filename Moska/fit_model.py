import os
from RLFramework.fit_model import fit_model
import numpy as np

import tensorflow as tf
from MoskaGame import MoskaGame
from MoskaResult import MoskaResult
from MoskaPlayer import MoskaPlayer
from MoskaNNPlayer import MoskaNNPlayer




def game_constructor(i):
    return MoskaGame(
        timeout=5,
        logger_args = None,
        render_mode = "",
        gather_data = f"gathered_data_{i}.csv",
        #custom_result_class = MoskaResult,
    )

def players_constructor(i, model_path):
    if not model_path:
        return [MoskaPlayer(name=f"Player{j}_{i}", logger_args=None) for j in range(4)]
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

def model_fit(ds, epoch, num_samples):
    
    # Get the input shape from the first element of the dataset
    input_shape = ds.take(1).as_numpy_iterator().next()[0].shape
    print(f"Input shape: {input_shape}")
    print(f"Num samples: {num_samples}")
    
    ds = ds.shuffle(10000)
    
    train_ds = ds.take(int(0.9*num_samples)).batch(1024)
    val_ds = ds.skip(int(0.1*num_samples)).batch(1024)
    
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    
    if epoch == 0:
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Dense(400, activation='relu')(inputs)
        x = tf.keras.layers.Dense(400, activation='relu')(x)
        x = tf.keras.layers.Dense(400, activation='relu')(x)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=output)

        model.compile(optimizer="adam",
                loss='binary_crossentropy',
                metrics=['mae', "accuracy"]
        )
        print(model.summary())
    else:
        model = tf.keras.models.load_model(f"models/model_{epoch-1}.keras")

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
              num_games=500,
              num_files=-1,
              num_cpus=12,
              folder="MoskaModelFit",
              )
