import os
from RLFramework.fit_model import fit_model
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
    if model_path:
        return [TTTPlayerNeuralNet(name=f"Player{j}_{i}", logger_args=None, model_path=model_path, move_selection_temp=1) for j in range(2)]
    return [TTTPlayer(name=f"Player{j}_{i}", logger_args=None) for j in range(2)]

def model_fit(ds, epoch, num_samples):
    # Expand ds to have a channel dimension
    ds = ds.map(lambda x, y: (tf.expand_dims(x, axis=-1), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Get the input shape from the first element of the dataset
    input_shape = ds.take(1).as_numpy_iterator().next()[0].shape
    print(f"Input shape: {input_shape}")
    print(f"Num samples: {num_samples}")
    
    train_ds = ds.take(int(0.8*num_samples)).batch(128)
    val_ds = ds.skip(int(0.8*num_samples)).batch(128)
    
    train_ds = train_ds.shuffle(1000).prefetch(tf.data.AUTOTUNE)

    
    if epoch == 0:
        inputs = tf.keras.Input(shape=input_shape)
        
        #x = tf.keras.layers.BatchNormalization()(inputs)
        x = tf.keras.layers.Conv1D(8, 3, activation="relu")(inputs)
        x = tf.keras.layers.Conv1D(16, 3, activation="relu")(x)
        #x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Conv1D(32, 3, activation="relu")(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=out)
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mean_squared_error',
                metrics=['mae', 'mse']
        )
    else:
        model = tf.keras.models.load_model(f"models/model_{epoch-1}.keras")

    tb_log = tf.keras.callbacks.TensorBoard(log_dir=f"logs/fit/{epoch}", histogram_freq=1)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(train_ds, epochs=100, callbacks=[tb_log, early_stop], validation_data=val_ds)
    model.save(f"models/model_{epoch}.keras")
    return os.path.abspath(f"models/model_{epoch}.keras")

if __name__ == "__main__":
    fit_model(players_constructor,
              game_constructor,
              model_fit,
              starting_model_path="/home/ilmari/python/RLFramework/models/model_0.tflite",
              num_epochs=3,
              num_games=5000,
              num_files=-1,
              num_cpus=12,
              folder="TicTacToeModelFit",
              )


