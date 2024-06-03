import tensorflow as tf

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