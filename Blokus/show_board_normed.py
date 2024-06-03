from matplotlib import colors, pyplot as plt
import numpy as np
import tensorflow as tf
from board_norming import normalize_board_to_perspective_tf
from RLFramework.read_to_dataset import read_to_dataset

ds, _, _ = read_to_dataset("TestIndividualPlayers2")

# Plot 4 boards, and their corresponding normed boards
fig, axs = plt.subplots(2, 4, figsize=(20, 10))

fig.suptitle("The boards are standardized, so that the player whose value we are assessing\nis always playing from the top left corner with ID 0.")

color_map = colors.ListedColormap(['black', 'white', 'blue', 'red', 'green'])
color_map.set_bad(color='black')

ds = ds.shuffle(200)

for i, x in enumerate(ds.take(4)):
    perspective = int(x[0].numpy()[0])
    print(f"Perspective: {perspective}")
    # Skip the first 2 elems
    board = x[0].numpy()[2:]
    board = np.reshape(board, (20, 20))
    
    # Plot the original board
    axs[0, i].matshow(board, cmap=color_map, vmin=-1, vmax=3)
    axs[0, i].set_title("Original board")

    for board_row in range(20):
        for board_col in range(20):
            #sprint(f"Board[{board_row},{board_col}] = {board[board_row, board_col]}")
            val = int(board[board_row, board_col])
            if val == -1:
                continue
            axs[0, i].text(board_col,board_row, val, ha='center', va='center', color='orange')
    axs[0, i].set_xticks(np.arange(-0.5, 20, 1), minor=True)
    axs[0, i].set_yticks(np.arange(-0.5, 20, 1), minor=True)
    axs[0, i].grid(which='minor', color='w', linestyle='-', linewidth=2)
    axs[0, i].set_xticks([])
    axs[0, i].set_yticks([])
    
    # Plot the normed board
    board = tf.convert_to_tensor(board, dtype=tf.int32)
    board = tf.expand_dims(board, axis=0)
    board = normalize_board_to_perspective_tf(board, perspective)
    board = board.numpy().reshape((20, 20))
    axs[1, i].imshow(board, cmap=color_map, vmin=-1, vmax=3)
    axs[1, i].set_title(f"'Standardized' to perspective {perspective}")

    for board_row in range(20):
        for board_col in range(20):
            val = int(board[board_row, board_col])
            if val == -1:
                continue
            axs[1, i].text(board_col,board_row, val, ha='center', va='center', color='orange')
    axs[1, i].set_xticks(np.arange(-0.5, 20, 1), minor=True)
    axs[1, i].set_yticks(np.arange(-0.5, 20, 1), minor=True)
    axs[1, i].grid(which='minor', color='w', linestyle='-', linewidth=2)
    axs[1, i].set_xticks([])
    axs[1, i].set_yticks([])
    
    
    
plt.show()
    
    