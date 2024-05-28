import os
import sys
from RLFramework import GameState
from RLFramework.Player import Player
from RLFramework.Result import Result
import numpy as np
from RLFramework import Game
from typing import Dict, List, Tuple, TYPE_CHECKING

import tensorflow as tf
from BlockusAction import BlockusAction
from BlockusPlayer import BlockusPlayer
from BlockusResult import BlockusResult
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from RLFramework.utils import TFLiteModel
from BlockusPieces import BLOCKUS_PIECE_MAP

if TYPE_CHECKING:
    from BlockusGameState import BlockusGameState

def rotate_board_to_perspective(board, perspective_pid):
    """ Rotate the board to the perspective of perspective_pid.
    """
    # First, we find the corner that has the perspective_pid
    top_left_pid = board[0,0]
    top_right_pid = board[0,-1]
    bottom_right_pid = board[-1,-1]
    bottom_left_pid = board[-1,0]
    corner_pids = [top_left_pid, top_right_pid, bottom_right_pid, bottom_left_pid]
    #print(f"Corner pids: {corner_pids}")
    
    # Find the index of the corner that has the perspective_pid
    if perspective_pid not in corner_pids:
        corner_index = 0
        print(f"Perspective pid {perspective_pid} not found in the corners: {corner_pids}")
    else:
        corner_index = corner_pids.index(perspective_pid)
    
    # Rotate the board to make the corner with the perspective_pid the top left corner
    board = np.expand_dims(board, axis=-1)
    board = np.rot90(board, k=corner_index)
    board = np.squeeze(board, axis=-1)
    #print(board)
    return board

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

def rotate_board_to_perspective_tf(board, perspective_pid):
    
    top_left_pids = tf.reshape(board[0,0], (-1,1))
    top_right_pids = tf.reshape(board[0,-1], (-1,1))
    bottom_right_pids = tf.reshape(board[-1,-1], (-1,1))
    bottom_left_pids = tf.reshape(board[-1,0], (-1,1))
    
    perspective_pid_curr_corner = tf.concat([top_left_pids, top_right_pids, bottom_right_pids, bottom_left_pids], axis=1)
    perspective_pid_curr_corner = tf.cast(perspective_pid_curr_corner, tf.int32)
    
    perspective_pids = tf.reshape(perspective_pid, (-1,1))
    
    mask = tf.equal(perspective_pid_curr_corner, perspective_pids)
    mask = tf.cast(mask, tf.float32)
    print(f"Mask: {mask}")
    
    # Take argmax to get the corner that has the perspective_pid: B x 1
    # This tells us how many counter clockwise rotations to do for each board in the batch
    number_of_rotations = tf.argmax(mask, axis=1)
    print(f"Number of rotations: {number_of_rotations}")

    #print(f"Number of rotations: {number_of_rotations}")
    # float and expand for concat with board and image rot90
    number_of_rotations = tf.cast(number_of_rotations, tf.float32)
    number_of_rotations = tf.reshape(number_of_rotations, (-1,1))
    
    board = tf.reshape(board, (-1, 20*20))
    board = tf.cast(board, tf.float32)
    number_of_rotations = -1*tf.cast(number_of_rotations, tf.float32)
    board_rot_pairs = tf.concat([board, number_of_rotations], axis=1)
    
    board = RotLayer()(board_rot_pairs)
    board = tf.reshape(board, (-1, 20, 20))
    board = tf.cast(board, tf.int32)
    
    return board

def normalize_board_to_perspective_tf(board, perspective_pid):
        
    board = rotate_board_to_perspective_tf(board, perspective_pid)
    
    # We want to make the neural net invariant to whose turn it is.
    # First, we get a matrix P by multiplying each perspective_id to a 20x20 board
    perspective_pids = tf.reshape(perspective_pid, (-1,1))
    perspective_pids_repeated = tf.reshape(perspective_pids, (-1,1,1))
    perspective_pids_repeated = tf.cast(perspective_pids_repeated, tf.float32)
    perspective_pids_repeated = tf.tile(perspective_pids_repeated, [1,20,20])
    perspective_pids_repeated = tf.cast(perspective_pids_repeated, tf.int32)
    
    # Then, we need a mask, same shape as board, that is -1 where the board == -1
    mask = tf.equal(board, -1)
    mask = tf.cast(mask, tf.float32)
    #mask = -1 * mask
    
    # Now, we can add the P matrix to the boards, and take mod 4
    board = board + perspective_pids_repeated
    board = tf.math.mod(board, 4)
    board = tf.cast(board, tf.float32)
    
    # Now, to maintain -1's, we'll set the -1's back to -1
    # We want to do a similar operation as "board = where(mask == -1, -1, board)",
    # but we can't use tf.where.
    board = tf.where(mask == 1.0, -1, board)

    return np.array(board,dtype=np.int32).squeeze()

    
    

def normalize_board_to_perspective(board, perspective_pid):
    """ Given a board, modify the so that the perspective_pid is always 0.
    """
    board = rotate_board_to_perspective(board, perspective_pid)
    print(f"Perspective pid: {perspective_pid}")
    # Broadcast the perspective_pid to the shape of the board
    perspective_full = np.full(board.shape, perspective_pid)
    # Get a mask that describes where the board == -1
    mask = board == -1
    
    # Now, we can add the perspective pid to each element
    # of the board and take mod 3
    # This makes the perspective_pid 0, the next player will be 1, and the next 2 ...
    board = board + perspective_full
    board = np.mod(board, 4)
    
    # In the add and mod we lose the -1's, so we need to set them back
    board = np.where(mask == 1, -1, board)
    
    return board

class BlockusGame(Game):
    """ The game class handles the play loop.
    """
    def __init__(self, board_size : Tuple[int, int] = (20, 20), model_paths=[], **kwargs):
        # Hack
        from BlockusGameState import BlockusGameState
        super().__init__(BlockusGameState, custom_result_class=BlockusResult, **kwargs)
        self.board_size = board_size
        self.model_paths = []
        self.board = None
        self.set_models(model_paths)
        
    def play_game(self, players: List[Player]) -> Result:
        # Load the models before playing the game
        current_models = set(self.model_paths)
        model_paths = set([os.path.abspath(p.model_path) for p in players if (hasattr(p, "model_path") and p.model_path is not None)])
        # If there are any new models, load them
        if model_paths - current_models:
            self.set_models(list(model_paths))
        out = super().play_game(players)
        if self.render_mode == "human":
            plt.savefig("blockus.png")
        return out
    
    def get_current_state(self, player: Player = None) -> 'BlockusGameState':
        return super().get_current_state(player)
    
    def __repr__(self) -> str:
        s = ""
        for i, row in enumerate(self.board):
            s += " ".join([str(x).ljust(2) for x in row]) + "\n"
        return s

    def get_model(self, model_name : str) -> TFLiteModel:
        """ Get the model with the given name.
        """
        model_name = os.path.abspath(model_name)
        try:
            return self.models[model_name]
        except KeyError:
            raise ValueError(f"Model with name {model_name} not found. Available models: {list(self.models.keys())}")
        
    def set_models(self, model_paths : List[str]) -> None:
        """ Set the models to the given paths.
        """
        self.model_paths = model_paths
        self.models = {path: TFLiteModel(path) for path in model_paths}
    
    def initialize_game(self, players: List[BlockusPlayer]) -> None:
        """ When the game is started, we need to set the board.
        """
        self.board = [[-1 for _ in range(self.board_size[1])] for _ in range(self.board_size[0])]
        #self.board = np.array(self.board)
        self.current_pid = 0
        self.player_remaining_pieces = [list(range(21)) for _ in players]
        self.finished_players = []
    
    def init_render_human(self) -> None:
        plt.cla()
        plt.clf()
        plt.close()
        self.fig, self.ax = plt.subplots(1, 2)
        plt.ion()
        plt.show()
        
    def render_human(self, ax: plt.Axes = None) -> None:
        """ Render the game.
        """
        board_ax : plt.Axes = self.ax[0]
        pieces_ax : plt.Axes = self.ax[1]
        board_ax.clear()
        
        # Always color the board using the same colors.
        # -1: black, 0: white, 1: blue, 2: red, 3: green
        color_map = colors.ListedColormap(['black', 'white', 'blue', 'red', 'green'])
        color_map.set_bad(color='black')
        
        board_normed = normalize_board_to_perspective_tf(np.array(self.board), self.current_pid)
        board_ax.matshow(board_normed, cmap=color_map, vmin=-1, vmax=3)
        # Label the grids
        for i in range(self.board_size[0]):
            for j in range(self.board_size[1]):
                board_ax.text(j,i, board_normed[i, j], ha='center', va='center', color='orange')
        board_ax.set_xticks(np.arange(-0.5, self.board_size[1], 1), minor=True)
        board_ax.set_yticks(np.arange(-0.5, self.board_size[0], 1), minor=True)
        board_ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
        board_ax.set_title(f"Scores: {self.player_scores}")
        board_ax.set_xticks([])
        board_ax.set_yticks([])

        # Add section to show remaining pieces for current player
        remaining_pieces = self.player_remaining_pieces[self.current_pid]
        remaining_pieces = [BLOCKUS_PIECE_MAP[piece] for piece in remaining_pieces]
        
        pieces_ax.clear()
        # Place all the pieces in the pieces_ax
        # We place the pieces on 20x20 grid, where each piece is placed in a 4x4 grid
                
        remaining_pieces_board = np.zeros_like(self.board) - 1
        row = 0
        col = 0
        for i, piece in enumerate(remaining_pieces):
            piece = np.where(piece == 0, -1, self.current_pid)
            remaining_pieces_board[row:row+piece.shape[0], col:col+piece.shape[1]] = piece
            col += 4
            if col >= 20:
                col = 0
                row += 4
        
        pieces_ax.imshow(remaining_pieces_board, cmap="tab20", vmin=-1, vmax=3)
        pieces_ax.set_xticks(np.arange(-0.5, 20, 1), minor=True)
        pieces_ax.set_yticks(np.arange(-0.5, 20, 1), minor=True)
        pieces_ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
        pieces_ax.set_title(f"Player {self.current_pid} remaining pieces")
        pieces_ax.set_xticks([])
        pieces_ax.set_yticks([])
        # Add a grid dividing the pieces in to 4x4 grids by addding black lines
        for i in range(5):
            pieces_ax.axhline(i * 4 - 0.5, color='black', linewidth=2)
            pieces_ax.axvline(i * 4 - 0.5, color='black', linewidth=2)


        plt.pause(0.01)

    def restore_game(self, game_state: 'BlockusGameState') -> None:
        """ Restore the game to the state described by the game_state.
        We don't need to worry about the players states or their scores, as they are automatically restored.
        """
        self.board = game_state.board
        self.player_remaining_pieces = game_state.player_remaining_pieces
        self.current_pid = game_state.current_pid
        self.previous_turns = game_state.previous_turns
        self.finished_players = game_state.finished_players
    
    def calculate_reward(self, pid : int, game_state: 'BlockusGameState') -> float:
        """ Calculate the reward for the player.
        If the player wins, the reward is 1.0.
        If the game is a draw, the reward is 0.5
        """
        return self.get_current_state().calculate_reward(pid)
    
    def environment_action(self, game_state : 'BlockusGameState') -> 'BlockusGameState':
        self.update_finished_players_in_gamestate(game_state)
        self.update_player_scores_in_gamestate(game_state)
        self.update_player_attributes()
        return super().environment_action(game_state)
    
    def check_is_player_finished(self, pid : int, game_state: GameState) -> bool:
        """ A player is finished if the game is finished.
        I.e. if the player has won, or if there are no more free spots, or if the other player is finished.
        """
        if pid in game_state.finished_players:
            return True
        return False
    
    def is_grid_inside_board(self, grid, board_size):
        """ Check if the grid is inside the board.
        """
        return self.get_current_state().is_grid_inside_board(grid, board_size)
    
    def get_corner_positions(self, pid : int) -> Tuple[List[Tuple[int, int]],
                                                       List[Tuple[int, int]]]:
        """ Return two lists:
        - The first list contains the grids, which have a player's piece, and that
        are a corner of the occupied area.
        - The second list contains the grids, that are free, that share a corner
        with atleast one of the grids in the first list, and that do not have any
        common sides with the player's pieces.
        """
        return self.get_current_state().get_corner_positions(pid)
    
    def find_num_connected_pieces(self, board, pid, x, y):
        """ Find the number of connected pieces, starting from the grid (x, y).
        """
        return self.get_current_state().find_num_connected_pieces(board, pid, x, y)
        
    
    def get_all_possible_actions(self) -> List[BlockusAction]:
        """ Return all possible actions.
        """
        return self.get_current_state().get_all_possible_actions()
    