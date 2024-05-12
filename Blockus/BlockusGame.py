import os
import sys
from RLFramework import GameState
from RLFramework.Player import Player
from RLFramework.Result import Result
import numpy as np
from RLFramework import Game
from typing import Dict, List, Tuple, TYPE_CHECKING
from BlockusAction import BlockusAction
from BlockusPlayer import BlockusPlayer
from BlockusResult import BlockusResult
import matplotlib.pyplot as plt
from RLFramework.utils import TFLiteModel
from BlockusPieces import BLOCKUS_PIECE_MAP

if TYPE_CHECKING:
    from BlockusGameState import BlockusGameState


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
        color_map = {-1 : "black", 0 : "red", 1 : "blue", 2 : "green", 3 : "yellow"}
        board_ax.imshow(self.board, cmap="tab20", vmin=-1, vmax=3)
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
    