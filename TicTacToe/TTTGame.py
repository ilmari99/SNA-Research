import os
import sys
from RLFramework import GameState
from RLFramework.Player import Player
from RLFramework.Result import Result
import numpy as np
from RLFramework import Game
from typing import Dict, List, Tuple
from TTTAction import TTTAction
from TTTGameState import TTTGameState
from TTTPlayer import TTTPlayer
import matplotlib.pyplot as plt
from RLFramework.utils import TFLiteModel


class TTTGame(Game):
    """ A class representing the game TicTacToe.
    """
    def __init__(self, board_size : Tuple[int, int] = (3, 3), **kwargs):
        super().__init__(TTTGameState, **kwargs)
        self.board_size = board_size
        self.model_paths = []
        
    def play_game(self, players: List[Player]) -> Result:
        # Load the models before playing the game
        current_models = set(self.model_paths)
        model_paths = set([p.model_path for p in players if (hasattr(p, "model_path") and p.model_path is not None)])
        # If there are any new models, load them
        if model_paths - current_models:
            self.set_models(list(model_paths))
        return super().play_game(players)
        
    
    def get_model(self, model_name : str) -> TFLiteModel:
        """ Get the model with the given name.
        """
        model_name = os.path.abspath(model_name)
        try:
            return self.models[model_name]
        except KeyError:
            raise ValueError(f"Model with name {model_name} not found. Available models: {list(self.models.keys())}")
        
    def set_models(self, model_paths : List[str], convolutionals : List[bool] = []) -> None:
        """ Set the models to the given paths.
        """
        self.model_paths = model_paths
        self.models = {path: TFLiteModel(path) for path in model_paths}
    
    
    def initialize_game(self, players: List[TTTPlayer]) -> None:
        """ When the game is started, we need to set the board.
        """
        self.board = [[-1 for _ in range(self.board_size[1])] for _ in range(self.board_size[0])]

    def restore_game(self, game_state: TTTGameState) -> None:
        """ Restore the game to the state described by the game_state.
        We don't need to worry about the players states or their scores, as they are automatically restored.
        """
        self.board = game_state.board
    
    def select_turn(self, players: List[TTTPlayer], previous_turns: List[int]) -> int:
        return super()._select_round_turn(players, previous_turns)
    
    def calculate_reward(self, pid : int, game_state: TTTGameState) -> float:
        """ Calculate the reward for the player.
        If the player wins, the reward is 1.0.
        If the game is a draw, the reward is 0.5
        """
        
        if self.check_player_has_won(pid, game_state):
            return 1.0
        # If game is a draw
        if -1 not in np.array(game_state.board).flatten() and not any([self.check_player_has_won(p, game_state) for p in range(len(self.players)) if p != pid]):
            return 0.5
        return 0.0
    
    def check_player_has_won(self, pid : int, game_state: GameState) -> bool:
        """ Check if a player has a full row/col/diag
        """
        arr = np.array(game_state.board)
        # Check rows and columns
        for row_idx in range(arr.shape[0]):
            if np.all(arr[row_idx] == pid):
                self.logger.info(f"{self.players[pid]} has won with a row!")
                return True
        for col_idx in range(arr.shape[1]):
            if np.all(arr[:, col_idx] == pid):
                self.logger.info(f"{self.players[pid]} has won with a column!")
                return True
        if np.all(np.diag(arr) == pid) or np.all(np.diag(np.fliplr(arr)) == pid):
            self.logger.info(f"{self.players[pid]} has won with a diagonal!")
            return True
        return False
        
    def check_is_player_finished(self, pid : int, game_state: GameState) -> bool:
        """ A player is finished if the game is finished.
        I.e. if the player has won, or if there are no more free spots, or if the other player is finished.
        """
        has_straight_line = self.check_player_has_won(pid, game_state)
        if has_straight_line:
            return True
        if -1 not in np.array(game_state.board).flatten():
            return True
        if any([self.check_player_has_won(p, game_state) for p in range(len(self.players)) if p != pid]):
            return True
        return False
    
    def get_all_possible_actions(self) -> List[TTTAction]:
        """ Return all possible actions.
        """
        actions = []
        for x in range(self.board_size[0]):
            for y in range(self.board_size[1]):
                if TTTAction.check_action_is_legal_from_args(self, x, y):
                    actions.append(TTTAction(x, y))
        return actions
    
    def render_human(self : Game, ax):
        """ Render the board to the ax and highlight the current player.
        """
        arr = np.array(self.board)
        curr_player_pid = self.current_player
        # Show the board, with all white squares and borders
        ax.imshow(arr)
        ax.set_xticks(np.arange(arr.shape[1]) - 0.5, minor=True)
        ax.set_yticks(np.arange(arr.shape[0]) - 0.5, minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
        

        # Annotate each square with the pid of the player who has it, or an empty string if it is empty
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                square_pid = arr[i, j]
                if square_pid != -1:
                    ax.text(j, i, str(square_pid), ha="center", va="center", fontsize=20)
        # Highlight the current player
        ax.set_title(f"{self.players[curr_player_pid].name}'s turn")
        
        if self.check_is_terminal():
            ax.set_title(f"Game over! Scores: {self.player_scores}")
            plt.pause(2.0)
        
        plt.pause(0.01)

    
    def __repr__(self):
        return f"TTTGame(\n{np.array(self.board)}\n)"
    