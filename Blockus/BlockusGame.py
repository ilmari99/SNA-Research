import os
import sys
from RLFramework import GameState
from RLFramework.Player import Player
from RLFramework.Result import Result
import numpy as np
from RLFramework import Game
from typing import Dict, List, Tuple
from BlockusAction import BlockusAction
from BlockusGameState import BlockusGameState
from BlockusPlayer import BlockusPlayer
from BlockusResult import BlockusResult
import matplotlib.pyplot as plt
from RLFramework.utils import TFLiteModel

from BlockusPieces import BLOCKUS_PIECE_MAP


class BlockusGame(Game):
    """ A class representing the game TicTacToe.
    """
    def __init__(self, board_size : Tuple[int, int] = (20, 20), **kwargs):
        super().__init__(BlockusGameState, custom_result_class=BlockusResult, **kwargs)
        self.board_size = board_size
        self.model_paths = []
        self.board = None
        
    def play_game(self, players: List[Player]) -> Result:
        # Load the models before playing the game
        current_models = set(self.model_paths)
        model_paths = set([p.model_path for p in players if (hasattr(p, "model_path") and p.model_path is not None)])
        # If there are any new models, load them
        if model_paths - current_models:
            self.set_models(list(model_paths))
        return super().play_game(players)
    
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

    def restore_game(self, game_state: BlockusGameState) -> None:
        """ Restore the game to the state described by the game_state.
        We don't need to worry about the players states or their scores, as they are automatically restored.
        """
        self.board = game_state.board
        self.player_remaining_pieces = game_state.player_remaining_pieces
        self.current_pid = game_state.current_pid
        self.previous_turns = game_state.previous_turns
    
    def calculate_reward(self, pid : int, game_state: BlockusGameState) -> float:
        """ Calculate the reward for the player.
        If the player wins, the reward is 1.0.
        If the game is a draw, the reward is 0.5
        """
        # Number of squares occupied by player
        player_board = np.array(game_state.board) == pid
        player_score = np.sum(player_board)
        return player_score - game_state.player_scores[pid]
    
    def check_is_player_finished(self, pid : int, game_state: GameState) -> bool:
        """ A player is finished if the game is finished.
        I.e. if the player has won, or if there are no more free spots, or if the other player is finished.
        """
        if pid in game_state.finished_players:
            return True
        return False
    
    def get_all_possible_actions(self) -> List[BlockusAction]:
        """ Return all possible actions.
        """
        available_pieces = self.player_remaining_pieces[self.current_pid]
        actions = []
        for piece_id in available_pieces:
            for x in range(self.board_size[0]):
                for y in range(self.board_size[1]):
                    for rotation in range(4):
                        for flip in [True, False]:
                            action = BlockusAction(piece_id, x, y, rotation, flip)
                            is_legal, msg = action.check_action_is_legal(self)
                            if is_legal:
                                actions.append(action)
        if len(actions) == 0:
            self.finishing_order.append(self.current_pid)
        return actions
    