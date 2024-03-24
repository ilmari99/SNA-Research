import os
import sys
from RLFramework import GameState
from RLFramework.Action import Action
from RLFramework.Player import Player
from RLFramework.Result import Result
import numpy as np
from RLFramework import Game
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from RLFramework.utils import TFLiteModel

from PFGameState import PFGameState
from PFPlayer import PFPlayer
from PFAction import PFAction

class PFGame(Game):
    
    def __init__(self, board_size : Tuple[int, int] = (3, 3), **kwargs):
        super().__init__(PFGameState, **kwargs)
        self.board_size = board_size
        self.model_paths = []
        self.num_moves = 0
        
    def render_human(self, ax: plt.Axes = None) -> None:
        """ Render the game state.
        """
        if ax is None:
            fig, ax = plt.subplots()
        ax.imshow(self.board)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"PathFinder Game: {self.num_moves} moves")
        if self.check_is_terminal():
            ax.set_title(f"Game over! Scores: {self.player_scores}")
            plt.pause(2.0)
        else:
            plt.pause(0.1)
        
    def restore_game(self, game_state: GameState) -> None:
        """ Restore the game to the state described by the game_state.
        We don't need to worry about the players states or their scores, as they are automatically restored.
        """
        self.board = game_state.board
        self.goal = game_state.goal
        self.num_moves = game_state.num_moves
        self.player.position = game_state.current_player_pos
    
    def initialize_game(self, players: List[PFPlayer]) -> None:
        """ When the game is started, we need to set the board.
        The board is full of 0s, except 2 at a random position which is the goal.
        """
        assert len(players) == 1, "PathFinder is a single player game."
        player = players[0]
        self.player = player
        self.board = [[0 for _ in range(self.board_size[1])] for _ in range(self.board_size[0])]
        self.goal = (np.random.randint(0, self.board_size[0]), np.random.randint(0, self.board_size[1]))
        self.board[self.goal[0]][self.goal[1]] = 2
        player.position = [np.random.randint(0, self.board_size[0]), np.random.randint(0, self.board_size[1])]
        self.board[player.position[0]][player.position[1]] = 1
        
    def calculate_reward(self, pid: int, new_state: GameState):
        """ Calculate the reward for the player with the given pid.
        """
        if self.check_is_player_finished(pid, new_state):
            return 1
        return -0.05
    
    def check_is_player_finished(self, pid : int, game_state : PFGameState) -> bool:
        """ Check if the player has reached the goal.
        """
        return game_state.board[self.goal[0]][self.goal[1]] == 1
    
    def select_turn(self, players: List[Player], previous_turns: List[int]) -> int:
        """ PathFinder is a single player game, so we always return 0.
        """
        return 0
    
    def get_all_possible_actions(self) -> List[Action]:
        """ At each step, the player can move to any of the 4 adjacent cells.
        """
        # The current position is 
        curr_pos = self.player.position
        possible_actions = []
        for x in range(-1, 2):
            for y in range(-1, 2):
                if abs(x) + abs(y) == 1:
                    new_x = curr_pos[0] + x
                    new_y = curr_pos[1] + y
                    if 0 <= new_x < self.board_size[0] and 0 <= new_y < self.board_size[1]:
                        possible_actions.append(PFAction(new_x, new_y))
        return possible_actions
    
    def play_game(self, players: List[PFPlayer]) -> Result:
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
        
    def set_models(self, model_paths : List[str]) -> None:
        """ Set the models to the given paths.
        """
        self.model_paths = model_paths
        self.models = {path: TFLiteModel(path) for path in model_paths}

