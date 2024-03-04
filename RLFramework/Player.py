from abc import ABC, abstractmethod
import random
from typing import List, TYPE_CHECKING
import numpy as np
from utils import _NoneLogger, _get_logger

from RLFramework.Action import Action
if TYPE_CHECKING:
    from RLFramework.GameState import GameState
    from RLFramework.Game import Game


class Player(ABC):
    """ A class representing a player in a game.
    The Player is simple, in that it only needs to be able to evaluate game states,
    and select action (index) based on the evaluation.
    """

    def __init__(self, name : str = "Player", logger_args : dict = None):
        self.name = name
        self.logger = _get_logger(logger_args)
        self.is_finished = False
        self.pid = None


    def choose_move(self, game : Game) -> Action:
        """ Given the game state, select the move to play.
        Note: This is only for games where the next state is known exactly.
        """
        possible_actions = game.get_all_possible_actions()
        # If there are no possible actions, return None
        if not possible_actions:
            return None
        next_states = [game.step(action, real_move = False) for action in possible_actions]
        evaluations = self.evaluate_states(next_states)
        assert len(evaluations) == len(possible_actions), f"Number of evaluations ({len(evaluations)}) must match the number of possible actions ({len(possible_actions)})"
        selected_move_idx = self.select_action_strategy(evaluations)
        return possible_actions[selected_move_idx]


    def _select_best_action(self, evaluations : List[float]) -> int:
        """ Select the action with the highest evaluation.
        """
        return evaluations.index(max(evaluations))
    
    def _select_random_action(self, evaluations : List[float]) -> int:
        """ Select a random action.
        """
        return random.choice(range(len(evaluations)))
    
    def _select_weighted_action(self, evaluations : List[float], temperature : float = 1.0) -> int:
        """ Select a random action, with the probability of each action being selected being proportional to the evaluation.
        Temperature of 1.0 means that the selection probabilities are exactly proportional to the evaluations.
        A temperature of 0.0 means that the action with the highest evaluation is always selected.
        """
        # If temperature is 0 all weight is on the best action
        if temperature == 0:
            return self._select_best_action(evaluations)
        assert temperature > 0, f"Temperature must be greater than 0, but was {temperature}"
        # Softmax
        evaluations_exp = np.exp(evaluations)
        # Adjust the selection probabilities by using temperature
        evaluations_exp_temp = evaluations_exp / temperature
        probs = evaluations_exp_temp / np.sum(evaluations_exp_temp)
        return np.random.choice(range(len(evaluations)), p=probs)
    
    def check_is_finished_wrap(self, game_state : GameState) -> bool:
        """ 'Cache' the check.
        """
        if not self.is_finished:
            self.is_finished = self.check_is_finished(game_state)
            # The player is also finished, if it is the only player in the game
            self.is_finished = self.is_finished or len(game_state.unfinished_players) == 1

        return self.is_finished
    
    @abstractmethod
    def evaluate_states(self, states : List[GameState]) -> List[float]:
        """ Evaluate the given game states.
        """
        pass

    @abstractmethod
    def check_is_finished(self, game_state : GameState) -> bool:
        """ Check if in the game_state, this player is finished.
        """
        pass

    @abstractmethod
    def select_action_strategy(self, evaluations : List[float]) -> int:
        """ Abstract method for selecting an action based on the evaluations of the game states.
        This can be one of the predefined strategies, or a custom strategy.
        """
        pass