from abc import ABC, abstractmethod
import random
from typing import List, TYPE_CHECKING
import numpy as np
import functools as ft

from .utils import _get_logger
from .Action import Action
if TYPE_CHECKING:
    from .GameState import GameState
    from .Game import Game


class Player(ABC):
    """ A class representing a player in a game.
    The Player is simple, in that it only needs to be able to evaluate game states,
    and select action (index) based on the evaluation.
    """

    def __init__(self, name : str = "Player", logger_args : dict = None):
        self.name = name
        default_logger_args = {
            "name" : f"{name}_logger",
            "level" : 10,
            "log_file" : None,
            "write_mode" : "w",
        }
        self.logger_args = {**default_logger_args, **logger_args} if logger_args is not None else None
        self.logger = _get_logger(self.logger_args)
        self.is_finished = False
        self.pid = None
        self.score = 0

    def as_json(self) -> dict:
        """ Return the player as a json.
        """
        return {
            "classname" : self.__class__.__name__,
            "name" : self.name,
            "pid" : self.pid,
            "score" : self.score,
            "is_finished" : self.is_finished,
            "logger_args" : self.logger_args,
            }


    def choose_move(self, game : 'Game') -> 'Action':
        """ Given the game state, select the move to play.
        Note: This is only for games where the next state is known exactly.
        """
        self.logger.debug(f"Game state:\n{game}")
        possible_actions = game.get_all_possible_actions()
        self.logger.info(f"Found {len(possible_actions)} possible actions.")
        # If there are no possible actions, return None
        if not possible_actions:
            return None
        next_states = []
        game_state = game.game_state_class.from_game(game, copy=True)
        for action in possible_actions:
            next_state = game.step(action, real_move = False)
            next_states.append(next_state)
        #next_states = [game.step(action, real_move = False) for action in possible_actions]
        evaluations = self.evaluate_states(next_states)
        self.logger.debug(f"Moves and evaluations:\n{list(zip(possible_actions, evaluations))}")
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
        return np.random.choice(range(len(evaluations)))
    
    def _select_weighted_action(self, evaluations : List[float], temperature : float = 1.0) -> int:
        """ Select a random action, with the probability of each action being selected being proportional to the evaluation.
        Temperature of 1.0 means that the selection probabilities are exactly proportional to the evaluations.
        A temperature of 0.0 means that the action with the highest evaluation is always selected.
        """
        # If temperature is 0 all weight is on the best action
        if temperature == 0:
            return self._select_best_action(evaluations)
        assert temperature > 0, f"Temperature must be greater than 0, but was {temperature}"
        choice_evals = [(i, e) for i, e in enumerate(evaluations)]
        choice_evals = sorted(choice_evals, key = lambda x : x[1], reverse = True)
        # Softmax
        evals_exp = np.exp([x[1] for x in choice_evals]).flatten()
        evals = evals_exp / np.sum(evals_exp)
        cumsum = np.cumsum(evals)
        # Find the first index where the cumulative sum is gte than p
        idx = np.argmax(cumsum >= temperature)
        # Redo the softmax
        valid_choices = choice_evals[:idx+1]
        evals_exp = np.exp([x[1] for x in valid_choices]).flatten()
        evals = evals_exp / np.sum(evals_exp)
        indices = [x[0] for x in valid_choices]
        #print(f"valid_choices: {valid_choices}")
        return np.random.choice(indices, p = evals)
        
        
        
        
    @staticmethod
    def initialize_player_decorator():
        """ Decorator for the initialize_player method."""
        def decorator(func):
            ft.wraps(func)
            def wrapper(self : 'Player', game : 'Game'):
                self.pid = game.players.index(self)
                self.is_finished = False 
                self.logger.debug(f"Initilaized player with arguments {self.as_json()}")
                return func(self, game)
            return wrapper
        return decorator
    
    def __init_subclass__(cls) -> None:
        """ Persist the decorators in the subclass.
        """
        super().__init_subclass__()
        cls.initialize_player = cls.initialize_player_decorator()(cls.initialize_player)
        #cls.check_is_finished = cls.check_is_finished_decorator()(cls.check_is_finished)        

    @abstractmethod
    def initialize_player(self, game : 'Game') -> None:
        """ Set some desired vriables for the player.
        """
        pass
    
    @abstractmethod
    def evaluate_states(self, states : List['GameState']) -> List[float]:
        """ Evaluate the given game states.
        """
        pass

    @abstractmethod
    def select_action_strategy(self, evaluations : List[float]) -> int:
        """ Abstract method for selecting an action based on the evaluations of the game states.
        This can be one of the predefined strategies, or a custom strategy.
        """
        pass