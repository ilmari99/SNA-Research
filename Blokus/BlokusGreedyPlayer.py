from typing import List
import warnings
from Blokus.BlokusGameState import BlokusGameState
from BlokusPlayer import BlokusPlayer
import numpy as np

class BlokusGreedyPlayer(BlokusPlayer):
    
    def __init__(self,name : str = "GreedyPlayer",
                 action_selection_strategy : str = "greedy",
                action_selection_args : tuple = ((), {}),
                 logger_args : dict = None):
        super().__init__(name=name, logger_args=logger_args)
        action_selection_map = {
            "greedy" : self._select_best_action,
            "random" : self._select_random_action,
            "weighted" : self._select_weighted_action,
            "epsilon_greedy" : self._select_epsilon_greedy_action,
        }
        if action_selection_strategy not in action_selection_map:
            raise ValueError(f"Unknown action selection strategy '{action_selection_strategy}'")
        if action_selection_strategy in ["greedy", "random"]:
            if action_selection_args != ((), {}):
                warnings.warn(f"action selection strategy '{action_selection_strategy}' does not use arguments.")
                action_selection_args = ((), {})
        f = action_selection_map[action_selection_strategy]
        self.select_action_strategy = lambda evaluations : f(evaluations, *action_selection_args[0], **action_selection_args[1])
    
    def _get_area(self,state : 'BlokusGameState'):
        """ Get the area covered by the player in the given state.
        """
        return np.sum(state.board == self.pid)
        
    def evaluate_states(self, states : List[BlokusGameState]) -> List[float]:
        """ Evaluate the given states, according to the amount of area they cover.
        """
        evaluations = [self._get_area(s) for s in states]
        return evaluations