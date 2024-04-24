from typing import List
from BlockusGameState import BlockusGameState
from BlockusPlayer import BlockusPlayer
import numpy as np

class BlockusGreedyPlayer(BlockusPlayer):
    
    def __init__(self,name : str = "GreedyPlayer",
                 move_selection_temp = 0,
                 logger_args : dict = None):
        super().__init__(name=name, logger_args=logger_args)
        self.move_selection_temp = move_selection_temp
        self.select_action_strategy = lambda evaluations : self._select_weighted_action(evaluations, move_selection_temp)
    
    def _get_area(self,state : 'BlockusGameState'):
        """ Get the area covered by the player in the given state.
        """
        return np.sum(state.board == self.pid)
        
    def evaluate_states(self, states : List[BlockusGameState]) -> List[float]:
        """ Evaluate the given states, according to the amount of area they cover.
        """
        evaluations = [self._get_area(s) for s in states]
        return evaluations
        