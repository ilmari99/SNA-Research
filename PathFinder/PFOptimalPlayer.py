import math
from typing import List

from PFPlayer import PFPlayer

class PFOptimalPlayer(PFPlayer):
    """ A class representing an optimal player for the PathFinder game
    """
    
    def evaluate_states(self, states) -> List[float]:
        """ Measure the value of the states.
        """
        evals = []
        for state in states:
            evals.append(-math.sqrt((state.goal[0] - state.current_player_pos[0])**2 + (state.goal[1] - state.current_player_pos[1])**2))
        return evals
    
    def select_action_strategy(self, evaluations):
        return self._select_best_action(evaluations)