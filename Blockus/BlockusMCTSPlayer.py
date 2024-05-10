from typing import Dict, List, Tuple
import warnings
from BlockusGameState import BlockusGameState
from BlockusPlayer import BlockusPlayer
from BlockusAction import BlockusAction
import numpy as np
from mcts import mcts


class BlockusStateForMCTS(BlockusGameState):
    """
    In order to run MCTS, you must implement a State class which can fully describe the state of the world. It must also implement four methods:

    - getPossibleActions(): Returns an iterable of all actions which can be taken from this state
    - takeAction(action): Returns the state which results from taking action action
    - isTerminal(): Returns whether this state is a terminal state
    - getReward(): Returns the reward for this state. Only needed for terminal states.
    
    You must also choose a hashable representation for an action as used in getPossibleActions and takeAction.
    Typically this would be a class with a custom __hash__ method, but it could also simply be a tuple or a string.
    """
    def __init__(self, game_state : BlockusGameState):
        super().__init__(game_state.state_json)
    
    def getPossibleActions(self):
        return self.get_all_possible_actions()
    
    def takeAction(self, action : BlockusAction):
        return action.modify_game(self.game, inplace=False)
    
    def isTerminal(self) -> bool:
        return self.is_terminal()
    
    def getReward(self) -> float:
        return self.calculate_reward(self.perspective_pid)
    

class BlockusMCTSPlayer(BlockusPlayer):
    
    def __init__(self,name : str = "MCTSPlayer",
                 mcts_timelimit : float = 1.0,
                 action_selection_strategy = "greedy",
                 action_selection_args : Tuple[Tuple,Dict] = ((), {}),
                 logger_args : dict = None,
                 ):
        super().__init__(name=name, logger_args=logger_args)
        self.mcts_timelimit = mcts_timelimit
        
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
        
            
        
    def evaluate_states(self, states : List[BlockusGameState]) -> List[float]:
        """ With the MCTS player we wont evaluate.
        Rather, we will run MCTS, find the best action, and return evaluations (1 for the best action, 0 for the rest)
        """
        curr_state = self.game.get_current_state()
        best_action = mcts(timeLimit=self.mcts_timelimit).search(initialState=BlockusStateForMCTS(curr_state))
        print(f"Best action: {best_action}")
        return [1 if a == best_action else 0 for a in states]
    