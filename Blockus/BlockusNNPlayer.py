from typing import Dict, List, Tuple
import warnings
from BlockusGameState import BlockusGameState
from BlockusPlayer import BlockusPlayer
import numpy as np

class BlockusNNPlayer(BlockusPlayer):
    
    def __init__(self,name : str = "NNPlayer",
                 model_path : str = "",
                 action_selection_strategy = "greedy",
                 action_selection_args : Tuple[Tuple,Dict] = ((), {}),
                 logger_args : dict = None,
                 ):
        super().__init__(name=name, logger_args=logger_args)
        assert model_path, "A model path must be given."
        self.model_path = model_path
        
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
        """ Evaluate the given states using the neural network.
        """
        # Load the model
        model = self.game.get_model(self.model_path)
        X = np.array([s.to_vector(self.pid) for s in states], dtype=np.float32)
        #print(f"X shape: {X.shape}")
        evaluations = model.predict(X)
        #print(f"evaluations: {evaluations}")
        return evaluations