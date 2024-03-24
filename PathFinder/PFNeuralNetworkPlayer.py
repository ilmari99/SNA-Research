from PFPlayer import PFPlayer
from typing import List
import numpy as np
from PFGameState import PFGameState

class PFNeuralNetworkPlayer(PFPlayer):
    """ A player that uses a neural network to select the next move.
    """
    def __init__(self,name : str = "PlayerNeuralNet", model_path : str = "", move_selection_temp = 0, logger_args : dict = None):
        super().__init__(name, logger_args)
        assert model_path, "A model path must be given."
        self.model_path = model_path
        self.move_selection_temp = move_selection_temp
        self.select_action_strategy = lambda evaluations : self._select_weighted_action(evaluations, move_selection_temp)
    
    def evaluate_states(self, states : List[PFGameState]) -> List[float]:
        """ Evaluate the given states using the neural network.
        """
        # Load the model
        model = self.game.get_model(self.model_path)
        X = np.array([s.to_vector(self.pid) for s in states], dtype=np.float32)
        #print(f"X shape: {X.shape}")
        evaluations = model.predict(X)
        #print(f"evaluations: {evaluations}")
        return evaluations