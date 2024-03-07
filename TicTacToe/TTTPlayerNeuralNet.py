from TTTPlayer import TTTPlayer
from typing import List
import numpy as np
from TTTGameState import TTTGameState

class TTTPlayerNeuralNet(TTTPlayer):
    """ A player that uses a neural network to select the next move.
    """
    def __init__(self,name : str = "PlayerNeuralNet", model_path : str = "", logger_args : dict = None):
        super().__init__(name, logger_args)
        assert model_path, "A model path must be given."
        self.model_path = model_path
    
    def evaluate_states(self, states : List[TTTGameState]) -> List[float]:
        """ Evaluate the given states using the neural network.
        """
        # Load the model
        model = self.game.get_model(self.model_path)
        X = np.array([s.to_vector() + [self.pid] for s in states], dtype=np.float32)
        evaluations = model.predict(X)
        print(f"evaluations: {evaluations}")
        return evaluations
        