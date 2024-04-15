from typing import List
from MoskaGameState import MoskaGameState
from MoskaPlayer import MoskaPlayer
import numpy as np
from Card import REFERENCE_DECK
from utils import get_killable_mapping

class MoskaHeuristicPlayer(MoskaPlayer):
    
    def __init__(self,name : str = "HeuristicPlayer", max_moves_to_consider = 1000, move_selection_temp = 0, logger_args : dict = None):
        super().__init__(name=name, logger_args=logger_args, max_moves_to_consider=max_moves_to_consider)
        self.move_selection_temp = move_selection_temp
        self.select_action_strategy = lambda evaluations : self._select_weighted_action(evaluations, move_selection_temp)
        
    def evaluate_states(self, states : List[MoskaGameState]) -> List[float]:
        evaluations = []
        for state in states:
            evaluations.append(self.evaluate_state(state))
        return evaluations
    
    def _find_remaining_cards(self, state : MoskaGameState) -> List[int]:
        """ Find all the cards, that have not been discarded yet.
        """
        discarded = state.discarded_cards
        remaining = []
        for card in REFERENCE_DECK:
            if card not in discarded:
                remaining.append(card)
        return remaining
        
    def evaluate_state(self, state : MoskaGameState) -> float:
        """ Evaluate the given state using a heuristic function.
        """
        evaluation = 0.0
        remaining_cards = self._find_remaining_cards(state)
        # Count how many of the remianing cards each card in my hand can kill
        kill_mapping = get_killable_mapping(self.hand, remaining_cards, state.trump_card.suit)
        kill_counts = [len(kill_mapping[card]) for card in self.hand]
        total_kill_counts = sum(kill_counts)
        evaluation += total_kill_counts
        
        # If we have missing cards, and the deck is empty, and we are not the target,
        # count the number of missing cards
        if len(self.hand) < 6 and len(state.deck) == 0 and state.current_pid != state.target_pid:
            # For each missing card, we add a score of len(remaing_cards)
            evaluation += len(remaining_cards) * (6 - len(self.hand))

        # If we are finished, add a score of 1000
        are_finished = self.game.check_is_player_finished(self.pid, state)
        if are_finished and len(state.get_finished_players()) < len(state.ready_players):
            evaluation += 1000
        
        return evaluation
        