from typing import Dict, List, Set, TYPE_CHECKING, Tuple
from RLFramework.Action import Action
from RLFramework.Game import Game
from RLFramework.Player import Player
import numpy as np

if TYPE_CHECKING:
    from MoskaGame import MoskaGame

from MoskaAction import VALID_MOVE_IDS, MoskaAction, get_moska_action
from MoskaPlayer import MoskaPlayer


VALID_MOVE_IDS = ["AttackInitial",
                  "AttackOther",
                  "AttackSelf",
                  "KillFromHand",
                  #"KillFromDeck",
                  "EndBout",
                  "Skip"
]
class MoskaHumanPlayer(MoskaPlayer):
    def __init__(self, name : str = "MoskaHumanPlayer", max_moves_to_consider : int = 1000, logger_args : dict = None):
        super().__init__(name, max_moves_to_consider, logger_args)
        return
    
    def choose_move(self, game: Game) -> Action:
        """ Show the valid move IDs,
        """
        valid_move_ids = []
        for move_id in VALID_MOVE_IDS:
            action = get_moska_action(self.pid, move_id)
            #print(f"Checking move ID: {move_id}, action: {action}")
            if action.is_move_id_legal(game):
                valid_move_ids.append(move_id)
                #print(f"Move ID: {move_id} is valid.")
        # Print a numbered list of valid move IDs
        for i, move_id in enumerate(valid_move_ids):
            print(f"{i+1}: {move_id}")
        if len(valid_move_ids) == 0:
            raise ValueError("No valid moves found.")
        
        # Get the user's choice
        choice = None
        if len(valid_move_ids) == 1:
            choice = 1
            
        while choice is None:
            choice = input("Enter the number of the move you want to make: ")
            try:
                choice = int(choice)
                if choice < 1 or choice > len(valid_move_ids):
                    raise ValueError
            except ValueError:
                print("Invalid choice. Please enter a number between 1 and", len(valid_move_ids))
                choice = None
                
        selected_move_id = valid_move_ids[choice-1]
        
        if selected_move_id == "Skip":
            action = get_moska_action(self.pid, selected_move_id)
            
        elif selected_move_id == "EndBout":
            args = self.get_end_bout_args(game)
            action = get_moska_action(self.pid, selected_move_id, *args[0], **args[1])
            
        elif selected_move_id == "KillFromHand":
            args = self.get_kill_cards_args(game)
            action = get_moska_action(self.pid, selected_move_id, *args[0], **args[1])
            
        elif selected_move_id == "AttackSelf":
            args = self.get_attack_self_args(game)
            action = get_moska_action(self.pid, selected_move_id, *args[0], **args[1])
        
        elif selected_move_id == "AttackOther":
            args = self.get_attack_target_args(game)
            action = get_moska_action(self.pid, selected_move_id, *args[0], **args[1])
        elif selected_move_id == "AttackInitial":
            args = self.get_attack_target_args(game)
            action = get_moska_action(self.pid, selected_move_id, *args[0], **args[1])
        else:
            raise ValueError(f"Invalid move ID: {selected_move_id}")
        is_legal, msg = action.check_action_is_legal(game)
        if not is_legal:
            print(f"The action {action} is not legal: {msg}")
            return self.choose_move(game)
        return action
    
    def get_input_args_for_moveid(self, game: Game, move_id: int) -> Tuple[Tuple, Dict]:
        """ Get the input arguments for the given move_id.
        """
        if move_id == "Skip":
            return (), {}
        if move_id == "EndBout":
            return self.get_end_bout_args(game)
        
    def get_end_bout_args(self, game: 'MoskaGame') -> Tuple[Tuple, Dict]:
        """ Select whether to pick all cards, or only cards_to_kill
        """
        inp = input("Do you want to pick all cards? (y/n): ")
        if inp.lower() == "y":
            return (game.cards_to_kill + game.killed_cards,), {}
        return (game.cards_to_kill,), {}
    
    def get_kill_cards_args(self, game: 'MoskaGame') -> Tuple[Tuple, Dict]:
        """ Select the cards to kill
        """
        inp = input("Enter the indices of the kill cards (from hand) and the killed cards (from table) separated by spaces: ")
        indices = inp.split()
        try:
            indices = [int(i) for i in indices]
        except ValueError:
            print("Invalid input. Please enter the indices separated by spaces.")
            return self.get_kill_cards_args(game)
        try:
            # every even index
            cards_from_hand = [game.players[self.pid].hand[i] for i in indices[::2]]
            # every odd index
            cards_from_table = [game.cards_to_kill[i] for i in indices[1::2]]
        except IndexError:
            print("Invalid indices. Please enter the indices of the cards to kill separated by spaces.")
            return self.get_kill_cards_args(game)
        #print(f"cards_from_hand: {cards_from_hand}, cards_from_table: {cards_from_table}")
        return ({hc : tc for hc, tc in zip(cards_from_hand, cards_from_table)},), {}
    
    def get_attack_self_args(self, game: 'MoskaGame') -> Tuple[Tuple, Dict]:
        """ Select the cards to attack self
        """
        inp = input("Enter the indices of the cards to attack self: ")
        indices = inp.split()
        try:
            indices = [int(i) for i in indices]
        except ValueError:
            print("Invalid input. Please enter the indices separated by spaces.")
            return self.get_attack_self_args(game)
        try:
            cards = [game.players[self.pid].hand[i] for i in indices]
        except IndexError:
            print("Invalid indices. Please enter the indices of the cards to attack self separated by spaces.")
            return self.get_attack_self_args(game)
        return (self.pid, cards), {}
    
    def get_attack_target_args(self, game: 'MoskaGame') -> Tuple[Tuple, Dict]:
        """ Select the cards to attack target
        """
        inp = input("Enter the indices of the cards to attack target: ")
        indices = inp.split()
        try:
            indices = [int(i) for i in indices]
        except ValueError:
            print("Invalid input. Please enter the indices separated by spaces.")
            return self.get_attack_target_args(game)
        try:
            cards = [game.players[self.pid].hand[i] for i in indices]
        except IndexError:
            print("Invalid indices. Please enter the indices of the cards to attack target separated by spaces.")
            return self.get_attack_target_args(game)
        return (game.target_pid, cards), {}
    
            