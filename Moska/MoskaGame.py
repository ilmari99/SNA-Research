import itertools
import os
import sys
from RLFramework import GameState
from RLFramework.Action import Action
from RLFramework.Player import Player
from RLFramework.Result import Result
import numpy as np
from RLFramework import Game
from typing import Dict, List, Set, Tuple
import matplotlib.pyplot as plt
from RLFramework.utils import TFLiteModel

from MoskaGameState import MoskaGameState
from MoskaPlayer import MoskaPlayer
from MoskaAction import MoskaAction, VALID_MOVE_IDS
from Card import Card
    
    

class MoskaGame(Game):
    
    def __init__(self, name : str = "MoskaGame", logger_args : dict = None):
        super().__init__(name, logger_args)
        self.deck : List[Card] = []
        self.trump_card : Card = None
        self.players : List[MoskaPlayer] = []
        self.cards_to_kill : List[Card] = []
        self.killed_cards : List[Card] = []
        self.discarded_cards : List[Card] = []
        self.target_pid : int = 0
        self.model_paths : List[str] = []
        self.models : Dict[str, TFLiteModel] = {}
    
    def get_all_possible_actions(self) -> List[MoskaAction]:
        """ Return a list with all possible actions, that the current player can make.
        """
        for move_id in VALID_MOVE_IDS:
            if move_id == "AttackInitial":
                pass
            elif move_id == "AttackOther":
                pass
            elif move_id == "AttackSelf":
                pass
            elif move_id == "KillFromHand":
                pass
            elif move_id == "KillFromDeck":
                pass
            elif move_id == "EndBout":
                pass
            elif move_id == "Skip":
                pass
            else:
                raise ValueError(f"Invalid move_id: {move_id}")
    
    def get_ranks_on_table(self) -> Set[int]:
        """ Get the ranks of the cards on the table.
        """
        return set([card.rank for card in self.cards_to_kill + self.killed_cards])
    
    def check_target_can_end_bout(self) -> bool:
        """ The target can end the bout,
        if there is atleast 1 card in the table, and
        all other players are ready.
        """
        return (len(self.cards_to_kill + self.killed_cards) > 0 and
                all(player.ready for player in self.players if player.pid != self.target_pid))
        
    def check_target_can_kill_from_deck(self) -> bool:
        return len(self.deck) > 0 and not any(card.kopled for card in self.cards_to_kill)
    

    def check_target_must_end_bout(self) -> bool:
        """ The target player must end the bout, if:
        - There are cards on the table, and
        - They have no playable cards, and 
        - All players are ready, and
        - The player can't play from deck (there is no deck left or there is a kopled card already)
        """
        if not self.check_target_can_end_bout():
            return False
        target = self.players[self.target_pid]
        kill_mapping = target.get_possible_kill_mapping()
        can_kill = any(kill_mapping.values())
        # If the target can kill
        if can_kill:
            return False
        has_playble_ranks = bool(target.get_playable_ranks_from_hand())
        # If the target has playable ranks (to play to self)
        if has_playble_ranks:
            return False
        can_kill_from_deck = self.check_target_can_kill_from_deck()
        # If the target can kill from deck
        if can_kill_from_deck:
            return False
        return True
                
    def restore_game(self, game_state: MoskaGameState) -> None:
        """ Given a gamestate, restore the game to that state.
        In Moska, this means restoring:
        - Deck,
        - Trump card,
        - Players' cards,
        - Known cards in each hand,
        - Cards to kill,
        - Killed cards,
        - Discarded cards,
        - Current player,
        - Target player,
        - Which players are ready (have played Skip once after the board last changed),        
        """
        self.deck = [Card(suit, value) for suit, value in game_state.deck]
        self.trump_card = Card(game_state.trump_card[0], game_state.trump_card[1])
        for i, player in enumerate(self.players):
            player.hand = [Card(suit, value) for suit, value in game_state.players_hands[i]]
            player.known_cards = [Card(suit, value) for suit, value in game_state.players_known_cards[i]]
            player.ready = game_state.players_ready[i]
        self.cards_to_kill = [Card(suit, value, kopled) for suit, value, kopled in game_state.cards_to_kill]
        self.killed_cards = [Card(suit, value) for suit, value in game_state.killed_cards]
        self.discarded_cards = [Card(suit, value) for suit, value in game_state.discarded_cards]
        self.current_pid = game_state.current_pid
        self.target_pid = game_state.target_pid
    
    def initialize_game(self, players: List[Player]) -> None:
        """ Initialize the MoskaGame instance.
        Here, we create the deck, shuffle it, and deal the cards to the players.
        We also select the trump card/suit.
        """
        assert len(players) >= 2 and len(players) <= 8, "Moska is a game for 2-8 players."
        self.players = players
        self.deck = [Card(suit, value) for suit, value in itertools.product(CARD_SUITS, CARD_VALUES)]
        np.random.shuffle(self.deck)
        for i, player in enumerate(players):
            # Remove 6 cards from the top of the deck
            player.hand = [self.deck.pop(0) for _ in range(6)]
        # The next card is the trump card
        self.trump_card = self.deck.pop(0)
        # Place the trump card at the bottom of the deck
        self.deck.append(self.trump_card)
        
    def check_is_player_finished(self, pid : int, game_state : MoskaGameState) -> bool:
        """ A player is finished, if they have no cards left in their hand,
        and the deck is empty.
        Also, if the player is the target, then additionally,
        all the cards on the table have to have been killed (cards_to_kill is empty)
        """
        player = self.players[pid]
        if pid != game_state.target_pid:
            return (len(player.hand) == 0 and
                    len(self.deck) == 0)
        
        return (len(player.hand) == 0 and
                len(self.deck) == 0 and
                len(game_state.cards_to_kill) == 0)
        
    def select_turn(self, players: List[Player], previous_turns: List[int]) -> int:
        """ In Moska, the turns are mostly based on speed.
        But in this implementation, we will select the next player randomly.
        """
        return self._select_random_turn(players, previous_turns)
    
    def play_game(self, players: List[MoskaPlayer]) -> Result:
        # Load the models before playing the game
        current_models = set(self.model_paths)
        model_paths = set([p.model_path for p in players if (hasattr(p, "model_path") and p.model_path is not None)])
        # If there are any new models, load them
        if model_paths - current_models:
            self.set_models(list(model_paths))
        return super().play_game(players)
    
    def get_model(self, model_name : str) -> TFLiteModel:
        """ Get the model with the given name.
        """
        model_name = os.path.abspath(model_name)
        try:
            return self.models[model_name]
        except KeyError:
            raise ValueError(f"Model with name {model_name} not found. Available models: {list(self.models.keys())}")
        
    def set_models(self, model_paths : List[str]) -> None:
        """ Set the models to the given paths.
        """
        self.model_paths = model_paths
        self.models = {path: TFLiteModel(path) for path in model_paths}
        
    @property
    def trump_suit(self) -> str:
        return self.trump_card.suit
    
    @property
    def num_fits_to_table(self) -> int:
        """ Return the number of cards that fit the cards to kill.
        """
        return len(self.players[self.target_pid].hand) - len(self.cards_to_kill)