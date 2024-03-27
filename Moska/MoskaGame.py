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
from Card import Card, REFERENCE_DECK

from utils import get_initial_attacks, get_all_matchings


class MoskaGame(Game):
    
    def __init__(self, name : str = "MoskaGame", logger_args : dict = None):
        super().__init__(name, logger_args)
        self.deck : List[Card] = []
        self.trump_card : Card = None
        self.players : List[MoskaPlayer] = []
        self.cards_to_kill : List[Card] = []
        self.killed_cards : List[Card] = []
        self.discarded_cards : List[Card] = []
        self.players_ready : List[bool] = []
        self.player_full_cards : List[List[Card]] = []
        self.player_public_cards : List[List[Card]] = []
        self.target_pid : int = 0
        self.model_paths : List[str] = []
        self.models : Dict[str, TFLiteModel] = {}
    
    def get_all_possible_actions(self) -> List[MoskaAction]:
        """ Return a list with all possible actions, that the current player can make.
        """
        if self.players[self.current_pid] in self.get_finished_players():
            return []
        actions = []
        gs : MoskaGameState = self.get_current_state()
        for move_id in VALID_MOVE_IDS:
            if move_id == "AttackInitial":
                # Only the initiating player can begin the bout
                if not gs.player_is_initiating(self.current_pid):
                    continue
                possible_plays = get_initial_attacks(self.player_full_cards[self.current_pid], self.num_fits_to_table)
                actions += [MoskaAction(self.current_pid, move_id, target_pid = self.target_pid, cards = list(play)) for play in possible_plays]
            elif move_id == "AttackOther":
                # If the bout is not initiated, then the player can't attack
                if not gs.bout_is_initiated():
                    continue
                playable_values = self.players[self.current_pid].get_playable_ranks_from_hand()
                playable_cards = [card for card in self.player_full_cards[self.current_pid] if card.rank in playable_values]
                play_iterables = []
                for i in range(1, min(len(playable_cards) + 1, self.num_fits_to_table + 1)):
                    play_iterables.append(itertools.combinations(playable_cards, i))
                plays = itertools.chain.from_iterable(play_iterables)
                actions += [MoskaAction(self.current_pid, move_id, target_pid = pid, cards = list(play)) for pid, play in plays]
                
            elif move_id == "AttackSelf":
                playable_values = self.players[self.current_pid].get_playable_ranks_from_hand()
                playable_cards = [card for card in self.player_full_cards[self.current_pid] if card.rank in playable_values]
                play_iterables = []
                for i in range(1, len(playable_cards) + 1):
                    play_iterables.append(itertools.combinations(playable_cards, i))
                plays = itertools.chain.from_iterable(play_iterables)
                actions += [MoskaAction(self.current_pid, move_id, target_pid = self.current_pid, cards = list(play)) for play in plays]

            elif move_id == "KillFromHand":
                if self.current_pid != self.target_pid:
                    continue
                possible_killings = get_all_matchings(self.player_full_cards[self.current_pid],
                                                      self.cards_to_kill,
                                                      trump=self.trump_card.suit,
                )
                # Elements of possble_killings
                # are Assignments (at utils.py)
                # They have two attrs: _hand_inds, _table_inds
                # lets convert them to cards
                possible_killings_as_cards = []
                for assignment in possible_killings:
                    hand_inds = assignment._hand_inds
                    table_inds = assignment._table_inds
                    cards_from_hand = [self.player_full_cards[self.current_pid][i] for i in hand_inds]
                    cards_from_table = [self.cards_to_kill[i] for i in table_inds]
                    possible_killings_as_cards.append({hc : tc for hc, tc in zip(cards_from_hand, cards_from_table)})
                actions += [MoskaAction(self.current_pid, move_id, kill_mapping = kill_mapping) for kill_mapping in possible_killings_as_cards]


            elif move_id == "KillFromDeck":
                # We can't kill from deck if we are not the target
                if self.current_pid != self.target_pid:
                    continue
                if not gs.target_can_play_from_deck():
                    continue
                actions.append(MoskaAction(self.current_pid, move_id))
                
                
                
            
            elif move_id == "EndBout":
                pass
            elif move_id == "Skip":
                pass
            else:
                raise ValueError(f"Invalid move_id: {move_id}")

        
    def get_players_condition(self, condition = None) -> List[MoskaPlayer]:
        """ Get the players that satisfy the condition.
        """
        if condition is None:
            return self.players
        return [player for player in self.players if condition(player)]
        

    def get_finished_players(self) -> List[int]:
        """ Get the finished players.
        """
        gs : MoskaGameState = self.get_current_state()
        return gs.get_finished_players()
    
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
    
    def step(self, action: Action, real_move=True) -> MoskaGameState:
        # Super step automatically sets the next player,
        # and appends them to the previous turns
        state : MoskaGameState = super().step(action, real_move)
        # Hide whose turn it is
        state.current_pid = -1
        state.previous_turns.pop()
        return state
    
    def set_missing_info(self) -> None:
        """ Set the missing info in the game state.
        So if no one has the turn, or people are missing cards.
        """
        # If no one has the turn, select a player
        if self.current_pid == -1:
            self.current_pid = self.select_turn(self.players, self.previous_turns)
            self.previous_turns.append(self.current_pid)
        # Fill the hands of the players
        self.fill_hands()
    
    def fill_hands(self, game_state : MoskaGameState = None, inplace = True) -> None:
        """ Fill the hands of the players.
        """
        if game_state is None:
            game_state = self.get_current_state()
        if len(game_state.deck) == 0:
            return
        # Find a player with less than 6 cards
        player_with_missing_cards = None
        for i, player_full_cards in enumerate(game_state.player_full_cards):
            if len(player_full_cards) < 6:
                player_with_missing_cards = i
                break
            
        if player_with_missing_cards is None:
            return
        # Fill the hand of the player
        pick_n_cards = min(6 - len(self.players[player_with_missing_cards].hand), len(self.deck))
        game_state.player_full_cards[player_with_missing_cards] += [self.deck.pop(0) for _ in range(pick_n_cards)]
        if inplace:
            self.restore_game(game_state)
        return
    
    def play_game(self, players : List['Player']) -> Result:
        """ Play a game with the given players.
        """
        self.reset()
        # Initilize the game by setting internal variables, custom variables, and checking the initialization
        self.initialize_game_wrap(players)
        
        self.current_pid = self.select_turn(players, self.previous_turns)
        self.render()
        self.game_states.append(self.get_current_state().deepcopy())
        # Play until all players are finished
        while not self.check_is_terminal() and self.total_num_played_turns < self.max_num_total_steps:
            #print(f"Starting turn for player {self.current_player_name}")
            # Select the next player to play
            self.current_player_name = players[self.current_pid].name
            player = players[self.current_pid]
            game_state = self.get_current_state(player)
            assert game_state.check_is_game_equal(self), "The game state was not created correctly."

            # Choose an action with the player
            # In MOSKA, the player makes the choice without knowing what are the
            # next cards, or the next player.
            action = player.choose_move(self)
            #print(f"{self.current_player_name} chose action {action}")
            if action is not None:
                # Here, we have not yet lifted the cards or set the next player
                new_state = self.step(action)
                
                self.logger.info(f"Player {self.current_player_name} chose action {action}.")

                # We'll add the gamestate with pending information
                # Since the neural net will have to
                # make a decision based on this information
                self.game_states.append(new_state.deepcopy())
                # Realize the pending information, so
                # give missing cards, and set the next player
                self.set_missing_info()
                # Also add the new state, after the cards have been lifted
                self.game_states.append(self.get_current_state().deepcopy())
                self.total_num_played_turns += 1
            else:
                new_state = game_state
                self.logger.info(f"Player '{self.current_player_name}' has no moves.")
                self.update_state_after_action(new_state)
                new_state.set_game_state(self)
            #print(f"Player scores: {self.player_scores}")
            
            #print(f"{self.current_player_name} has {player.score} score")
            self.logger.debug(f"Game state:\n{new_state}")
            self.render()
        
        # Winner is the player with the higher score
        winner = players[np.argmax(self.player_scores)].name
        # If multiple players have the same score, the winner is None
        if len(set(self.player_scores)) != len(self.player_scores):
            winner = None
        
        result = self.result_class(successful = True if self.check_is_terminal() else False,
                        player_jsons = [player.as_json() for player in players],
                        finishing_order = self.finishing_order,
                        logger_args = self.logger_args,
                        game_state_class = self.game_state_class,
                        game_states = self.game_states,
                        previous_turns = self.previous_turns,
                        winner=winner,
                        )
        s = "Game finished with results:"
        for k,v in result.as_json(states_as_num = True).items():
            s += f"\n{k}: {v}"
        self.logger.info(s)
        if self.gather_data and result.successful:
            result.save_game_states_to_file(self.gather_data)
        return result
    
    
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
        self.deck = game_state.deck
        self.trump_card = game_state.trump_card
        self.player_full_cards = game_state.player_full_cards
        self.player_public_cards = game_state.player_public_cards
        self.ready_players = game_state.players_ready
        
        self.cards_to_kill = game_state.cards_to_kill
        self.killed_cards = game_state.killed_cards
        self.discarded_cards = game_state.discarded_cards
        self.current_pid = game_state.current_pid
        self.target_pid = game_state.target_pid
    
    def initialize_game(self, players: List[Player]) -> None:
        """ Initialize the MoskaGame instance.
        Here, we create the deck, shuffle it, and deal the cards to the players.
        We also select the trump card/suit.
        """
        assert len(players) >= 2 and len(players) <= 8, "Moska is a game for 2-8 players."
        self.players = players
        self.deck = REFERENCE_DECK.copy()
        np.random.shuffle(self.deck)
        player_full_cards = []
        for i, player in enumerate(players):
            # Remove 6 cards from the top of the deck
            player_full_cards.append([self.deck.pop(0) for _ in range(6)])
        self.player_full_cards = player_full_cards
        # The next card is the trump card
        self.trump_card = self.deck.pop(0).copy()
        # Place the trump card at the bottom of the deck
        self.deck.append(self.trump_card)
        self.players_ready = [False for _ in players]
        
    def check_is_player_finished(self, pid : int, game_state : MoskaGameState) -> bool:
        """ A player is finished, if they have no cards left in their hand,
        and the deck is empty.
        Also, if the player is the target, then additionally,
        all the cards on the table have to have been killed (cards_to_kill is empty)
        """
        player_hand = game_state.player_full_cards[pid]
        if pid == game_state.target_pid:
            return (len(player_hand) == 0 and
                    len(self.deck) == 0 and
                    len(game_state.cards_to_kill) == 0)
            
        return (len(player_hand) == 0 and
                len(self.deck) == 0)
        
    def select_turn(self, players: List[Player], previous_turns: List[int]) -> int:
        """ In Moska, the turns are mostly based on speed.
        But in this implementation, we will select the next player randomly.
        """
        # If the game is not initiated, return the player before
        # the target
        #if len(self.cards_to_kill + self.killed_cards) == 0:
        #    target_pid = self.target_pid
        #    total_num_players = len(players)
        #    return (target_pid - 1) % total_num_players
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