from collections import Counter
import itertools
import random
from typing import List, Set, Tuple
import tracemalloc

import numpy as np

class Assignment:
    """ An assignment is a mapping from cards in the hand to cards on the table.
    Two assignments are equal if the same cards are played to the same cards, regardless of order.
    """
    __slots__ = ['inds']
    def __init__(self, inds : Tuple[int]):
        self.inds = inds
    
    @property
    def _hand_inds(self):
        return self.inds[::2]
    
    @property
    def _table_inds(self):
        return self.inds[1::2]
    
    def __eq__(self, other):
        """ Two assignments are equal if the same cards are played to the same cards, regardless of order."""
        return set(self._hand_inds) == set(other._hand_inds) and set(self._table_inds) == set(other._table_inds)
    
    def __repr__(self):
        return f"Assignment({self.inds})"
    
    def __hash__(self):
        """ Two assignments are equal if the same cards are played to the same cards, regardless of order."""
        hand_inds = sorted(self._hand_inds)
        table_inds = sorted(self._table_inds)
        return hash((tuple(hand_inds), tuple(table_inds)))
        

class Card:
    """ A class representing a card.
    """
    __slots__ = ['suit', 'rank', 'kopled']
    def __init__(self, suit : str, rank : int, kopled = False):
        self.suit = suit
        self.rank = rank
        self.kopled = kopled
        
    def __eq__(self, other) -> bool:
        return self.suit == other.suit and self.rank == other.rank
    
    def __repr__(self) -> str:
        s = f"{self.suit}{self.rank}"
        s += "\'" if self.kopled else ""
        return s
    
    def __hash__(self) -> int:
        return hash((self.suit, self.rank))
    
    def __lt__(self, other) -> bool:
        return self.rank < other.rank
    
    def __eq__(self, other) -> bool:
        return self.rank == other.rank and self.suit == other.suit
    
    def __copy__(self):
        return Card(self.suit, self.rank, self.kopled)
    
    def copy(self):
        return self.__copy__()
    
    def __deepcopy__(self, memo=None):
        return Card(self.suit, self.rank, self.kopled)
    
    def deepcopy(self):
        return self.__deepcopy__()


def make_kill_matrix(from_ : List[Card], to : List[Card], trump : str) -> np.ndarray:
    """ Make a matrix (len(from_) x len(to)), where index (i,j) is 1
    if the card in hand at index i can kill the card on the table at index j, 0 otherwise.
    """
    matrix = np.zeros((len(from_),len(to)), dtype=np.int8)
    for i, card in enumerate(from_):
        for j, killed_card in enumerate(to):
            if check_can_kill_card(card,killed_card,trump):
                matrix[i,j] = 1
    return matrix

def get_single_kills(matrix : np.ndarray) -> List[List[int]]:
    """ Return all single card assignments from the assignment matrix as a list of (row, col) tuples.
    """
    nz = np.nonzero(matrix)
    return list(zip(nz[0],nz[1]))

def get_all_matchings(from_ : List[Card], to : List[Card] = [], trump : str = "", start=[], found_assignments = None, max_moves : int = 1000) -> Set[Assignment]:
    """ Return a set of found Assignments, containing all possible assignments of cards from the hand to the cards to fall.
    Symmetrical assignments are considered the same when the same cards are played to the same cards, regardless of order.
    
    The assignments (partial matchings) are searched for recursively, in a depth-first manner:
    - Find all single card assignments (row-col pairs where the intersection == 1)
    - Add the assignment to found_assignments
    - Mark the vertices (row and column of the cost_matrix) as visited (0) (played_card = column, hand_card = row)
    - Repeat
    """
    matrix = None
    if not to:
        if not isinstance(from_, np.ndarray):
            raise TypeError("from_ must be a numpy array if to_ is not given")
        matrix = from_
    # Create a set of found assignments, if none is given (first call)
    if not found_assignments:
        found_assignments = set()
    # If no matrix is given, create the matrix, where 1 means that the card can fall, and 0 means it can't
    if matrix is None:
        if not trump:
            raise ValueError("trump must be given if from_ is not a cost matrix")
        matrix = make_kill_matrix(from_ = from_, to = to, trump = trump)
        
    # Find all single card assignments (row-col pairs where the intersection == 1)
    new_assignments = get_single_kills(matrix)
    
    # If there are no more assignments, or the max number of states is reached, return the found assignments
    for row, col in new_assignments:
        if len(found_assignments) >= max_moves:
            return found_assignments
        og_len = len(found_assignments)
        # Named tuple with a custom __eq__ and __hash__ method
        assignment = Assignment(tuple(start + [row,col]))
        found_assignments.add(assignment)
        # If the assignment was already found, there is no need to recurse deeper, since there could only be more symmetrical assignments
        if len(found_assignments) == og_len:
            continue
        # Get the visited cards
        # The cards in hand are even (0,2...), the cards on the table are odd (1,3...)
        #hand_cards = [c for i,c in enumerate(start + [row,col]) if i % 2 == 0]
        #table_cards = [c for i,c in enumerate(start + [row,col]) if i % 2 == 1]
        hand_cards = assignment._hand_inds
        table_cards = assignment._table_inds
        # Store the values, to restore them later
        row_vals = matrix[hand_cards,:]
        col_vals = matrix[:,table_cards]
        # Mark the cards as visited (0)
        matrix[hand_cards,:] = 0
        matrix[:,table_cards] = 0
        # Find the next assignments, which adds the new assignments to the found_assignments
        get_all_matchings(from_ = matrix, start = start +[row,col], found_assignments = found_assignments, max_moves=max_moves)
        # Restore matrix
        matrix[hand_cards,:] = row_vals
        matrix[:,table_cards] = col_vals
    return found_assignments

def check_can_kill_card(kill_card : Card,
                        killed_card : Card,
                        trump : str) -> bool:
    """Returns true, if the kill_card, can kill the killed_card.
    The kill card can kill killed_card, if:
    - The kill card has the same suit and is greater than killed_card
    - If the kill_card is trump suit, and the killed_card is not.

    Args:
        kill_card (Card): The card played from hand
        killed_card (Card): The card on the table
        trump (str): The trump suit of the current game

    Returns:
        bool: True if kill_card can kill killed_card, false otherwise
    """
    success = False
    # Jos kortit ovat samaa maata ja pelattu kortti on suurempi
    if kill_card.suit == killed_card.suit and kill_card.rank > killed_card.rank:
        success = True
    # Jos pelattu kortti on valttia, ja kaadettava kortti ei ole valttia
    elif kill_card.suit == trump and killed_card.suit != trump:
            success = True
    return success

if __name__ == "__main__":
    tracemalloc.start()
    random.seed(0)
    deck = [Card(suit, value)for suit, value in itertools.product(["C","D","H","S"], range(2,15))]
    random.shuffle(deck)
    # Profile the memory use if get_initial_attacks
    cards = deck[:10]
    trump = "H"
    table_cards = deck[10:30]
    max_moves = 10000
    mem1 = tracemalloc.take_snapshot()
    killings = get_all_matchings(cards, table_cards, trump, max_moves=max_moves)
    mem2 = tracemalloc.take_snapshot()
    print(f"Found {len(killings)} assignments")
    diff = mem2.compare_to(mem1, 'lineno')
    print("[ Top 10 differences ]")
    for d in diff[:10]:
        print(d)
    
        
    