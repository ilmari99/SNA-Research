


import itertools
from typing import Dict, Iterable, List


CARD_VALUES = tuple(range(2,15))                            # Initialize the standard deck
CARD_SUITS = ("C","D","H","S") 
CARD_SUIT_SYMBOLS = {"S":'♠', "D":'♦',"H": '♥',"C": '♣',"X":"X"}    #Conversion table
USING_CARD_SYMBOLS = False

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
        # If we are using card symbols, then print the card with the symbol
        if USING_CARD_SYMBOLS:
            s = f"{CARD_SUIT_SYMBOLS[self.suit]}{self.rank}"
        else:
            # If we have a different encoding, then print the suit as a letter
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

    #def __dict__(self):
    #    return {"suit": self.suit, "rank": self.rank, "kopled": self.kopled}

REFERENCE_DECK = [Card(suit, value) for suit, value in itertools.product(CARD_SUITS, CARD_VALUES)]
    
def serialize_cards(cards : Iterable[Card]) -> List[Dict]:
    """ Serialize a list of cards to a list of dictionaries.
    """
    return [{"rank" : c.rank,
             "suit" : c.suit, 
             "kopled" : c.kopled} for c in cards]
    
def deserialize_cards(cards : List[Dict]) -> List[Card]:
    """ Deserialize a list of dictionaries to a list of cards.
    """
    return [Card(c["suit"],c["rank"],c["kopled"]) for c in cards]