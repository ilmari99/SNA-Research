


CARD_VALUES = tuple(range(2,15))                            # Initialize the standard deck
CARD_SUITS = ("C","D","H","S") 
CARD_SUIT_SYMBOLS = {"S":'♠', "D":'♦',"H": '♥',"C": '♣',"X":"X"}    #Conversion table
USING_CARD_SYMBOLS = True

class Card:
    """ A class representing a card.
    """
    def __init__(self, suit : str, rank : int, kopled = False):
        self.suit = suit
        self.rank = rank
        self.kopled = kopled
        
    def __eq__(self, other) -> bool:
        return self.suit == other.suit and self.rank == other.rank
    
    def __repr__(self) -> str:
        # If we are using card symbols, then print the card with the symbol
        if USING_CARD_SYMBOLS:
            return f"{CARD_SUIT_SYMBOLS[self.suit]}{self.rank}"
        # If we have a different encoding, then print the suit as a letter
        return f"{self.suit}{self.rank}"
    
    def __hash__(self) -> int:
        return hash((self.suit, self.rank))
    
    def __lt__(self, other) -> bool:
        return self.rank < other.rank
    
    def __eq__(self, other) -> bool:
        return self.rank == other.rank and self.suit == other.suit