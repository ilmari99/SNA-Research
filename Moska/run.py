from MoskaGame import MoskaGame
from MoskaPlayer import MoskaPlayer

players = [MoskaPlayer("Player 1"), MoskaPlayer("Player 2")]
game = MoskaGame()
game.play_game(players)