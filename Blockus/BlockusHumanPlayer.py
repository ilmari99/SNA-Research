from typing import Dict, List, Set, TYPE_CHECKING, Tuple
from RLFramework.Action import Action
from RLFramework.Game import Game
from RLFramework.Player import Player
import numpy as np
from BlockusPieces import BLOCKUS_PIECE_MAP

if TYPE_CHECKING:
    from BlockusGame import BlockusGame

from BlockusAction import BlockusAction
from BlockusPlayer import BlockusPlayer



class BlockusHumanPlayer(BlockusPlayer):
    def __init__(self, name : str = "BlockusHumanPlayer", logger_args : dict = None):
        super().__init__(name, logger_args)
        return
    
    def get_player_evals(self, game: 'BlockusGame') -> np.ndarray:
        """ Get the player evaluations.
        """
        assert game.model_paths, "A The game must have model paths."
        gs = game.get_current_state()
        evals = []
        for i in range(len(game.players)):
            gs_vector = gs.to_vector(i)
            model = game.get_model(game.model_paths[0])
            evaluation = model.predict(np.array([gs_vector], dtype=np.float32))
            evals.append(evaluation[0][0])
        return evals
            
    
    def choose_move(self, game: 'BlockusGame') -> 'BlockusAction':
        """ With Blokus, the human player chooses a move by clicking on tiles in the board, and pressing enter on the console.
        We then read which tiels were clicked, check if they are a valid selection, and return the action.
        """
        valid_actions = game.get_all_possible_actions()
        if len(valid_actions) == 1 and valid_actions[0].piece_id == -1:
            print("No valid actions.")
            return valid_actions[0]
        if len(valid_actions) < 10:
            print("Valid actions: ", valid_actions)
        # The game board
        fig = game.fig
        ax = game.ax
        board_ax = ax[0]
        pieces_ax = ax[1]
        # Receive graphical input from the user
        clicked_tiles = []
        def onclick(event):
            # Check that the click is in either ax
            if event.inaxes not in [board_ax, pieces_ax]:
                return
            # The board is a 20x20 grid
            x = event.xdata + 0.5
            y = event.ydata + 0.5
            print(f"Clicked at x: {x}, y: {y} ",end="")
            row = int(y)
            col = int(x)
            print(f"Row: {row}, col: {col}")
            
            # If the click is in the pieces ax, we go to the start of the last 4x4 grid
            # and rotate the 4x4 grid 90 degrees clockwise
            if event.inaxes == pieces_ax:
                print("Rotating piece.")
                # Go back rows until we are at the start of a 4x4 grid
                while row % 4 != 0:
                    row -= 1
                # Go back columns until we are at the start of a 4x4 grid
                while col % 4 != 0:
                    col -= 1
                # Now, find the piece that was clicked
                piece_idx = row // 4 * 5 + col // 4
                piece_id = game.player_remaining_pieces[game.current_pid][piece_idx]
                piece_arr = BLOCKUS_PIECE_MAP[piece_id]
                piece_arr = np.array(piece_arr)
                # Rotate the piece 90 degrees clockwise
                piece_arr = np.rot90(piece_arr)
                # Convert to a list
                #piece_arr = piece_arr.tolist()
                #print(f"New piece array: {piece_arr}")
                BLOCKUS_PIECE_MAP[piece_id] = piece_arr
                game.render()
                return
            clicked_tiles.append((row, col))
            # Draw the clicked tile
            board_ax.plot(col, row, 'ro')
            fig.canvas.draw()
            return
        
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        
        input("Press enter to confirm selection.")
        
        fig.canvas.mpl_disconnect(cid)
        print("Clicked tiles: ", clicked_tiles)
        
        # Remove the dots
        board_ax.clear()
        game.render()
        
        
        
        # Check if the selection is valid
        valid_grid_selections = [action.get_piece_coordinates() for action in valid_actions]
        clicked_tiles_set = set(clicked_tiles)
        valid_action = None
        for idx, valid_grids in enumerate(valid_grid_selections):
            if set(valid_grids) == clicked_tiles_set:
                valid_action = valid_actions[idx]
                break
        if not valid_action:
            print("Invalid selection.")
            return self.choose_move(game)
        return valid_action