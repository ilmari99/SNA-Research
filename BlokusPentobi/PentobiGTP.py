from collections import Counter
import os
import random
import subprocess
import threading
import time
from typing import List
import multiprocessing

import numpy as np
import tensorflow as tf

def parse_gtp_board_to_matrix(board):
    """
    20 X X . X O O O O # # # # X X X X X . . O  Blue(X): 82!
    19 . X . X X . . # O . X X . . O O O X . O  I3 I4
    18 . X X # X . . # O . X X O . O X X O O O
    17 . # # X # # # . O X . . O O . O O X X X
    16 # . # X X # # . O X X X O . O O . O O X  Yellow(O): 82!
    15 # . # . X X . . O X O . O . O . . O . .  I3 Z4
    14 # # . X . . X X X O O . . O . O O . O O
    13 . # X X X X # # X O . O O O X . O O . O
    12 # X # # . # . # X O . O . . X X O X O O  Red(#): 77!
    11 # X # . # # # . # X X X . X X O X X X .  O T4 Z4
    10 # X # # @ # X # # # # X X>O . O . X O O
    9 @ X X @ @ @ X X @ . @ # . X O O O . O O
    8 @ @ @ X @ X @ @ @ . @ # X X X . . O . .  Green(@): 61!
    7 @ . . X X X @ X X @ @ # # # O O O O . .  I3 I4 L4 O P V3 Y
    6 . @ @ . @ @ X X X @ # @ @ O # # . . O .
    5 @ @ . @ @ # @ @ # # # @ O O @ # # O O .
    4 . . . @ . # # # . # . @ @ O @ @ # O O .
    3 @ @ @ . . # @ @ @ @ @ . O @ @ . @ # . .
    2 @ # # # # @ # # # # # O O O . @ @ # # #
    1 @ # . . . @ @ @ @ O O . O . @ . @ . . #
    A B C D E F G H I J K L M N O P Q R S T
    """
    # Substitute > or < with a space
    board = board.replace(">", " ").replace("<", " ")
    board_in_lines = board.split("\n")
    board_in_lines = board_in_lines[1:-1]
    board_in_lines_splitted = [line.split(" ") for line in board_in_lines]
    # remove empty elements
    board_in_lines_splitted = [list(filter(lambda x: x != "", line)) for line in board_in_lines_splitted]
    #print(board_in_lines_splitted)

    # Skip the row number, and only take the first 20 columns
    for i in range(0, len(board_in_lines_splitted)):
        #print(board_in_lines_splitted[i])
        board_in_lines_splitted[i] = board_in_lines_splitted[i][1:21]
        #print(board_in_lines_splitted[i])
    # Remove the last row
    #board_in_lines_splitted = board_in_lines_splitted[:-1]
    #print(board_in_lines_splitted)
    conversion_map = {
        "." : -1,
        "X" : 0,
        "O" : 1,
        "#" : 2,
        "@" : 3,
        "+" : -1,
    }
    board_matrix = []
    for line in board_in_lines_splitted:
        board_matrix.append([conversion_map[x] for x in line])
    return np.array(board_matrix)
    

class PentobiGTP:
    def __init__(self, command=None,
                 book=None,
                 config=None,
                 game="classic",
                 level=1,
                 seed=None,
                 showboard=False,
                 nobook=False,
                 noresign=True,
                 quiet=False,
                 threads=1,
                 ):
        if command is None:
            command = os.environ.get("PENTOBI_GTP")
            if command is None:
                command = self._find_pentobi_gtp_binary()
                if command is None:
                    raise ValueError("Pentobi GTP binary not found")
                
        # Build the command to start the pentobi-gtp process
        command = [command]
        if book:
            command.append(f'--book {book}')
        if config:
            command.append(f'--config {config}')
        command.append(f'--game {game}')
        command.append(f'--level {level}')
        if seed:
            command.append(f'--seed {seed}')
        if showboard:
            command.append('--showboard')
        if nobook:
            command.append('--nobook')
        if noresign:
            command.append('--noresign')
        if quiet:
            command.append('--quiet')
        command.append(f'--threads {threads}')
        #print(f"Starting pentobi-gtp with command: {command}")
        command = ' '.join(command)
        # Start the pentobi-gtp process in an invisible window
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            shell=True,
        )
        self.game_states = []
        
        self.current_player = 1
        # Lock to ensure thread-safe access to the process
        self.lock = threading.Lock()
        
    def _find_pentobi_gtp_binary(self):
        # From the currect directory, search for the pentobi-gtp binary
        current_dir = os.path.dirname(os.path.realpath(__file__))
        for root, dirs, files in os.walk(current_dir):
            for file in files:
                if file == "pentobi-gtp":
                    pent_gtp = os.path.join(root, file)
                    os.environ["PENTOBI_GTP"] = pent_gtp
        return None
        
    @property
    def pid(self):
        return self.current_player
    
    @property
    def board(self):
        return self.send_command("showboard")
    
    @property
    def board_np(self):
        return parse_gtp_board_to_matrix(self.board)
    
    @property
    def score(self):
        sc = self.send_command("final_score")
        #= 85 81 77 65
        sc = sc.split(" ")
        sc = [int(x) for x in sc[1:]]
        return sc
    
        
    def close(self):
        self.send_command("quit")
        self.process.terminate()
        self.process.wait()
    
    def write_states_to_file(self, filename, overwrite=False):
        if not filename:
            raise ValueError("Filename is empty")
        states = np.array(self.game_states)
        # To every state, append the score that was obtained by pid state[0]
        scores = self.score
        scores = np.array(scores)
        # Get the score for the player whose perspective the state is from
        scores = scores[states[:,0].astype(int)]
        states = np.column_stack([states, scores])
        # The states are in the format [pid, current_player, board, score] All values are integers
        # Save as csv
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if os.path.exists(filename) and not overwrite:
            raise FileExistsError(f"File {filename} already exists")
        np.savetxt(filename, states, fmt="%d", delimiter=",")

    def send_command(self, command):
        with self.lock:
            # Send the command to the process
            self.process.stdin.write(command + '\n')
            self.process.stdin.flush()
            # Read the response from the process
            response = self._read_response()
        return response

    def _read_response(self):
        response = []
        while True:
            line = self.process.stdout.readline().strip()
            if line == '':
                break
            response.append(line)
        return '\n'.join(response)
    
    def _check_pid_has_turn(self,pid):
        if pid != self.current_player:
            print(f"Error: Player {pid} is not the current player")
            return False
        return True
    
    def change_player(self, pid):
        self.current_player = (pid % 4) + 1
    
    def final_score(self):
        return self.send_command("final_score")
    
    def bot_get_move(self, pid):
        if not self._check_pid_has_turn(pid):
            return False
        out = self.send_command(f"reg_genmove {pid}")
        # = a1,b1 ...
        move = out.split("=")
        move = move[1]
        return move
    
    def play_move(self, pid, move, mock_move=False):
        """ Move is in the format a1,b1,a2, etc.
        """
        if not self._check_pid_has_turn(pid):
            return False
        out = self.send_command(f"play {pid} {move}")
        self.change_player(pid)
        if not mock_move:
            misc = np.array([pid-1, self.current_player-1])
            board = self.board_np.flatten()
            self.game_states.append(np.concatenate([misc, board]))        
        return True
    
    def get_legal_moves(self, pid): 
        out = self.send_command(f"all_legal {pid}")
        moves = out.replace("=", "").split("\n")
        moves = list(map(lambda mv : mv.strip(),filter(lambda x: x != "", moves)))
        if len(moves) == 0:
            #print(f"Player {pid} has no legal moves")
            moves = ["pass"]
        return moves
    
    def is_game_finished(self):
        for pid in range(1,5):
            out = self.send_command(f"all_legal {pid}")
            moves = out.split("=")
            moves = list(filter(lambda x: x != "", moves))
            # If the response is empty, the player has no legal moves
            if len(moves) > 0:
                return False
        return True
    
    def play_game(self, players):
        for player in players:
            player.play_move()