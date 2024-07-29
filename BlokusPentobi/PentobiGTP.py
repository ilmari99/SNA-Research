from collections import Counter
import os
import random
import signal
import subprocess
import threading
import time
from typing import List
import multiprocessing
import gc
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
        command = ' '.join(command)
        self.command = command
        #print(f"Starting pentobi-gtp with command: {command}")
        # Start the pentobi-gtp process in an invisible window
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            shell=True,
            #preexec_fn=os.setsid,
        )
        self.game_states = []
        
        self.current_player = 1
        # Lock to ensure thread-safe access to the process
        self.lock = multiprocessing.Lock()
        # Send showboard to get the initial board state
        test = self.send_command("showboard")
        # Check that the process is running, and what is the output
        if self.process.poll() is not None:
            raise ValueError(f"Error:  GTP process terminated with code {self.process.returncode}")
        
        
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
        #if self.is_game_finished():
        #    winner_idx = np.argmax(sc)
        #    sc[winner_idx] += 50
        return sc
    
    def write_states_to_file(self, filename, overwrite=False, use_discount=False, label_rank=False,append_mode=False):
        if not filename:
            raise ValueError("Filename is empty")
        states = np.array(self.game_states)
        #print(f"States shape: {states.shape}", flush=True)
        # To every state, append the score that was obtained by pid state[0]
        scores = self.score
        scores = np.array(scores)
        #winner_idx = np.argmax(scores)
        #if np.sum(scores == scores[winner_idx]) == 1:
        #    scores[winner_idx] += 50
        # Get the rank of the scores, i.e. the index of the scores in the sorted array. BUT so that
        # same scores have the same (lower) rank
        if label_rank:
            ranks = []
            for sc in scores:
                # E.g. if scores = [77,85,77,75] and sc = 77, then the rank is 3, since there are 3 scores
                # that are lower than or equal to 77
                rank = np.sum(scores >= sc)
                ranks.append(rank)
            scores = np.array(ranks) - 1
        else:
            winner_idx = np.argmax(scores)
            if np.sum(scores == scores[winner_idx]) == 1:
                scores[winner_idx] += 50
        # Get the score for the player whose perspective the state is from
        pids = states[:,0].astype(int)
        scores = scores[pids]
        if use_discount:
            assert not label_rank, "Can't use discount and label rank at the same time"
            for i, sc, pid in zip(range(len(scores)), scores, pids):
                num_played_moves = np.sum(pids[:i] == pid)
                # If we have noly played a few moves, we can't trust the score
                # so we want to discount the score if we have only played a few moves
                discount = np.min([1, (num_played_moves + 1)/15])
                scores[i] = round(sc * discount)
        
        states = np.column_stack([states, scores])
        # The states are in the format [pid, current_player, board, score] All values are integers
        # Save as csv
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if os.path.exists(filename) and not (overwrite or append_mode):
            raise FileExistsError(f"File {filename} already exists")
        #fmt = ["%d" for _ in range(states.shape[1] - 1)] + ["%f"]
        if append_mode:
            with open(filename, "ab") as f:
                np.savetxt(f, states, fmt="%d", delimiter=",")
        else:
            np.savetxt(filename, states, fmt="%d", delimiter=",")


    def send_command(self, command, errors="raise", lock = True):
        #print(f"Sending command '{command}'.")
        if lock:
            with self.lock:
                # Send the command to the process
                self.process.stdin.write(command + '\n')
                self.process.stdin.flush()
                # Read the response from the process
                response = self._read_response()
                if "?" in response and errors=="raise":
                    raise Exception(f"Command '{command}' failed: {response}")
        else:
            # Send the command to the process
            self.process.stdin.write(command + '\n')
            self.process.stdin.flush()
            # Read the response from the process
            response = self._read_response()
            if "?" in response and errors=="raise":
                raise Exception(f"Command '{command}' failed: {response}")
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
    
    def bot_get_move(self, pid, lock=True):
        if not self._check_pid_has_turn(pid):
            return False
        out = self.send_command(f"reg_genmove {pid}",errors="ignore",lock=lock)
        # = a1,b1 ...
        if "?" in out:
            return "pass"
        move = out.replace("= ", "")
        return move
    
    def play_move(self, pid, move, mock_move=False):
        """ Move is in the format a1,b1,a2, etc.
        """
        if not self._check_pid_has_turn(pid):
            raise ValueError(f"Player {pid} is not in turn!")
        if move != "pass":
            out = self.send_command(f"play {pid} {move}")
        self.change_player(pid)
        if not mock_move and move != "pass":
            misc = np.array([pid-1, self.current_player-1])
            board = self.board_np.flatten()
            self.game_states.append(np.concatenate([misc, board]))        
        return True
    
    def close(self):
        self.send_command("quit")
        self.process.communicate()
        self.process.terminate()
        self.process.wait()
    
    def get_legal_moves(self, pid): 
        out = self.send_command(f"all_legal {pid}")
        moves = out.replace("=", "").split("\n")
        moves = list(map(lambda mv : mv.strip(),filter(lambda x: x != "", moves)))
        #print(f"Found moves: {moves}")
        if len(moves) == 0:
            #print(f"Player {pid} has no legal moves")
            moves = ["pass"]
        return moves
    
    def is_game_finished(self):
        for pid in range(1,5):
            moves = self.get_legal_moves(pid)
            #print(f"Found {len(moves)} moves for pid {pid}: {moves}")
            #print(f"Moves: {moves}", flush=True)
            # If the response is empty, the player has no legal moves
            if len(moves) != 1 or moves[0] != "pass":
                return False
            else:
                pass
                #print(f"No moves left for player {pid}")
        return True
    
    def play_game(self, players):
        for player in players:
            player.play_move()
            
            
def random_playout(proc : PentobiGTP, state_file, start_pid):
    """ Play a random game starting from the current state.
    """
    proc.send_command("clear_board")
    proc.change_player(start_pid)
    # Set the board to the state
    proc.send_command("loadsgf " + state_file)
    # Play random moves until the game is finished
    while not proc.is_game_finished():
        pid = proc.current_player
        moves = proc.get_legal_moves(pid)
        move = random.choice(moves)
        proc.play_move(pid, move)
    return proc.score

