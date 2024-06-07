import random
import numpy as np

from PentobiGTP import PentobiGTP


class PentobiInternalPlayer:
    def __init__(self, pid, pentobi_sess):
        self.pid = pid
        self.pentobi_sess : PentobiGTP = pentobi_sess
        
    def play_move(self):
        mv = self.pentobi_sess.bot_get_move(self.pid)
        self.pentobi_sess.play_move(self.pid, mv, mock_move=False)
    
class PentobiInternalEpsilonGreedyPlayer:
    def __init__(self, pid, pentobi_sess, epsilon=0.1):
        self.pid = pid
        self.pentobi_sess : PentobiGTP = pentobi_sess
        self.epsilon = epsilon
        
    def play_move(self):
        if np.random.rand() < self.epsilon:
            all_moves = self.pentobi_sess.get_legal_moves(self.pid)
            selected_move = random.choice(all_moves)
            self.pentobi_sess.play_move(self.pid, selected_move, mock_move=False)
        else:
            mv = self.pentobi_sess.bot_get_move(self.pid)
            self.pentobi_sess.play_move(self.pid, mv, mock_move=False)
        return
    
class PentobiNNPlayer:
    def __init__(self, pid, pentobi_sess, model, move_selection_strategy="best", move_selection_kwargs={}):
        self.pid = pid
        self.pentobi_sess : PentobiGTP = pentobi_sess
        self.model = model
        self.move_selection_strategy = move_selection_strategy
        self.move_selection_kwargs = move_selection_kwargs
        
    def play_move(self):
        self.make_action_with_nn(self.move_selection_strategy, self.move_selection_kwargs)
        return
        
    def get_next_states(self):
        all_moves = self.pentobi_sess.get_legal_moves(self.pid)
        next_states = []
        for move in all_moves:
            if move == "pass":
                continue
            self.pentobi_sess.play_move(self.pid, move, mock_move=True)
            board_np = self.pentobi_sess.board_np.flatten()
            misc = np.array([self.pid-1, self.pid-1])
            next_states.append(np.concatenate([misc, board_np]))
            
            resp = self.pentobi_sess.send_command("undo")
            if "?" in resp:
                raise Exception(f"'undo' command failed")
            self.pentobi_sess.current_player = self.pid
        return all_moves, np.array(next_states, dtype=np.float32)
        
    def make_action_with_nn(self, move_selection_strategy="best", move_selection_kwargs={}):
        moves, next_states = self.get_next_states()
        if move_selection_strategy == "random":
            selected_move = random.choice(moves)
        elif move_selection_strategy == "epsilon_greedy":
            epsilon = move_selection_kwargs.get("epsilon", 0.1)
            if np.random.rand() < epsilon:
                selected_move = random.choice(moves)
            else:
                move_selection_strategy = "best"
                move_selection_kwargs = {}
        if len(moves) == 1 and moves[0] == "pass":
            return self.pentobi_sess.play_move(self.pid, "pass", mock_move=False)
        predictions = self.model.predict(next_states)
        #print(predictions,flush=True)
        if move_selection_strategy == "best":
            best_move = np.argmax(predictions)
            #print(f"Best move index: {best_move}")
            selected_move = moves[best_move]
        elif move_selection_strategy == "weighted":
            top_p = move_selection_kwargs.get("top_p", 1.0)
            psoftmax = np.exp(predictions) / np.sum(np.exp(predictions))
            psoftmax = psoftmax.flatten()
            #print(psoftmax,flush=True)
            # Sort from best to worst
            sort_idxs = np.argsort(psoftmax)[::-1]
            #print(sort_idxs,flush=True)
            psoftmax = psoftmax[sort_idxs]
            #print(moves,flush=True)
            moves = [moves[i] for i in sort_idxs]
            # Take the top p moves
            psoftmax_cumsum = np.cumsum(psoftmax)
            cutoff = psoftmax_cumsum >= top_p - 10**-6
            cutoff_idx = np.argmax(cutoff) + 1
            #print(f"cutoff idx: {cutoff_idx}", flush=True)
            psoftmax = psoftmax[:cutoff_idx]
            moves = moves[:cutoff_idx]
            # Renormalize
            psoftmax = psoftmax / np.sum(psoftmax)
            
            selected_move = np.random.choice(moves, p=psoftmax)
        #print(f"Selected move: {selected_move}")
        self.pentobi_sess.play_move(self.pid, selected_move)
        return