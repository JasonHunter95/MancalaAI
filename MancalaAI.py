from games import Game, GameState, alpha_beta_cutoff_search
import copy
from Mancala import Mancala
import numpy as np

class MancalaAI(Game):
    """Wrapper to use our Mancala version with the games.py functions for AI players."""
    
    def __init__(self, pits_per_player=6, stones_per_pit=4):
        self.pits_per_player = pits_per_player
        self.stones_per_pit = stones_per_pit
        # create an initial instance of the game
        initial_mancala = Mancala(pits_per_player, stones_per_pit)
        self.initial = GameState(
            to_move=1,
            utility=0,
            board=initial_mancala,
            moves=self.get_valid_moves(initial_mancala)
        )
    
    def get_valid_moves(self, mancala):
        """Gets a list of valid pits for the current player"""
        valid_moves = []
        for pit in range(1, self.pits_per_player + 1):
            if mancala.valid_move(pit):
                valid_moves.append(pit)
        return valid_moves
    
    def actions(self, state):
        """Returns list of legal moves"""
        return self.get_valid_moves(state.board) if not self.terminal_test(state) else []
    
    def result(self, state, move):
        """Returns new updated game state after making the given move"""
        # deep copy the board to avoid accidentally modifying the original state
        new_mancala = copy.deepcopy(state.board)
        new_mancala.play(move)
        return GameState(
            to_move=new_mancala.current_player,
            utility=self.calculate_utility(new_mancala),
            board=new_mancala,
            moves=self.get_valid_moves(new_mancala)
        )
        
    def utility(self, state, player):
        """Returns utility from perspective of given player.
        Player 1 is the maximizing player, Player 2 is minimizing."""
        mancala = state.board
        p1_stones = mancala.board[mancala.p1_mancala_index]
        p2_stones = mancala.board[mancala.p2_mancala_index]
        
        util = p1_stones - p2_stones
        return util if player == 1 else -util
    
    def calculate_utility(self, mancala):
        """Helper that calculates utility from Player 1's perspective."""
        return (mancala.board[mancala.p1_mancala_index] - 
                mancala.board[mancala.p2_mancala_index])
        
    def terminal_test(self, state):
        """Checks if game is over."""
        return state.board.winning_eval()
    
    def to_move(self, state):
        """Return the current player whose turn it is to move."""
        return state.to_move
    
    def display(self, state):
        """Display the board."""
        state.board.display_board()
            
def get_alpha_beta_minimax_move(mancala_game, state, plie_lim):
    """Gets the best move using alpha-beta pruning with a given number of plies"""
    def cutoff_test(state, current_plies):
        return current_plies > plie_lim or mancala_game.terminal_test(state)
    
    def eval_fn(state):
        return mancala_game.utility(state, mancala_game.to_move(mancala_game.initial))

    return alpha_beta_cutoff_search(state, mancala_game, d=plie_lim, 
                                    cutoff_test=cutoff_test, eval_fn=eval_fn)
    
def get_basic_minimax_move(mancala_game, state, plie_lim):
    """Minimax with depth cutoff but no alpha-beta pruning."""
    player = mancala_game.to_move(state)
    
    def max_value(state, depth):
        if depth > plie_lim or mancala_game.terminal_test(state):
            return mancala_game.utility(state, player)
        v = -np.inf
        for action in mancala_game.actions(state):
            v = max(v, min_value(mancala_game.result(state, action), depth + 1))
        return v
    
    def min_value(state, depth):
        if depth > plie_lim or mancala_game.terminal_test(state):
            return mancala_game.utility(state, player)
        v = np.inf
        for action in mancala_game.actions(state):
            v = min(v, max_value(mancala_game.result(state, action), depth + 1))
        return v
    
    # choose the action with the best minimax value
    best_action = None
    best_value = -np.inf
    for action in mancala_game.actions(state):
        value = min_value(mancala_game.result(state, action), 1)
        if value > best_value:
            best_value = value
            best_action = action
    
    return best_action