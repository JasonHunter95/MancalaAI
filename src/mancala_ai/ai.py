"""
Mancala AI Module.

This module provides AI players for Mancala using:
- Minimax algorithm with depth cutoff
- Alpha-Beta Pruning optimization

The AI components are based on AIMA (Artificial Intelligence: A Modern Approach)
game-playing algorithms.
"""

from __future__ import annotations

import copy
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from mancala_ai.game import MancalaGame


# Game state representation for search algorithms
GameState = namedtuple("GameState", ["to_move", "utility", "board", "moves"])


class Game:
    """
    Abstract base class for adversarial search games.

    Subclass this and implement actions, result, utility, and terminal_test.
    Based on AIMA Figure 5.1.
    """

    def actions(self, state: GameState) -> list[Any]:
        """Return a list of allowable moves at this state."""
        raise NotImplementedError

    def result(self, state: GameState, move: Any) -> GameState:
        """Return the state that results from making a move."""
        raise NotImplementedError

    def utility(self, state: GameState, player: int) -> float:
        """Return the value of this final state to player."""
        raise NotImplementedError

    def terminal_test(self, state: GameState) -> bool:
        """Return True if this is a final state for the game."""
        return not self.actions(state)

    def to_move(self, state: GameState) -> int:
        """Return the player whose move it is in this state."""
        return state.to_move

    def display(self, state: GameState) -> None:
        """Print or display the state."""
        print(state)


def alpha_beta_search(
    state: GameState,
    game: Game,
    depth: int = 4,
    cutoff_test: Optional[Callable[[GameState, int], bool]] = None,
    eval_fn: Optional[Callable[[GameState], float]] = None,
) -> Optional[Any]:
    """
    Search game tree using alpha-beta pruning with depth cutoff.

    Based on AIMA Figure 5.7, extended with depth cutoff.

    Args:
        state: The current game state.
        game: The game being played.
        depth: Maximum search depth (plies).
        cutoff_test: Optional function to test if search should stop.
                     Receives (state, current_depth), returns bool.
        eval_fn: Optional evaluation function for non-terminal states.
                 Receives state, returns utility estimate.

    Returns:
        The best action found, or None if no actions available.
    """
    player = game.to_move(state)

    # Default cutoff and evaluation
    if cutoff_test is None:
        cutoff_test = lambda s, d: d >= depth or game.terminal_test(s)
    if eval_fn is None:
        eval_fn = lambda s: game.utility(s, player)

    def max_value(state: GameState, alpha: float, beta: float, current_depth: int) -> float:
        if cutoff_test(state, current_depth):
            return eval_fn(state)
        v = -np.inf
        for action in game.actions(state):
            v = max(v, min_value(game.result(state, action), alpha, beta, current_depth + 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state: GameState, alpha: float, beta: float, current_depth: int) -> float:
        if cutoff_test(state, current_depth):
            return eval_fn(state)
        v = np.inf
        for action in game.actions(state):
            v = min(v, max_value(game.result(state, action), alpha, beta, current_depth + 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Find best action
    best_score = -np.inf
    beta = np.inf
    best_action = None

    for action in game.actions(state):
        v = min_value(game.result(state, action), best_score, beta, 1)
        if v > best_score:
            best_score = v
            best_action = action

    return best_action


def minimax_search(
    state: GameState,
    game: Game,
    depth: int = 4,
) -> Optional[Any]:
    """
    Minimax search with depth cutoff (no alpha-beta pruning).

    This is included for comparison and educational purposes.
    Alpha-beta pruning should be preferred for actual use.

    Args:
        state: The current game state.
        game: The game being played.
        depth: Maximum search depth (plies).

    Returns:
        The best action found, or None if no actions available.
    """
    player = game.to_move(state)

    def max_value(state: GameState, current_depth: int) -> float:
        if current_depth >= depth or game.terminal_test(state):
            return game.utility(state, player)
        v = -np.inf
        for action in game.actions(state):
            v = max(v, min_value(game.result(state, action), current_depth + 1))
        return v

    def min_value(state: GameState, current_depth: int) -> float:
        if current_depth >= depth or game.terminal_test(state):
            return game.utility(state, player)
        v = np.inf
        for action in game.actions(state):
            v = min(v, max_value(game.result(state, action), current_depth + 1))
        return v

    # Find best action
    best_action = None
    best_value = -np.inf

    for action in game.actions(state):
        value = min_value(game.result(state, action), 1)
        if value > best_value:
            best_value = value
            best_action = action

    return best_action


@dataclass
class MancalaAI(Game):
    """
    Mancala game wrapper for use with adversarial search algorithms.

    This class adapts the MancalaGame to work with AIMA-style game search
    algorithms like minimax and alpha-beta pruning.

    Args:
        pits_per_player: Number of pits on each side. Defaults to 6.
        stones_per_pit: Initial stones in each pit. Defaults to 4.

    Attributes:
        initial: The initial game state for search algorithms.
    """

    pits_per_player: int = 6
    stones_per_pit: int = 4
    initial: GameState = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the game and create the initial state."""
        game = MancalaGame(
            pits_per_player=self.pits_per_player,
            stones_per_pit=self.stones_per_pit,
        )
        self.initial = GameState(
            to_move=1,
            utility=0,
            board=game,
            moves=game.get_valid_moves(),
        )

    def actions(self, state: GameState) -> list[int]:
        """
        Get valid moves for the current state.

        Args:
            state: The current game state.

        Returns:
            List of valid pit numbers (1-indexed).
        """
        if self.terminal_test(state):
            return []
        return state.board.get_valid_moves()

    def result(self, state: GameState, move: int) -> GameState:
        """
        Compute the resulting state after a move.

        Args:
            state: The current game state.
            move: The pit number to play (1-indexed).

        Returns:
            The new game state after the move.
        """
        # Deep copy to avoid modifying the original
        new_game = state.board.copy()
        new_game.play(move)

        return GameState(
            to_move=new_game.current_player,
            utility=self._calculate_utility(new_game),
            board=new_game,
            moves=new_game.get_valid_moves(),
        )

    def utility(self, state: GameState, player: int) -> float:
        """
        Compute utility from a player's perspective.

        Player 1 is the maximizing player. The utility is the difference
        in mancala scores (positive favors Player 1).

        Args:
            state: The current game state.
            player: The player to evaluate for (1 or 2).

        Returns:
            Utility value from the player's perspective.
        """
        game = state.board
        util = game.p1_score - game.p2_score
        return util if player == 1 else -util

    def _calculate_utility(self, game: MancalaGame) -> float:
        """Helper to calculate utility from Player 1's perspective."""
        return game.p1_score - game.p2_score

    def terminal_test(self, state: GameState) -> bool:
        """
        Check if the game is over.

        Args:
            state: The current game state.

        Returns:
            True if the game has ended, False otherwise.
        """
        return state.board.is_terminal()

    def to_move(self, state: GameState) -> int:
        """
        Get the current player.

        Args:
            state: The current game state.

        Returns:
            The current player (1 or 2).
        """
        return state.to_move

    def display(self, state: GameState) -> None:
        """
        Display the game board.

        Args:
            state: The current game state to display.
        """
        print(state.board.render_board())


def get_alpha_beta_move(
    ai: MancalaAI,
    state: GameState,
    depth: int,
) -> Optional[int]:
    """
    Get the best move using alpha-beta pruning.

    Args:
        ai: The MancalaAI game instance.
        state: The current game state.
        depth: Search depth in plies.

    Returns:
        The best pit number to play, or None if no moves available.
    """
    # The player initiating the search (always maximizing from their perspective)
    root_player = ai.to_move(state)

    def cutoff_test(state: GameState, current_depth: int) -> bool:
        return current_depth >= depth or ai.terminal_test(state)

    def eval_fn(state: GameState) -> float:
        # Always evaluate from the root player's perspective
        return ai.utility(state, root_player)

    return alpha_beta_search(state, ai, depth=depth, cutoff_test=cutoff_test, eval_fn=eval_fn)


def get_minimax_move(
    ai: MancalaAI,
    state: GameState,
    depth: int,
) -> Optional[int]:
    """
    Get the best move using basic minimax (no pruning).

    This is provided for comparison and educational purposes.
    Alpha-beta should be preferred for performance.

    Args:
        ai: The MancalaAI game instance.
        state: The current game state.
        depth: Search depth in plies.

    Returns:
        The best pit number to play, or None if no moves available.
    """
    return minimax_search(state, ai, depth=depth)
