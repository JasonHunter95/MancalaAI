"""
Mancala Game Engine.

This module provides a complete implementation of the Mancala board game,
following standard rules including stone distribution, capturing, and
extra turn mechanics.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MancalaGame:
    """
    A complete Mancala game implementation.

    The board is represented as a list of integers where:
    - Indices 0 to pits_per_player-1 are Player 1's pits
    - Index pits_per_player is Player 1's mancala (store)
    - Indices pits_per_player+1 to 2*pits_per_player are Player 2's pits
    - Index 2*pits_per_player+1 is Player 2's mancala (store)

    Args:
        pits_per_player: Number of pits on each side. Defaults to 6.
        stones_per_pit: Initial stones in each pit. Defaults to 4.
        rng: Random number generator for reproducibility. If None, uses
             the global random module.

    Attributes:
        board: The game board as a list of stone counts.
        current_player: The player whose turn it is (1 or 2).
        moves_history: List of (player, pit) tuples recording all moves.
    """

    pits_per_player: int = 6
    stones_per_pit: int = 4
    rng: Optional[random.Random] = field(default=None, repr=False)

    # Computed fields
    board: list[int] = field(init=False, repr=False)
    current_player: int = field(init=False, default=1)
    moves_history: list[tuple[int, int]] = field(init=False, default_factory=list)

    # Index cache
    _p1_pits_start: int = field(init=False, repr=False)
    _p1_pits_end: int = field(init=False, repr=False)
    _p1_mancala: int = field(init=False, repr=False)
    _p2_pits_start: int = field(init=False, repr=False)
    _p2_pits_end: int = field(init=False, repr=False)
    _p2_mancala: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the board and compute index positions."""
        # Validate inputs
        if self.pits_per_player < 1:
            raise ValueError("pits_per_player must be at least 1")
        if self.stones_per_pit < 0:
            raise ValueError("stones_per_pit cannot be negative")

        # Initialize board
        board_size = (self.pits_per_player + 1) * 2
        self.board = [self.stones_per_pit] * board_size

        # Compute indices
        self._p1_pits_start = 0
        self._p1_pits_end = self.pits_per_player - 1
        self._p1_mancala = self.pits_per_player
        self._p2_pits_start = self.pits_per_player + 1
        self._p2_pits_end = board_size - 2
        self._p2_mancala = board_size - 1

        # Zero out mancalas
        self.board[self._p1_mancala] = 0
        self.board[self._p2_mancala] = 0

    def copy(self) -> MancalaGame:
        """Create a deep copy of the game state."""
        return copy.deepcopy(self)

    @property
    def p1_score(self) -> int:
        """Player 1's current score (stones in mancala)."""
        return self.board[self._p1_mancala]

    @property
    def p2_score(self) -> int:
        """Player 2's current score (stones in mancala)."""
        return self.board[self._p2_mancala]

    @property
    def p1_pits(self) -> list[int]:
        """List of stone counts in Player 1's pits."""
        return self.board[self._p1_pits_start : self._p1_pits_end + 1]

    @property
    def p2_pits(self) -> list[int]:
        """List of stone counts in Player 2's pits."""
        return self.board[self._p2_pits_start : self._p2_pits_end + 1]

    def is_valid_move(self, pit: int) -> bool:
        """
        Check if a move is valid for the current player.

        Args:
            pit: 1-indexed pit number (1 to pits_per_player).

        Returns:
            True if the move is valid, False otherwise.
        """
        if not 1 <= pit <= self.pits_per_player:
            return False

        if self.current_player == 1:
            pit_index = self._p1_pits_start + (pit - 1)
        else:
            pit_index = self._p2_pits_start + (pit - 1)

        return self.board[pit_index] > 0

    def get_valid_moves(self) -> list[int]:
        """
        Get all valid moves for the current player.

        Returns:
            List of valid pit numbers (1-indexed).
        """
        return [pit for pit in range(1, self.pits_per_player + 1) if self.is_valid_move(pit)]

    def get_random_move(self) -> Optional[int]:
        """
        Get a random valid move for the current player.

        Returns:
            A random valid pit number, or None if no moves available.
        """
        valid_moves = self.get_valid_moves()
        if not valid_moves:
            return None

        if self.rng is not None:
            return self.rng.choice(valid_moves)
        return random.choice(valid_moves)

    def is_terminal(self) -> bool:
        """
        Check if the game has ended.

        The game ends when either player's pits are all empty.

        Returns:
            True if the game is over, False otherwise.
        """
        p1_empty = sum(self.p1_pits) == 0
        p2_empty = sum(self.p2_pits) == 0
        return p1_empty or p2_empty

    def get_winner(self) -> Optional[int]:
        """
        Determine the winner of a finished game.

        Returns:
            1 if Player 1 wins, 2 if Player 2 wins, 0 for tie,
            or None if game is not over.
        """
        if not self.is_terminal():
            return None

        if self.p1_score > self.p2_score:
            return 1
        elif self.p2_score > self.p1_score:
            return 2
        return 0

    def _get_opposite_pit(self, pit_index: int) -> int:
        """Calculate the index of the pit directly opposite."""
        return 2 * self.pits_per_player - pit_index

    def _is_on_player_side(self, pit_index: int, player: int) -> bool:
        """Check if a pit index is on the given player's side."""
        if player == 1:
            return self._p1_pits_start <= pit_index <= self._p1_pits_end
        return self._p2_pits_start <= pit_index <= self._p2_pits_end

    def _distribute_stones(self, pit_index: int, player_mancala: int, opponent_mancala: int) -> int:
        """
        Distribute stones from a pit following Mancala rules.

        Args:
            pit_index: The board index of the pit to sow from.
            player_mancala: Index of the current player's mancala.
            opponent_mancala: Index of the opponent's mancala.

        Returns:
            The final index where the last stone was placed.
        """
        board_size = len(self.board)
        stones = self.board[pit_index]
        self.board[pit_index] = 0

        current_index = pit_index
        while stones > 0:
            current_index = (current_index + 1) % board_size

            # Skip opponent's mancala
            if current_index == opponent_mancala:
                continue

            self.board[current_index] += 1
            stones -= 1

        return current_index

    def _sweep_remaining_stones(self) -> None:
        """Sweep all remaining stones to the respective mancalas when game ends."""
        # Sweep Player 1's pits
        for i in range(self._p1_pits_start, self._p1_pits_end + 1):
            self.board[self._p1_mancala] += self.board[i]
            self.board[i] = 0

        # Sweep Player 2's pits
        for i in range(self._p2_pits_start, self._p2_pits_end + 1):
            self.board[self._p2_mancala] += self.board[i]
            self.board[i] = 0

    def play(self, pit: int) -> bool:
        """
        Execute a move for the current player.

        Args:
            pit: 1-indexed pit number (1 to pits_per_player).

        Returns:
            True if the move was successful, False if invalid or game over.

        Raises:
            ValueError: If the move is invalid.
        """
        if self.is_terminal():
            raise ValueError("Game is already over")

        if not self.is_valid_move(pit):
            raise ValueError(f"Invalid move: pit {pit}")

        # Record the move
        self.moves_history.append((self.current_player, pit))

        # Get indices for current player
        if self.current_player == 1:
            pit_index = self._p1_pits_start + (pit - 1)
            player_mancala = self._p1_mancala
            opponent_mancala = self._p2_mancala
            player_pits = (self._p1_pits_start, self._p1_pits_end)
        else:
            pit_index = self._p2_pits_start + (pit - 1)
            player_mancala = self._p2_mancala
            opponent_mancala = self._p1_mancala
            player_pits = (self._p2_pits_start, self._p2_pits_end)

        # Distribute stones
        final_index = self._distribute_stones(pit_index, player_mancala, opponent_mancala)

        # Check for capture: last stone lands in empty pit on player's side
        if (
            self._is_on_player_side(final_index, self.current_player)
            and self.board[final_index] == 1
        ):
            opposite_index = self._get_opposite_pit(final_index)
            if 0 <= opposite_index < len(self.board) and self.board[opposite_index] > 0:
                # Capture opposite stones plus the capturing stone
                captured = self.board[opposite_index] + 1
                self.board[opposite_index] = 0
                self.board[final_index] = 0
                self.board[player_mancala] += captured

        # Check for game end and sweep remaining stones
        if sum(self.board[player_pits[0] : player_pits[1] + 1]) == 0 or self.is_terminal():
            self._sweep_remaining_stones()

        # Check for extra turn (last stone in player's mancala)
        if final_index != player_mancala and not self.is_terminal():
            self.current_player = 2 if self.current_player == 1 else 1

        return True

    def render_board(self) -> str:
        """
        Generate an ASCII representation of the board.

        The board is displayed horizontally with:
        - P2's mancala on the left
        - P2's pits on top (numbered 6->1 from left to right)
        - P1's pits on bottom (numbered 1->6 from left to right)
        - P1's mancala on the right

        Returns:
            A multi-line string showing the current board state.
        """
        # Calculate cell width (need enough for 2-digit numbers)
        cell_width = 4
        pit_section_width = self.pits_per_player * cell_width

        lines = []

        # P2's pit labels (reversed: 6, 5, 4, 3, 2, 1)
        p2_labels = "     "  # Left mancala space
        for i in range(self.pits_per_player, 0, -1):
            p2_labels += f"{i:^{cell_width}}"
        lines.append(p2_labels)

        # Top border
        top_border = "┌───┬" + "─" * pit_section_width + "┬───┐"
        lines.append(top_border)

        # P2's pits row (reversed order for display)
        p2_pits_str = ""
        for stone in reversed(self.p2_pits):
            p2_pits_str += f"{stone:^{cell_width}}"
        p2_row = f"│{self.p2_score:^3}│{p2_pits_str}│   │"
        lines.append(p2_row)

        # Middle separator with labels
        mid_border = "│ P2├" + "─" * pit_section_width + "┤P1 │"
        lines.append(mid_border)

        # P1's pits row
        p1_pits_str = ""
        for stone in self.p1_pits:
            p1_pits_str += f"{stone:^{cell_width}}"
        p1_row = f"│   │{p1_pits_str}│{self.p1_score:^3}│"
        lines.append(p1_row)

        # Bottom border
        bottom_border = "└───┴" + "─" * pit_section_width + "┴───┘"
        lines.append(bottom_border)

        # P1's pit labels
        p1_labels = "     "  # Left mancala space
        for i in range(1, self.pits_per_player + 1):
            p1_labels += f"{i:^{cell_width}}"
        lines.append(p1_labels)

        # Current player indicator or game over message
        if not self.is_terminal():
            lines.append(f"\n    Player {self.current_player}'s turn")
        else:
            winner = self.get_winner()
            if winner == 0:
                lines.append("\n    Game Over - It's a tie!")
            else:
                lines.append(f"\n    Game Over - Player {winner} wins!")

        return "\n".join(lines)

    def __str__(self) -> str:
        """String representation shows the board."""
        return self.render_board()
