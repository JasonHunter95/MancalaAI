"""
MancalaAI - A Mancala game implementation with Minimax and Alpha-Beta Pruning AI.

This package provides:
- A complete Mancala game engine with standard rules
- AI players using Minimax and Alpha-Beta Pruning algorithms
- CLI for playing against the AI or running benchmarks
"""

from mancala_ai.game import MancalaGame
from mancala_ai.ai import MancalaAI, get_alpha_beta_move, get_minimax_move

__version__ = "1.0.0"
__all__ = [
    "MancalaGame",
    "MancalaAI",
    "get_alpha_beta_move",
    "get_minimax_move",
]
