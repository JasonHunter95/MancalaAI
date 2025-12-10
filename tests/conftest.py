"""
Pytest configuration and fixtures for MancalaAI tests.
"""

import random

import pytest

from mancala_ai.game import MancalaGame
from mancala_ai.ai import MancalaAI


@pytest.fixture
def default_game() -> MancalaGame:
    """Create a standard 6-pit, 4-stone game."""
    return MancalaGame(pits_per_player=6, stones_per_pit=4)


@pytest.fixture
def seeded_game() -> MancalaGame:
    """Create a game with a seeded RNG for reproducibility."""
    rng = random.Random(42)
    return MancalaGame(pits_per_player=6, stones_per_pit=4, rng=rng)


@pytest.fixture
def small_game() -> MancalaGame:
    """Create a smaller game for faster tests."""
    return MancalaGame(pits_per_player=3, stones_per_pit=2)


@pytest.fixture
def default_ai() -> MancalaAI:
    """Create a standard AI game wrapper."""
    return MancalaAI(pits_per_player=6, stones_per_pit=4)


@pytest.fixture
def small_ai() -> MancalaAI:
    """Create a smaller AI game for faster tests."""
    return MancalaAI(pits_per_player=3, stones_per_pit=2)
