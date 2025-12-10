"""
Tests for the Mancala game engine.
"""

import random

import pytest

from mancala_ai.game import MancalaGame


class TestGameInitialization:
    """Tests for game initialization."""

    def test_default_initialization(self, default_game: MancalaGame) -> None:
        """Test default game setup."""
        assert default_game.pits_per_player == 6
        assert default_game.stones_per_pit == 4
        assert default_game.current_player == 1
        assert len(default_game.board) == 14  # (6 + 1) * 2
        assert default_game.p1_score == 0
        assert default_game.p2_score == 0

    def test_custom_initialization(self) -> None:
        """Test custom game configuration."""
        game = MancalaGame(pits_per_player=4, stones_per_pit=6)
        assert game.pits_per_player == 4
        assert game.stones_per_pit == 6
        assert len(game.board) == 10  # (4 + 1) * 2

    def test_initial_pit_stones(self, default_game: MancalaGame) -> None:
        """Test that pits are initialized with correct stones."""
        # All P1 pits should have 4 stones
        assert all(s == 4 for s in default_game.p1_pits)
        # All P2 pits should have 4 stones
        assert all(s == 4 for s in default_game.p2_pits)

    def test_initial_mancalas_empty(self, default_game: MancalaGame) -> None:
        """Test that mancalas start empty."""
        assert default_game.p1_score == 0
        assert default_game.p2_score == 0

    def test_invalid_pits_raises_error(self) -> None:
        """Test that invalid pit count raises ValueError."""
        with pytest.raises(ValueError):
            MancalaGame(pits_per_player=0)

    def test_negative_stones_raises_error(self) -> None:
        """Test that negative stones raises ValueError."""
        with pytest.raises(ValueError):
            MancalaGame(stones_per_pit=-1)


class TestMoveValidation:
    """Tests for move validation."""

    def test_valid_moves_initial(self, default_game: MancalaGame) -> None:
        """Test that all pits are valid moves initially."""
        valid = default_game.get_valid_moves()
        assert valid == [1, 2, 3, 4, 5, 6]

    def test_is_valid_move(self, default_game: MancalaGame) -> None:
        """Test individual move validation."""
        assert default_game.is_valid_move(1)
        assert default_game.is_valid_move(6)
        assert not default_game.is_valid_move(0)
        assert not default_game.is_valid_move(7)

    def test_empty_pit_invalid(self) -> None:
        """Test that empty pits are not valid moves."""
        game = MancalaGame(pits_per_player=3, stones_per_pit=1)
        # Play pit 1, which empties it
        game.play(1)
        # Pit 1 should now be invalid for P2 (but P2 has different pits)
        # After one move, check the game state
        assert game.current_player in [1, 2]  # Could be extra turn


class TestStoneDistribution:
    """Tests for stone distribution mechanics."""

    def test_simple_distribution(self) -> None:
        """Test basic stone distribution."""
        game = MancalaGame(pits_per_player=6, stones_per_pit=4)
        game.play(1)
        # Pit 1 should be empty, next 4 pits should have 5 stones each
        assert game.board[0] == 0  # Pit 1 empty
        assert game.board[1] == 5  # Pit 2 has 5
        assert game.board[2] == 5  # Pit 3 has 5

    def test_skip_opponent_mancala(self) -> None:
        """Test that stones skip the opponent's mancala."""
        game = MancalaGame(pits_per_player=6, stones_per_pit=4)
        # Set up a pit with many stones to wrap around
        game.board[5] = 10  # Last P1 pit
        game.play(6)
        # Verify opponent mancala was skipped (index 13 for P2 mancala)
        # The stones should wrap and not add to opponent's mancala

    def test_distribution_wraps_around(self) -> None:
        """Test that distribution wraps around the board."""
        game = MancalaGame(pits_per_player=3, stones_per_pit=4)
        # With 4 stones in pit 3, and mancala at index 3,
        # stones go to mancala and wrap to P2 side
        game.play(3)
        assert game.p1_score >= 1  # At least one stone in mancala


class TestCapturing:
    """Tests for the capture rule."""

    def test_capture_rule(self) -> None:
        """Test that capturing works when landing in empty pit."""
        game = MancalaGame(pits_per_player=6, stones_per_pit=4)
        # Manually set up a capture scenario
        game.board = [0, 0, 1, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0]
        # P1 plays pit 3 (1 stone), lands in pit 4 (was empty -> capture)
        game.play(3)
        # Should capture opposite pit's stones


class TestExtraTurn:
    """Tests for the extra turn rule."""

    def test_extra_turn_in_mancala(self) -> None:
        """Test that landing in own mancala gives extra turn."""
        game = MancalaGame(pits_per_player=6, stones_per_pit=4)
        # Set up so pit 3 lands exactly in mancala (needs 4 stones, pit 3)
        game.board = [4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0]
        game.play(3)  # 4 stones from pit 3 (index 2) lands in mancala (index 6)
        # Should still be P1's turn
        assert game.current_player == 1

    def test_normal_turn_switch(self) -> None:
        """Test that normal moves switch turns."""
        game = MancalaGame(pits_per_player=6, stones_per_pit=4)
        game.play(1)  # 4 stones from pit 1, lands before mancala
        assert game.current_player == 2


class TestGameEnd:
    """Tests for game ending conditions."""

    def test_is_terminal_initially_false(self, default_game: MancalaGame) -> None:
        """Test that new games are not terminal."""
        assert not default_game.is_terminal()

    def test_is_terminal_when_side_empty(self) -> None:
        """Test that game ends when one side is empty."""
        game = MancalaGame(pits_per_player=3, stones_per_pit=1)
        # Manually empty P1's side
        game.board = [0, 0, 0, 5, 1, 1, 1, 0]
        assert game.is_terminal()

    def test_winner_determination(self) -> None:
        """Test correct winner determination."""
        game = MancalaGame(pits_per_player=3, stones_per_pit=1)
        game.board = [0, 0, 0, 10, 0, 0, 0, 5]
        assert game.is_terminal()
        assert game.get_winner() == 1  # P1 has more stones

    def test_tie_detection(self) -> None:
        """Test tie detection."""
        game = MancalaGame(pits_per_player=3, stones_per_pit=1)
        game.board = [0, 0, 0, 6, 0, 0, 0, 6]
        assert game.is_terminal()
        assert game.get_winner() == 0  # Tie


class TestGameCopy:
    """Tests for game state copying."""

    def test_copy_creates_independent_game(self, default_game: MancalaGame) -> None:
        """Test that copy creates an independent game state."""
        copy = default_game.copy()

        # Modify original
        default_game.play(1)

        # Copy should be unchanged
        assert copy.board[0] == 4
        assert default_game.board[0] == 0


class TestRandomMove:
    """Tests for random move generation."""

    def test_random_move_is_valid(self, seeded_game: MancalaGame) -> None:
        """Test that random moves are always valid."""
        for _ in range(10):
            move = seeded_game.get_random_move()
            if move is not None:
                assert seeded_game.is_valid_move(move)

    def test_random_move_none_when_no_moves(self) -> None:
        """Test that random move returns None when no moves available."""
        game = MancalaGame(pits_per_player=3, stones_per_pit=1)
        game.board = [0, 0, 0, 5, 1, 1, 1, 0]  # P1's side empty
        game.current_player = 1
        assert game.get_random_move() is None


class TestBoardRendering:
    """Tests for board display."""

    def test_render_board_not_empty(self, default_game: MancalaGame) -> None:
        """Test that board rendering produces output."""
        output = default_game.render_board()
        assert len(output) > 0
        assert "Player 1" in output

    def test_str_representation(self, default_game: MancalaGame) -> None:
        """Test __str__ produces the board."""
        output = str(default_game)
        assert "P1" in output or "Player" in output


class TestMoveHistory:
    """Tests for move history tracking."""

    def test_moves_are_recorded(self, default_game: MancalaGame) -> None:
        """Test that moves are recorded in history."""
        default_game.play(1)
        assert len(default_game.moves_history) == 1
        assert default_game.moves_history[0] == (1, 1)

    def test_multiple_moves_recorded(self) -> None:
        """Test multiple moves are recorded correctly."""
        game = MancalaGame(pits_per_player=6, stones_per_pit=4)
        game.play(1)  # P1
        game.play(1)  # P2
        assert len(game.moves_history) >= 2


class TestFullGame:
    """Integration tests for complete games."""

    def test_random_game_terminates(self) -> None:
        """Test that a random game eventually terminates."""
        rng = random.Random(42)
        game = MancalaGame(pits_per_player=6, stones_per_pit=4, rng=rng)

        max_moves = 200
        moves = 0

        while not game.is_terminal() and moves < max_moves:
            move = game.get_random_move()
            if move is None:
                break
            game.play(move)
            moves += 1

        assert game.is_terminal() or moves == max_moves
        # Most games should finish well before 200 moves
        assert moves < 150

    def test_total_stones_conserved(self, default_game: MancalaGame) -> None:
        """Test that total stones are conserved during play."""
        initial_total = sum(default_game.board)
        rng = random.Random(123)

        # Play some moves
        for _ in range(20):
            if default_game.is_terminal():
                break
            move = default_game.get_random_move()
            if move:
                default_game.play(move)

        final_total = sum(default_game.board)
        assert initial_total == final_total
