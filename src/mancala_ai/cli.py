"""
Mancala AI Command Line Interface.

This module provides CLI commands for:
- Playing Mancala against the AI
- Running benchmark simulations
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from typing import Optional

import numpy as np

from mancala_ai.ai import GameState, MancalaAI, get_alpha_beta_move, get_minimax_move
from mancala_ai.game import MancalaGame


def play_game(
    depth: int = 6,
    player_first: bool = True,
    pits: int = 6,
    stones: int = 4,
) -> None:
    """
    Play an interactive game against the AI.

    Args:
        depth: AI search depth in plies.
        player_first: If True, human plays as Player 1 (first).
        pits: Number of pits per player.
        stones: Initial stones per pit.
    """
    ai = MancalaAI(pits_per_player=pits, stones_per_pit=stones)
    state = ai.initial

    human_player = 1 if player_first else 2
    ai_player = 2 if player_first else 1

    print("\n" + "=" * 50)
    print("   MANCALA - Human vs AI (Alpha-Beta, Depth {})".format(depth))
    print("=" * 50)
    print(f"\nYou are Player {human_player}")
    print("Enter pit number (1-{}) to make a move.".format(pits))
    print("Enter 'q' to quit.\n")

    while not ai.terminal_test(state):
        # Display current state
        ai.display(state)
        print()

        current_player = ai.to_move(state)

        if current_player == human_player:
            # Human's turn
            valid_moves = ai.actions(state)
            print(f"Your valid moves: {valid_moves}")

            while True:
                try:
                    user_input = input("Your move: ").strip().lower()
                    if user_input == "q":
                        print("\nGame abandoned.")
                        return

                    move = int(user_input)
                    if move in valid_moves:
                        break
                    print(f"Invalid move. Choose from: {valid_moves}")
                except ValueError:
                    print("Please enter a valid pit number.")
        else:
            # AI's turn
            print("AI is thinking...")
            start_time = time.time()
            move = get_alpha_beta_move(ai, state, depth)
            elapsed = time.time() - start_time
            print(f"AI plays pit {move} (took {elapsed:.2f}s)\n")

        # Execute the move and check for extra turn
        previous_player = current_player
        state = ai.result(state, move)
        next_player = ai.to_move(state)

        # Check if the same player gets another turn (extra turn!)
        if not ai.terminal_test(state) and previous_player == next_player:
            if previous_player == human_player:
                print("★ EXTRA TURN! Your last stone landed in your mancala. Go again!")
            else:
                print("★ EXTRA TURN! AI's last stone landed in its mancala.")

    # Game over
    print("\n" + "=" * 50)
    print("                  GAME OVER")
    print("=" * 50)
    ai.display(state)

    winner = state.board.get_winner()
    p1_score = state.board.p1_score
    p2_score = state.board.p2_score

    print(f"\nFinal Score: Player 1: {p1_score} | Player 2: {p2_score}")

    if winner == 0:
        print("It's a tie!")
    elif winner == human_player:
        print("Congratulations! You win!")
    else:
        print("AI wins. Better luck next time!")


def run_benchmark(
    num_sims: int = 100,
    depth: int = 5,
    use_alpha_beta: bool = True,
    pits: int = 6,
    stones: int = 4,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> dict:
    """
    Run AI vs Random benchmark simulations.

    Args:
        num_sims: Number of games to simulate.
        depth: AI search depth in plies.
        use_alpha_beta: If True, use alpha-beta pruning; otherwise basic minimax.
        pits: Number of pits per player.
        stones: Initial stones per pit.
        seed: Random seed for reproducibility.
        verbose: If True, print progress updates.

    Returns:
        Dictionary with benchmark results.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    algorithm = "Alpha-Beta Pruning" if use_alpha_beta else "Basic Minimax"
    get_move = get_alpha_beta_move if use_alpha_beta else get_minimax_move

    print(f"\n{'=' * 60}")
    print(f"  Mancala Benchmark: AI ({algorithm}, Depth {depth}) vs Random")
    print(f"{'=' * 60}")
    print(f"  Simulations: {num_sims}")
    print(f"  Board: {pits} pits, {stones} stones each")
    if seed is not None:
        print(f"  Random seed: {seed}")
    print()

    ai_wins = 0
    random_wins = 0
    ties = 0
    moves_per_game: list[int] = []
    game_times: list[float] = []
    ai_move_times: list[float] = []

    for sim in range(num_sims):
        if verbose and (sim + 1) % 10 == 0:
            print(f"  Running game {sim + 1}/{num_sims}...")

        ai_game = MancalaAI(pits_per_player=pits, stones_per_pit=stones)
        state = ai_game.initial
        moves = 0
        game_start = time.time()

        while not ai_game.terminal_test(state):
            current_player = ai_game.to_move(state)

            if current_player == 1:
                # AI player
                move_start = time.time()
                move = get_move(ai_game, state, depth)
                ai_move_times.append(time.time() - move_start)
            else:
                # Random player
                move = state.board.get_random_move()

            if move is None:
                break

            state = ai_game.result(state, move)
            moves += 1

        game_times.append(time.time() - game_start)
        moves_per_game.append(moves)

        # Determine winner
        final_board = state.board
        if final_board.p1_score > final_board.p2_score:
            ai_wins += 1
        elif final_board.p1_score < final_board.p2_score:
            random_wins += 1
        else:
            ties += 1

    # Calculate statistics
    results = {
        "algorithm": algorithm,
        "depth": depth,
        "num_sims": num_sims,
        "ai_wins": ai_wins,
        "random_wins": random_wins,
        "ties": ties,
        "ai_win_rate": ai_wins / num_sims * 100,
        "random_win_rate": random_wins / num_sims * 100,
        "tie_rate": ties / num_sims * 100,
        "avg_moves_per_game": np.mean(moves_per_game),
        "avg_game_time": np.mean(game_times),
        "avg_ai_move_time": np.mean(ai_move_times) if ai_move_times else 0,
        "total_time": sum(game_times),
    }

    # Print results
    print("\n  Results:")
    print("  " + "-" * 40)
    print(f"  AI Wins:       {ai_wins:4} ({results['ai_win_rate']:.1f}%)")
    print(f"  Random Wins:   {random_wins:4} ({results['random_win_rate']:.1f}%)")
    print(f"  Ties:          {ties:4} ({results['tie_rate']:.1f}%)")
    print("  " + "-" * 40)
    print(f"  Avg moves/game:    {results['avg_moves_per_game']:.1f}")
    print(f"  Avg game time:     {results['avg_game_time']:.3f}s")
    print(f"  Avg AI move time:  {results['avg_ai_move_time']:.3f}s")
    print(f"  Total time:        {results['total_time']:.1f}s")
    print()

    return results


def run_depth_comparison(
    depths: list[int],
    num_sims: int = 100,
    pits: int = 6,
    stones: int = 4,
    seed: Optional[int] = None,
) -> list[dict]:
    """
    Compare AI performance across different search depths.

    Args:
        depths: List of depths to test.
        num_sims: Games per depth level.
        pits: Number of pits per player.
        stones: Initial stones per pit.
        seed: Random seed for reproducibility.

    Returns:
        List of result dictionaries, one per depth.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    print(f"\n{'=' * 60}")
    print("  Mancala AI Depth Comparison")
    print(f"{'=' * 60}")
    print(f"  Depths: {depths}")
    print(f"  Simulations per depth: {num_sims}")
    print()

    all_results = []

    for depth in depths:
        print(f"\n  Testing depth {depth}...")
        result = run_benchmark(
            num_sims=num_sims,
            depth=depth,
            use_alpha_beta=True,
            pits=pits,
            stones=stones,
            seed=seed,
            verbose=False,
        )
        all_results.append(result)

    # Summary table
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Depth':<8} {'Win Rate':<12} {'Avg Move Time':<15} {'Total Time':<12}")
    print("  " + "-" * 50)
    for r in all_results:
        print(
            f"  {r['depth']:<8} {r['ai_win_rate']:>6.1f}%     "
            f"{r['avg_ai_move_time']:>8.3f}s       {r['total_time']:>8.1f}s"
        )
    print()

    return all_results


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="mancala",
        description="Mancala AI - Play against an AI or run benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mancala play --depth 5           Play against AI with depth 5
  mancala benchmark --sims 100     Run 100 simulations at default depth
  mancala compare --depths 2 5 8   Compare AI performance at different depths
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Play command
    play_parser = subparsers.add_parser("play", help="Play against the AI")
    play_parser.add_argument(
        "--depth", "-d", type=int, default=6, help="AI search depth in plies (default: 6)"
    )
    play_parser.add_argument(
        "--ai-first",
        action="store_true",
        help="Let AI go first (default: human first)",
    )
    play_parser.add_argument(
        "--pits", "-p", type=int, default=6, help="Pits per player (default: 6)"
    )
    play_parser.add_argument(
        "--stones", "-s", type=int, default=4, help="Initial stones per pit (default: 4)"
    )

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run AI vs Random simulations")
    bench_parser.add_argument(
        "--sims", "-n", type=int, default=100, help="Number of simulations (default: 100)"
    )
    bench_parser.add_argument(
        "--depth", "-d", type=int, default=5, help="AI search depth (default: 5)"
    )
    bench_parser.add_argument(
        "--no-pruning",
        action="store_true",
        help="Use basic minimax instead of alpha-beta pruning",
    )
    bench_parser.add_argument(
        "--pits", "-p", type=int, default=6, help="Pits per player (default: 6)"
    )
    bench_parser.add_argument(
        "--stones", "-s", type=int, default=4, help="Initial stones per pit (default: 4)"
    )
    bench_parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    bench_parser.add_argument("--verbose", "-v", action="store_true", help="Print progress")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare AI at different depths")
    compare_parser.add_argument(
        "--depths",
        "-d",
        type=int,
        nargs="+",
        default=[2, 5, 8],
        help="Depths to compare (default: 2 5 8)",
    )
    compare_parser.add_argument(
        "--sims", "-n", type=int, default=50, help="Simulations per depth (default: 50)"
    )
    compare_parser.add_argument(
        "--pits", "-p", type=int, default=6, help="Pits per player (default: 6)"
    )
    compare_parser.add_argument(
        "--stones", "-s", type=int, default=4, help="Initial stones per pit (default: 4)"
    )
    compare_parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    if args.command == "play":
        play_game(
            depth=args.depth,
            player_first=not args.ai_first,
            pits=args.pits,
            stones=args.stones,
        )
    elif args.command == "benchmark":
        run_benchmark(
            num_sims=args.sims,
            depth=args.depth,
            use_alpha_beta=not args.no_pruning,
            pits=args.pits,
            stones=args.stones,
            seed=args.seed,
            verbose=args.verbose,
        )
    elif args.command == "compare":
        run_depth_comparison(
            depths=args.depths,
            num_sims=args.sims,
            pits=args.pits,
            stones=args.stones,
            seed=args.seed,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
