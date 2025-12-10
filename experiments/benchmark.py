#!/usr/bin/env python3
"""
Mancala AI Benchmark Script.

This script provides comprehensive benchmarking for the MancalaAI package,
including:
- AI vs Random player simulations
- Depth comparison analysis
- Performance visualization

Usage:
    python experiments/benchmark.py --sims 100 --depths 2 5 8 10
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mancala_ai.ai import MancalaAI, get_alpha_beta_move, get_minimax_move
from mancala_ai.game import MancalaGame


def run_random_vs_random(num_sims: int = 1000, seed: int = 42) -> dict:
    """
    Simulate random vs random games for baseline comparison.

    Args:
        num_sims: Number of simulations to run.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with simulation results.
    """
    random.seed(seed)
    np.random.seed(seed)

    p1_wins = 0
    p2_wins = 0
    ties = 0
    moves_per_game: list[int] = []

    for _ in range(num_sims):
        game = MancalaGame(pits_per_player=6, stones_per_pit=4)
        moves = 0

        while not game.is_terminal():
            move = game.get_random_move()
            if move is None:
                break
            game.play(move)
            moves += 1

        moves_per_game.append(moves)

        if game.p1_score > game.p2_score:
            p1_wins += 1
        elif game.p2_score > game.p1_score:
            p2_wins += 1
        else:
            ties += 1

    return {
        "p1_wins": p1_wins,
        "p2_wins": p2_wins,
        "ties": ties,
        "p1_win_rate": p1_wins / num_sims * 100,
        "p2_win_rate": p2_wins / num_sims * 100,
        "tie_rate": ties / num_sims * 100,
        "avg_moves": np.mean(moves_per_game),
        "num_sims": num_sims,
    }


def run_ai_vs_random(
    num_sims: int = 100,
    depth: int = 5,
    use_alpha_beta: bool = True,
    seed: int = 42,
) -> dict:
    """
    Simulate AI vs random player games.

    Args:
        num_sims: Number of simulations.
        depth: AI search depth.
        use_alpha_beta: Use alpha-beta pruning (vs basic minimax).
        seed: Random seed.

    Returns:
        Dictionary with simulation results.
    """
    random.seed(seed)
    np.random.seed(seed)

    get_move = get_alpha_beta_move if use_alpha_beta else get_minimax_move

    ai_wins = 0
    random_wins = 0
    ties = 0
    moves_per_game: list[int] = []
    game_times: list[float] = []
    ai_move_times: list[float] = []

    for _ in range(num_sims):
        ai = MancalaAI(pits_per_player=6, stones_per_pit=4)
        state = ai.initial
        moves = 0
        game_start = time.time()

        while not ai.terminal_test(state):
            if ai.to_move(state) == 1:
                move_start = time.time()
                move = get_move(ai, state, depth)
                ai_move_times.append(time.time() - move_start)
            else:
                move = state.board.get_random_move()

            if move is None:
                break

            state = ai.result(state, move)
            moves += 1

        game_times.append(time.time() - game_start)
        moves_per_game.append(moves)

        if state.board.p1_score > state.board.p2_score:
            ai_wins += 1
        elif state.board.p1_score < state.board.p2_score:
            random_wins += 1
        else:
            ties += 1

    return {
        "ai_wins": ai_wins,
        "random_wins": random_wins,
        "ties": ties,
        "ai_win_rate": ai_wins / num_sims * 100,
        "random_win_rate": random_wins / num_sims * 100,
        "tie_rate": ties / num_sims * 100,
        "avg_moves": np.mean(moves_per_game),
        "avg_game_time": np.mean(game_times),
        "avg_ai_move_time": np.mean(ai_move_times) if ai_move_times else 0,
        "total_time": sum(game_times),
        "depth": depth,
        "algorithm": "alpha_beta" if use_alpha_beta else "minimax",
        "num_sims": num_sims,
    }


def run_depth_sweep(
    depths: list[int],
    num_sims: int = 100,
    seed: int = 42,
) -> list[dict]:
    """
    Run benchmarks across multiple depths.

    Args:
        depths: List of depths to test.
        num_sims: Simulations per depth.
        seed: Random seed.

    Returns:
        List of result dictionaries.
    """
    results = []

    for depth in depths:
        print(f"  Running depth {depth}...")
        result = run_ai_vs_random(
            num_sims=num_sims,
            depth=depth,
            use_alpha_beta=True,
            seed=seed,
        )
        results.append(result)
        print(f"    Win rate: {result['ai_win_rate']:.1f}%, "
              f"Avg move time: {result['avg_ai_move_time']:.3f}s")

    return results


def plot_results(results: list[dict], output_path: str = "ai_performance.png") -> None:
    """
    Create visualization of benchmark results.

    Args:
        results: List of result dictionaries from depth sweep.
        output_path: Path to save the plot.
    """
    depths = [r["depth"] for r in results]
    win_rates = [r["ai_win_rate"] for r in results]
    move_times = [r["avg_ai_move_time"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Win rate plot
    ax1.plot(depths, win_rates, marker="o", linewidth=3, markersize=10, color="#2E86AB")
    ax1.set_xlabel("Search Depth (Plies)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("AI Win Rate (%)", fontsize=12, fontweight="bold")
    ax1.set_title("AI Performance vs Search Depth", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_xticks(depths)
    ax1.set_ylim(0, 105)

    # Move time plot
    ax2.plot(depths, move_times, marker="s", linewidth=3, markersize=10, color="#E94F37")
    ax2.set_xlabel("Search Depth (Plies)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Average Move Time (seconds)", fontsize=12, fontweight="bold")
    ax2.set_title("Computational Cost vs Search Depth", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.set_xticks(depths)
    ax2.set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")
    plt.show()


def main() -> None:
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(
        description="Run MancalaAI benchmarks and analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--sims", "-n", type=int, default=100,
        help="Number of simulations per configuration (default: 100)"
    )
    parser.add_argument(
        "--depths", "-d", type=int, nargs="+", default=[2, 5, 8],
        help="Search depths to test (default: 2 5 8)"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--random-baseline", action="store_true",
        help="Include random vs random baseline"
    )
    parser.add_argument(
        "--plot", "-p", type=str, default="ai_performance.png",
        help="Output path for performance plot"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip plot generation"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  MancalaAI Benchmark Suite")
    print("=" * 60)
    print(f"  Simulations per depth: {args.sims}")
    print(f"  Depths to test: {args.depths}")
    print(f"  Random seed: {args.seed}")
    print()

    # Random baseline
    if args.random_baseline:
        print("Running Random vs Random baseline...")
        baseline = run_random_vs_random(num_sims=args.sims * 10, seed=args.seed)
        print(f"  P1 Win Rate: {baseline['p1_win_rate']:.1f}%")
        print(f"  P2 Win Rate: {baseline['p2_win_rate']:.1f}%")
        print(f"  Tie Rate: {baseline['tie_rate']:.1f}%")
        print(f"  Avg Moves: {baseline['avg_moves']:.1f}")
        print()

    # Depth sweep
    print("Running AI vs Random depth sweep...")
    results = run_depth_sweep(
        depths=args.depths,
        num_sims=args.sims,
        seed=args.seed,
    )

    # Summary
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Depth':<8} {'Win Rate':<12} {'Avg Move Time':<16} {'Total Time':<12}")
    print("  " + "-" * 52)
    for r in results:
        print(
            f"  {r['depth']:<8} {r['ai_win_rate']:>6.1f}%     "
            f"{r['avg_ai_move_time']:>10.3f}s     {r['total_time']:>8.1f}s"
        )

    # Plot
    if not args.no_plot and len(results) > 1:
        print("\nGenerating performance plot...")
        plot_results(results, args.plot)


if __name__ == "__main__":
    main()
