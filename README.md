# MancalaAI

An implementation of the Mancala board game with AI players using Minimax and Alpha-Beta Pruning algorithms.

## Features

- **Complete Mancala Game Engine**: Full implementation of traditional Mancala rules including:
  - Stone distribution with wrap-around
  - Capture mechanics
  - Extra turn on landing in mancala
  - Automatic game termination and stone sweeping

- **AI Players**:
  - **Alpha-Beta Pruning**: Efficient adversarial search with configurable depth
  - **Basic Minimax**: For educational comparison
  - Configurable search depth for balancing strength vs. speed

- **Command Line Interface**:
  - Play against the AI interactively
  - Run benchmark simulations
  - Compare AI performance at different depths

- **Comprehensive Test Suite**: Full pytest coverage for game rules and AI

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MancalaAI.git
cd MancalaAI

# Install in development mode
pip install -e ".[dev]"
```

Or install dependencies manually:

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Play Against AI

```bash
# Play against AI with default settings (depth 6)
mancala play

# Play with custom settings
mancala play --depth 8 --ai-first

# Adjust board configuration
mancala play --pits 6 --stones 4 --depth 5
```

### Run Benchmarks

```bash
# Run 100 AI vs Random simulations
mancala benchmark --sims 100 --depth 5

# Compare without alpha-beta pruning
mancala benchmark --sims 50 --depth 4 --no-pruning

# Verbose output
mancala benchmark --sims 100 --depth 5 --verbose
```

### Depth Comparison

```bash
# Compare AI at different search depths
mancala compare --depths 2 5 8 --sims 50
```

### Experiments Script

For more detailed analysis with visualization:

```bash
python experiments/benchmark.py --sims 100 --depths 2 5 8 10 --random-baseline
```

## Package Structure

```
MancalaAI/
├── src/
│   └── mancala_ai/
│       ├── __init__.py      # Package exports
│       ├── game.py          # Mancala game engine
│       ├── ai.py            # AI algorithms (Minimax, Alpha-Beta)
│       └── cli.py           # Command line interface
├── tests/
│   ├── conftest.py          # Pytest fixtures
│   ├── test_game.py         # Game engine tests
│   └── test_ai.py           # AI algorithm tests
├── experiments/
│   └── benchmark.py         # Detailed benchmark script
├── pyproject.toml           # Package configuration
├── requirements.txt         # Dependencies
└── README.md
```

## API Usage

```python
from mancala_ai import MancalaGame, MancalaAI, get_alpha_beta_move

# Create a game
game = MancalaGame(pits_per_player=6, stones_per_pit=4)

# Make moves
game.play(3)  # Play pit 3
print(game.render_board())

# Check game state
if game.is_terminal():
    print(f"Winner: Player {game.get_winner()}")

# Use AI
ai = MancalaAI(pits_per_player=6, stones_per_pit=4)
state = ai.initial

# Get AI move with alpha-beta search
move = get_alpha_beta_move(ai, state, depth=5)
state = ai.result(state, move)
```

## Algorithm Overview

### Minimax with Alpha-Beta Pruning

The AI uses the classic Minimax algorithm enhanced with Alpha-Beta pruning for efficient tree search:

1. **Minimax**: Explores all possible game states, assuming optimal play from both players. Player 1 maximizes score difference, Player 2 minimizes.

2. **Alpha-Beta Pruning**: Prunes branches that cannot affect the final decision, dramatically reducing the search space.

3. **Depth Cutoff**: Search is limited to a configurable depth with heuristic evaluation at leaf nodes.

**Evaluation Function**: The utility is calculated as `P1_score - P2_score`, representing the stone advantage for Player 1.

### Performance

Typical results with Alpha-Beta Pruning (AI as Player 1 vs Random):

| Depth | Win Rate | Avg Move Time |
|-------|----------|---------------|
| 2     | ~85%     | <0.01s        |
| 5     | ~95%     | ~0.02s        |
| 8     | ~98%     | ~0.5s         |
| 10    | ~99%     | ~3s           |

## Development

### Running Tests

```bash
# Run all tests
pytest -n auto

# Run with coverage
pytest --cov=mancala_ai -n auto

# Run specific test file
pytest tests/test_game.py -v
```

## Acknowledgments

- Game-playing algorithms based on [AIMA](https://github.com/aimacode/aima-python) (Artificial Intelligence: A Modern Approach)
- Mancala rules follow the traditional Kalah variant
