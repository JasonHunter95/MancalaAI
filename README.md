# CSCI 3202 Final Project - Mancala AI with Minimax Alpha-Beta Pruning

## Overview

1. We have a version of mancala using the ruleset discussed in class and in the project notes.
    - Currently the print statements are commented out to avoid cluttering the notebook output and speeding up the sims.
2. ```Mancala.py``` has all the game logic and it's class implementation.
3. ```MancalaAI.py``` serves as a wrapper allowing us to use functions from ```aima-python``` in our mancala implementation.
4. ```project.ipynb``` is where all the simulations are run and metrics are shown.
5. The game can be run with a combination of different player types:

- Random vs Random
- Minimax AI vs Random
- Minimax AI with Alpha-Beta Pruning vs Random

In the simulations that use an AI player, the depth limit can also be changed.

## Simulations for 100,000 games

The player that moves first has a small advantage when the game is played randomly. This is seen mostly when the simulation count gets turned up. The 100 simulation run they suggested isn't consistent enough to make any conclusions. When we turn it up to 1000, 10,000, or even 100,000 games, the first player wins more often by a small amount. We are also accounting for ties and number of moves to win on average.

Here's output from running a 100,000 game simulation:

```text
Number of ties: 6588
Random Player 1 wins: 47936 times.
Random Player 2 wins: 45476 times.
Average moves played to win: 44.369781184430266
Average moves played overall: 44.57434
Player 1 wins 47.936 % of the time.
Player 2 wins 45.476 % of the time.
The players tied 6.587999999999999 % of the time.
```

## AI MiniMax Player Architecture

Using the libraries from ```aima-python```, we have implemented a version of the game with an AI player that uses minimax alpha-beta pruning to choose its moves down to a given depth. The utility function used is the difference between the number of stones in the two mancalas.

## ```MancalaAI.py```

This is how the AI player is set up. In order to properly use the ```aima-python``` library code, we needed to create a wrapper class that could essentially translate between our version of mancala and the game framework used in    ```aima-python```.

The ```MancalaAI``` class uses ```games.py``` from the aima repo to create a game object that is compatible with the ```alpha_beta_cutoff_search``` function.

The class initializes a fresh mancala instance and wraps it in a ```GameState``` object from ```games.py```. The object holds info as follows:

- ```to_move```: indicates which player's turn it is (ai player (1) or random player (2)).
- ```board```: the actual Mancala game object that holds its state.
- ```utility```: the score difference between the two players.
- ```moves```: the list of pit indices that can be chosen for the current game state turn.

The class has versions of the required methods from ```games.py``` like ```actions```, ```result```, ```utility```, ```terminal_test```, and ```to_move```.

```actions(state)```:

- Returns available moves for the current state
- Calls ```get_valid_moves()``` from ```Mancala.py``` which checks which pits can be played
- Returns and empty list if the game is over

```result(state, move)```:

- This is our **most important** method because it generates the game tree
- It makes a deep copy of the current board state to avoid making any modifications to the original state
- It calls ```mancala.play(move)``` to simulate a move and get the next state of the game
- It returns a new ```GameState``` object with:
  - updated board
  - current player gets swapped
  - updated utility score
  - new list of valid moves

```utility(state, player)```:

- Number of stones in ai player's mancala - number of stones in random player's mancala
- Returns either positive values when ai is winning, or negative if losing
- The sign is flipped based on which player is being evaluated

```terminal_test(state)```:

- Checks if the game is over by calling ```mancala.winning_eval()```
- Returns ```True``` if either player's pits are all empty

## Minimax Search Tree Visual

```text
                    Current State
                   /      |      \
                 Move1  Move2  Move3  ← MAX level (Player 1)
                 /  \    /  \    /  \
               ... ...  ... ... ... ... ← MIN level (Player 2)
                \   /    \   /   \   /
                  Eval1   Eval2   Eval3  ← Leaf nodes (Utility values)
```

### The Alpha-Beta Pruning Process

1. Start at the root (current game state)
2. MAX's turn (AI player):
   - Try each valid move (1-6)
   - For each one of these moves, recursively evaluate MIN's response
   - Track $\alpha$ (best value that MAX can guarantee)
3. MIN's turn (Random player):
    - For each of MAX's moves, try each valid move (1-6)
    - For each of these moves, recursively evaluate MAX's response
    - Track $\beta$ (worst value MIN can force on MAX)
4. Pruning:
    - If at any point $\alpha \geq \beta$, prune that branch
5. Cutoff: Stop searching when:
    - Depth limit is reached ( 4 plies )
    - Terminal state is reached.
6. Evaluation: At cutoff nodes, use utility function.

### The ```get_minimax_move``` Function

This function calls the ```alpha_beta_cutoff_search```. It takes in the mancala game object, the current game state, and the number of plies/depth to search.

- It uses a ```cutoff_test``` helper function to stop searching if either the depth limit is reached or the game is over.
- The ```eval_fn``` uses the utility from the MAX player's perspective
- Positive values favor the AI, while negative favor the random player

### Why Did We Use Deep Copy?

Without deep copy:

- All game states would share the same board object to reference.
- Making a move in one branch would affect other branches.
- This would corrupt the game tree and lead to bad evaluations.

With deep copy:

- Each game state has its own independent board.
- Moves can be simulated without side effects.
- Branches remain isolated.

### Running Simulations with Varying Depth Limits and Versions of the AI in ```project.ipynb```

We have several simulations set up that vary in depth limit, and AI player strategy (minimax with and without alpha-beta pruning).

Each simulation tracks wins, losses, and ties along with their percentages, as well as average moves to win, average moves played overall (in case of ties), average time per game, and average time per move for the AI player.

- The first simulation uses a depth of 5 plies and a basic minimax search without alpha-beta pruning optimizations.

  ```text
  AI Player wins: 95 (95.0%)
  Random Player wins: 3 (3.0%)
  Ties: 2 (2.0%)
  Average moves played: 29.2
  Average moves played to win: 29.3
  Average time per move for the AI player with basic minimax 5-ply depth: 0.239 seconds
  Average time per game: 3.557 seconds
  ```

- The second simulation uses a depth of 5 plies and a more optimal minimax approach with alpha-beta pruning.

  ```text
  AI Player wins: 97 (97.0%)
  Random Player wins: 2 (2.0%)
  Ties: 1 (1.0%)
  Average moves played: 28.8
  Average moves played to win: 28.9
  Average time per move for the AI player with alpha-beta pruning with 5-ply depth: 0.036 seconds
  Average time per game: 0.529 seconds
  ```

- The third simulation uses a depth of 7 plies with alpha-beta pruning.
  - The doc says that we should use a depth of 10, but it isn't feasible to be so I think 7 is good.

  ```text
  AI Player wins: 97 (97.0%)
  Random Player wins: 2 (2.0%)
  Ties: 1 (1.0%)
  Average moves played: 27.5
  Average moves played to win: 27.6
  Average time per move for the AI player with alpha-beta pruning and 7-ply depth: 0.269 seconds
  Average time per game: 3.749 seconds
  ```
