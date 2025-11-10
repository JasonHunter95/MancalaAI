# This repo has everything to do with our final project in CSCI 3202

## What we've got going on so far

1. We have a version of mancala using the ruleset discussed in class and in the project notes.
    - Currently the print statements are commented out to avoid cluttering the notebook output and speeding up the sims.
2. Mancala.py has all the game logic and it's class implementation.
3. Currently the game is set up for two random players to play against each other with absolutely no reasoning or strategy, just randomness.
4. The simulations are in the project notebook. The notebook prints all the required metrics/outputs that they want by Nov 5th.

## Simulations for 100,000 games

The player that moves first has a small advantage when the game is played randomly. This is seen mostly when the simulation count gets turned up. The 100 simulation run they suggested isn't consistent enough to make any conclusions. When we turn it up to 1000,10,000, or even 100,000 games, the first player wins more often by a small amount. We are also accounting for ties and number of moves to win on average.

Here's output from running a 100,000 game simulation:

```
Number of ties: 6588
Random Player 1 wins: 47936 times.
Random Player 2 wins: 45476 times.
Average moves played to win: 44.369781184430266
Average moves played overall: 44.57434
Player 1 wins 47.936 % of the time.
Player 2 wins 45.476 % of the time.
The players tied 6.587999999999999 % of the time.
```