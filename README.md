# Simple Reinforcement Learning Framework

## Overview
This project is a reinforcement learning framework designed to facilitate the creation of AI players for single or multi-player environments. It is versatile and simple, focusing on environments with dynamic action spaces.

The main benefits of this framework are: 
- it's simplicity,
- the possibility to have a dynamic action space, so even games with a large action space are possible, and further, the model doesn't have to *learn* which moves are possible by negative rewards,
- the possibility to use any kind of prior evaluation function (for example hand crafted) to improve performance and convergence.


## Features
- **Discrete Action Spaces**: Works with environments that have discrete or discretizable action spaces.
- **State-Dependent Future**: The future state depends only on the current state (but the current state can consist of multiple states).
- **Known Immediate State**: The immediate next state after an action is known. Or atleast, we need to know what is not known after an action is made.

## Creating an Environment
To use this framework, the user must:
1. **Specify the game rules and actions**: Which actions can be made at each state, and how does making each action change the state?
2. **Specify what the environment does after each action**: After each action, how does the environment (not the players) react, for example by filling each player's hands with cards?
3. **Define a reward function**: How much reward is given to a player, for being in a certain state?
4. **Represent the game state numerically**: The state must be represented as a vector of numbers, which can be fed into a neural network.
5. **Create an initial guess for an evaluation function**: Create a function which takes in a state, and tells how good the state is. Can be random or based on domain knowledge.
6. **Define a neural net architecture and training loop**: The framework will handle the rest.

## Simulation and Training
The framework handles:
- Simulating games in parallel.
- Introducing exploration.
- Alternating between fitting the neural net and simulating games.

