# Tic-Tac-Toe AI with Neural Network and Minimax

## Description

This project implements an artificial intelligence for the Tic-Tac-Toe game using the Minimax algorithm and a simple neural network for decision-making.

## Features

- **Minimax Algorithm**: A recursive algorithm that evaluates the best possible move.
- **Neural Network**: Learns from saved training data to improve its decisions.
- **Training Data Storage**: AI saves played games to `training_data.json` to enhance its gameplay.
- **Interactive Gameplay**: Users can play against the AI.

## Mathematical Background

### Minimax Algorithm
Minimax is a decision rule used in game theory to minimize the possible loss for a worst-case scenario. It operates by simulating all possible game states and selecting the move that minimizes the opponent’s best possible outcome.

Mathematically, the Minimax function is defined as:

```math
V(s) =
\begin{cases}
  \max_{a \in A(s)} V(\text{result}(s, a)) & \text{if player is maximizing} \\
  \min_{a \in A(s)} V(\text{result}(s, a)) & \text{if player is minimizing}
\end{cases}
```

where:
- \(V(s)\) is the value of a state \(s\),
- \(A(s)\) is the set of possible actions in state \(s\),
- \(\text{result}(s, a)\) is the new state after taking action \(a\).

### Neural Network
The neural network used in this project consists of two fully connected layers. The activation function used is the hyperbolic tangent function:

```math
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
```

which ensures non-linearity and maps the input values into a range between \(-1,1\). The weights of the network are updated using backpropagation with gradient descent:

```math
W_{\text{new}} = W_{\text{old}} - \alpha \frac{\partial E}{\partial W}
```

where:
- \(\alpha\) is the learning rate,
- \(E\) is the error function,
- \(W\) are the network weights.

### Game State Representation
The Tic-Tac-Toe board is represented as a vector of length 9:

```math
\text{board} = [x_1, x_2, ..., x_9]
```

where:
- \(x_i = 1\) if occupied by the player,
- \(x_i = -1\) if occupied by the AI,
- \(x_i = 0\) if empty.

This representation allows the neural network to learn optimal strategies based on past game data.

## Requirements

### Dependencies

```sh
cargo add rand serde serde_json
```

## How to Run?

```sh
git clone <repo-url>
cd <repo-folder>
cargo run
```

## Code Structure

```
NeuralNetwork         # Implements a simple neural network for decision-making
minimax               # Recursive algorithm that evaluates optimal moves
save_training_data    # Handles storing training data
load_training_data    # Handles loading training data
generate_training_data # Creates sample training data
display_board         # Visualizes the board in the terminal
get_player_move       # Retrieves the player's move
get_computer_move     # AI makes a move based on the neural network
check_win             # Checks for victory conditions
check_draw            # Checks for draw conditions
```

## Author

Project created by [Maciej Myszkiewicz](https://github.com/mysma-9403).

## License

MIT License
