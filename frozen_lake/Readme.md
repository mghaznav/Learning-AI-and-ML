# Frozen Lake Q-Learning

This folder contains my implementation of training a Q-learning model using epsilon-greedy to explore and learn the Frozen Lake environment from OpenAI's Gymnasium. The Frozen Lake environment can be found at the following link: [Frozen Lake Environment](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)

## Instructions

To run the `frozen_lake.py` file with the desired optional arguments, follow these steps:

1. Ensure you have followed the installation instructions in the readme at the base of this directory.

2. Run the `frozen_lake.py` file with the desired optional arguments using the following command structure:

    ```bash
    python frozen_lake.py [-n NUM_GAMES] [-a ALPHA] [-g GAMMA] [-emax EPSILON_MAX] [-emin EPSILON_MIN] [-edec EPSILON_DECREMENT]
    ```

- `-n`: Number of games to play (default: 500000).
- `-a`: Learning rate (default: 0.001).
- `-g`: Discount factor (default: 0.9).
- `-emax`: Maximum epsilon value (default: 1.0).
- `-emin`: Minimum epsilon value (default: 0.01).
- `-edec`: Epsilon decrement value (default: 0.9999995).

For example, to run the Frozen Lake training with 10000 games, an alpha of 0.01, and a gamma of 0.95, you can use the following command:

    ```bash
    python frozen_lake.py -n 10000 -a 0.01 -g 0.95
    ```

You can also use the -h flag to view the optional arguments in the terminal

    ```bash
    python frozen_lake.py -h
    ```

3. The program will execute the Q-learning training process using epsilon-greedy exploration to learn the Frozen Lake environment.

## Note

This implementation showcases my training of a Q-learning model using epsilon-greedy to explore and learn the Frozen Lake environment from OpenAI's Gymnasium. The solutions provided here are just one of the many possible approaches to training reinforcement learning models. Feel free to experiment, modify, and enhance the code to suit your learning needs and explore different RL algorithms.

Happy learning, and have fun exploring the Frozen Lake environment with Q-learning!
