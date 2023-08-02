# Cart Pole Deep Q-Learning

This folder contains my implementation of training a Deep Q-learning model with experience replay that ezplores and learns the Cart Pole environment from OpenAI's Gymnasium with limited knowledge of the game. The Cart Pole environment can be found at the following link: [Cart Pole Environment](https://gymnasium.farama.org/environments/classic_control/cart_pole/)

## Instructions

To run the `cart_pole.py` file with the desired optional arguments, follow these steps:

1. Ensure you have followed the installation instructions in the readme at the base of this directory.

2. Run the `cart_pole.py` file with the desired optional arguments using the following command structure:

    ```bash
    python cart_pole.py [-n NUM_GAMES] [-a ALPHA] [-g GAMMA] [-u TAU] [-emax EPSILON_MAX] [-emin EPSILON_MIN] [-edec EPSILON_DECREMENT]
    ```

- `-n`: Number of games to play (default: 500000).
- `-a`: Learning rate (default: 0.001).
- `-g`: Discount factor (default: 0.9).
- `-u`: Update rate (default: 0.005).
- `-emax`: Maximum epsilon value (default: 1.0).
- `-emin`: Minimum epsilon value (default: 0.01).
- `-edec`: Epsilon decrement value (default: 0.9999995).
- `-t`: Training T/F (default: T).
- `-l`: Filename to load the network from (default: None).
- `-r`: Render T/F (default: F).

For example, to run the Cart Pole training with 10000 games, an alpha of 0.01, and a gamma of 0.95, you can use the following command:

    ```bash
    python cart_pole.py -n 10000 -a 0.01 -g 0.95
    ```

You can also use the -h flag to view the optional arguments in the terminal

    ```bash
    python cart_pole.py -h
    ```

3. The program will execute the Q-learning training process using epsilon-greedy exploration to learn the Cart Pole environment.

## Note

This implementation showcases my training of a Deep Q-learning model to explore and learn the Cart Pole environment from OpenAI's Gymnasium. The solutions provided here are just one of the many possible approaches to training reinforcement learning models. Feel free to experiment, modify, and enhance the code to suit your learning needs and explore different RL algorithms.

Happy learning, and have fun exploring the Cart Pole environment with Q-learning!
