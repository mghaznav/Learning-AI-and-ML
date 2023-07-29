import argparse
import gymnasium as gym
import numpy as np
from agent import Agent
from visualize import Visualize


class FrozenLake:
    def __init__(
        self,
        num_games: int,
        alpha: float,
        gamma: float,
        epsilon_max: float,
        epsilon_min: float,
        epsilon_decrement: float,
    ) -> None:
        self.num_games = num_games
        self.env = gym.make("FrozenLake-v1", render_mode="rgb_array")
        self.agent = Agent(
            alpha,
            gamma,
            epsilon_max,
            epsilon_min,
            epsilon_decrement,
            4,
            16,
        )

        # Lists to collect data along the way
        self.scores = []
        self.win_percentage = []
        self.epsilon = []

    def run(self) -> None:
        for i in range(self.num_games):
            done = False
            obs, info = self.env.reset()
            score = 0
            while not done:
                action = self.agent.choose_action(obs)
                new_obs, reward, terminated, truncated, info = self.env.step(action)
                self.agent.update(obs, reward, action, new_obs)
                score += reward
                obs = new_obs
                done = terminated or truncated

            self.scores.append(score)

            # Store avg wins and epsilon value after every 100 games
            if i % 100 == 0:
                avg = np.mean(self.scores[-100:])
                self.win_percentage.append(avg)
                self.epsilon.append(self.agent.epsilon)

            # Print progress after every 1000 games
            if i % 1000 == 0:
                print(
                    f"Episode = {i}, Win Percentage = {self.win_percentage[len(self.win_percentage) - 1]}, Epsilon = {self.agent.epsilon:.2f}"
                )

        visual = Visualize(
            self.agent.q_table, self.env, 4, self.win_percentage, self.epsilon
        )
        visual.plot()

        self.env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frozen Lake problem")

    # Optional arguments
    parser.add_argument("-n", type=int, default=500000, help="Number of games to play")
    parser.add_argument("-a", type=float, default=0.001, help="Learning rate")
    parser.add_argument("-g", type=float, default=0.9, help="Discount factor")
    parser.add_argument("-emax", type=float, default=1.0, help="Maximum epsilon value")
    parser.add_argument("-emin", type=float, default=0.01, help="Minimum epsilon value")
    parser.add_argument(
        "-edec", type=float, default=0.9999995, help="Epsilon decrement value"
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Create an instance of the FrozenLake class with the parsed arguments
    game_instance = FrozenLake(
        num_games=args.n,
        alpha=args.a,
        gamma=args.g,
        epsilon_max=args.emax,
        epsilon_min=args.emin,
        epsilon_decrement=args.edec,
    )

    # Run the Game class
    game_instance.run()
