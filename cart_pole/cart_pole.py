import argparse
import gymnasium as gym
import numpy as np
from agent import Agent
from visualize import plot_learning_curve


class CartPole:
    def __init__(
        self,
        num_games: int,
        alpha: float,
        gamma: float,
        tau: float,
        epsilon_max: float,
        epsilon_min: float,
        epsilon_dec: float,
        train: bool,
        load_file: str,
        render: bool,
    ) -> None:
        if render:
            self.env = gym.make("CartPole-v1", render_mode="human")
        else:
            self.env = gym.make("CartPole-v1")

        self.num_games = num_games
        self.scores = []
        self.epsilon = []
        self.train = train

        self.agent = Agent(
            alpha,
            gamma,
            tau,
            epsilon_max,
            epsilon_min,
            epsilon_dec,
            self.env.action_space.n,
            self.env.observation_space.shape,
        )

        if load_file:
            self.agent.load(load_file)

        if not train:
            self.agent.epsilon = 0

    def run(self) -> None:
        filename = "Cart_Pole_DQN.png"

        for i in range(self.num_games):
            score = 0
            done = False
            obs, info = self.env.reset()

            while not done:
                action = self.agent.choose_action(obs)
                new_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                score += reward

                # If terminated, set next state to None
                if done:
                    new_obs = None

                # Training or playing?
                if self.train:
                    self.agent.update(obs, reward, action, new_obs)

                obs = new_obs

            self.scores.append(score)
            self.epsilon.append(self.agent.epsilon)

            if i % 10 == 0:
                avg_score = np.mean(self.scores[-10:])
                print(
                    f"Episode = {i}, Score = {score:.2f}, Average Score = {avg_score:.2f}, Epsilon = {self.agent.epsilon:.2f}"
                )

            if self.train and np.mean(self.scores[-100:]) == 500:
                file_name = self.agent.save("brain")
                print("\n\nTraining complete!\n\n")
                print(f"Agent brain was saved in file named {file_name}\n\n")
                plot_learning_curve(self.scores, self.epsilon, filename)
                exit()

        plot_learning_curve(self.scores, self.epsilon, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frozen Lake problem")

    # Optional arguments
    parser.add_argument("-n", type=int, default=2000, help="Number of games to play")
    parser.add_argument("-a", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("-g", type=float, default=0.99, help="Discount factor")
    parser.add_argument("-u", type=float, default=0.005, help="Update rate")
    parser.add_argument("-emax", type=float, default=1.0, help="Maximum epsilon value")
    parser.add_argument("-emin", type=float, default=0.01, help="Minimum epsilon value")
    parser.add_argument(
        "-edec", type=float, default=1e-5, help="Epsilon decrement value"
    )
    parser.add_argument("-t", type=str, default="T", help="Training [T/F]")
    parser.add_argument(
        "-l", type=str, default=None, help="filename to load the network from"
    )
    parser.add_argument("-r", type=str, default="F", help="Render [T/F]")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Create an instance of the CartPole class with the parsed arguments
    game_instance = CartPole(
        num_games=args.n,
        alpha=args.a,
        gamma=args.g,
        tau=args.u,
        epsilon_max=args.emax,
        epsilon_min=args.emin,
        epsilon_dec=args.edec,
        train=True if args.t == "T" else False,
        load_file=args.l,
        render=False if args.r == "F" else True,
    )

    # Run the Game class
    game_instance.run()
