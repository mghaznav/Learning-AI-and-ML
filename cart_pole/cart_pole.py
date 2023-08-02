import gymnasium as gym
import numpy as np
from agent import Agent
from visualize import plot_learning_curve

if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    num_games = 2500
    scores = []
    epsilon = []

    agent = Agent(
        0.0001,
        0.99,
        0.005,
        1.0,
        0.01,
        1e-5,
        env.action_space.n,
        env.observation_space.shape,
    )

    for i in range(num_games):
        score = 0
        done = False
        obs, info = env.reset()

        while not done:
            action = agent.choose_action(obs)
            new_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward

            # If terminated, set next state to None
            if done:
                new_obs = None

            agent.update(obs, reward, action, new_obs)
            obs = new_obs

        scores.append(score)
        epsilon.append(agent.epsilon)

        if i % 10 == 0:
            avg_score = np.mean(scores[-10:])
            print(
                f"Episode = {i}, Score = {score:.2f}, Average Score = {avg_score:.2f}, Epsilon = {agent.epsilon:.2f}"
            )

    filename = "Cart_Pole_DQN.png"
    x = [i for i in range(num_games)]
    plot_learning_curve(x, scores, epsilon, filename)
