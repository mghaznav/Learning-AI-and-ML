# Visualization code based on OpenAI's Gymnasium tutorial:
# https://gymnasium.farama.org/tutorials/training_agents/FrozenLake_tuto/#visualization

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class Visualize:
    def __init__(self, q_table, env, map_size, win_percentage, epsilon) -> None:
        self.q_table = q_table
        self.env = env
        self.map_size = map_size
        self.win_percentage = win_percentage
        self.epsilon = epsilon

    def qtable_directions_map(self) -> None:
        """Get the best learned action & map it to arrows."""
        qtable_val_max = self.q_table.max(axis=1).reshape(self.map_size, self.map_size)
        qtable_best_action = np.argmax(self.q_table, axis=1).reshape(
            self.map_size, self.map_size
        )
        directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
        qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
        eps = np.finfo(float).eps  # Minimum float number on the machine
        for idx, val in enumerate(qtable_best_action.flatten()):
            if qtable_val_max.flatten()[idx] > eps:
                # Assign an arrow only if a minimal Q-value has been learned as best action
                # otherwise since 0 is a direction, it also gets mapped on the tiles where
                # it didn't actually learn anything
                qtable_directions[idx] = directions[val]
        qtable_directions = qtable_directions.reshape(self.map_size, self.map_size)
        return qtable_val_max, qtable_directions

    def plot(self) -> None:
        """Plot the last frame of the simulation and the policy learned."""
        qtable_val_max, qtable_directions = self.qtable_directions_map()

        # Plot the last frame
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
        ax[0, 0].imshow(self.env.render())
        ax[0, 0].axis("off")
        ax[0, 0].set_title("Last frame")

        # Plot the policy
        sns.heatmap(
            qtable_val_max,
            annot=qtable_directions,
            fmt="",
            ax=ax[0, 1],
            cmap=sns.color_palette("Blues", as_cmap=True),
            linewidths=0.7,
            linecolor="black",
            xticklabels=[],
            yticklabels=[],
            annot_kws={"fontsize": "xx-large"},
        ).set(title="Learned Q-values\nArrows represent best action")
        for _, spine in ax[0, 1].spines.items():
            spine.set_visible(True)
            spine.set_linewidth(0.7)
            spine.set_color("black")

        # Plot the win percentage
        ax[1, 0].plot(self.win_percentage)
        ax[1, 0].set_title("Win percentage")

        # Plot the epsilon value
        ax[1, 1].plot(self.epsilon)
        ax[1, 1].set_title("Epsilon value over time")

        img_title = f"frozenlake_q_values_{self.map_size}x{self.map_size}.png"
        fig.savefig(f"./{img_title}", bbox_inches="tight")
        plt.show()
