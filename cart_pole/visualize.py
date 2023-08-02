import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate


def plot_learning_curve(scores, epsilons, filename):
    # Create a smooth curve for Scores
    smooth_training_steps = np.linspace(1, len(scores), 100)
    scores_interp = interpolate.interp1d(
        np.arange(1, len(scores) + 1), scores, kind="cubic"
    )
    smooth_scores = scores_interp(smooth_training_steps)

    # Create a smooth curve for Epsilon values
    epsilons_interp = interpolate.interp1d(
        np.arange(1, len(epsilons) + 1), epsilons, kind="cubic"
    )
    smooth_epsilon_values = epsilons_interp(smooth_training_steps)

    # Plot the data and the curves for averages
    plt.figure(figsize=(10, 6))

    # Plot Epsilon values on the left y-axis
    ax1 = plt.gca()
    ax1.plot(
        smooth_training_steps,
        smooth_epsilon_values,
        label="Epsilon Values",
        color="green",
    )
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Epsilon Values", color="green")
    ax1.tick_params(axis="y", labelcolor="green")
    ax1.set_ylim(0.0, 1.0)  # Set y-axis limits for Epsilon from 0.0 to 1.0

    # Plot Scores on the right y-axis
    ax2 = ax1.twinx()
    ax2.plot(smooth_training_steps, smooth_scores, label="Scores", color="blue")
    ax2.set_ylabel("Scores", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")
    ax2.set_ylim(0, 500)  # Set y-axis limits for Scores from 0 to 500

    plt.title("Cart Pole DQL Agent Training")

    plt.savefig(filename)
