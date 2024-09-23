import numpy as np
import matplotlib.pyplot as plt


def draw_summary_results(env, rewards_per_episode, game_name = None):
    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
    axs[0].set_title("Episode rewards")
    # compute and assign a rolling average of the data to provide a smoother graph
    reward_moving_average = (
        np.convolve(
            np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
        )
        / rolling_length
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[1].set_title("Episode lengths")
    length_moving_average = (
        np.convolve(
            np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
        )
        / rolling_length
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)

    sum_rewards = np.zeros(len(rewards_per_episode))
    for t in range(len(rewards_per_episode)):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    axs[2].set_title("rewards accumulated")
    axs[2].plot(sum_rewards)
    plt.tight_layout()
    if game_name:
        plt.savefig(f"{game_name}_training_results")
    plt.show()