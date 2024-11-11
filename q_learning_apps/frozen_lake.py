import sys
sys.path.append('..')
import numpy as np
import gymnasium as gym
from agents import QLearningAgent
from gymnasium.wrappers import RecordEpisodeStatistics
from lib import draw_summary_results
import math

def print_dqn(env, agent):
    ACTIONS = ['L','D','R','U']     # for printing 0,1,2,3 => L(eft),D(own),R(ight),U(p)

    # Get number of input nodes
    num_states = env.observation_space.n
    num_cols = int(math.sqrt(num_states + 1))

    # Loop each state and print policy to console
    for obs in range(num_states):

        # Map the best action to L D R U
        best_action = ACTIONS[agent.get_action_from_q_result(obs)]

        # The printed layout matches the FrozenLake map.
        print(f'{obs:02},{best_action}', end=' ')

        if (obs+1)%num_cols==0:
            print() # Print a newline every 4 states

def training(env, agent, n_episodes, environment_id):
    rewards_per_episode = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        terminated = False

        # play one episode
        while not terminated:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            # update the agent
            agent.update(obs, action, reward, terminated, next_obs)

            obs = next_obs

        agent.decay_epsilon()
        rewards_per_episode.append(reward)
        mean_rewards = np.mean(rewards_per_episode[len(rewards_per_episode)-1000:])
        if episode > 0 and episode % 1000==0:
            print(f'Episode: {episode} {reward}  Epsilon: {agent.epsilon:0.2f}  Mean Rewards {mean_rewards:0.1f}')
    draw_summary_results(env, rewards_per_episode, environment_id)


def play(env, agent, times=100):
    successed = 0
    failed = 0
    draw = 0

    for _ in range(times):
        obs, _ = env.reset()
        terminated = False
        truncated = False

        # play one episode
        while not terminated and not truncated:
            action = agent.get_action_from_q_result(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            obs = next_obs

        if truncated:
            draw += 1
        elif terminated and reward > 0:
            successed += 1
        else:
            failed += 1
        print("reward = ", reward)
    print(f"#successed = {successed} / #draw = {draw} / #failed = {failed}")


RL_traning = False
if len(sys.argv) > 1:
    RL_traning = True

environment_id = "FrozenLake-v1"
filename = f"{environment_id}.pkl"
env = gym.make(environment_id,
               is_slippery=True,
               map_name="8x8",
               render_mode=None)
               #render_mode="human" if not RL_traning else None)

if RL_traning:
    learning_rate = 0.1
    n_episodes = 100_000
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 1.5)  # reduce the exploration over time
    final_epsilon = 0.001
    agent = QLearningAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    env = gym.wrappers.RecordEpisodeStatistics(env)
    training(env, agent, n_episodes, environment_id)
    agent.save_to_file(filename)
else:
    agent = QLearningAgent(env)
    agent.load_from_file(filename)
    print_dqn(env, agent)
    play(env, agent, times=1000)
