import sys
sys.path.append('..')   
import numpy as np
import gymnasium as gym
from agents import QLearningAgent
from gymnasium.wrappers import RecordEpisodeStatistics
from lib import draw_summary_results


def training(env, agent, n_episodes, environment_id):
    rewards_per_episode = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        terminated = False

        # play one episode
        while not terminated:
            action = agent.get_action(obs)
            next_obs, reward, terminated, _, _ = env.step(action)

            # update the agent
            agent.update(obs, action, reward, terminated, next_obs)
            
            obs = next_obs

        agent.decay_epsilon()
        rewards_per_episode.append(reward)
        mean_rewards = np.mean(rewards_per_episode[len(rewards_per_episode)-1000:])
        if episode > 0 and episode % 1000==0:
            print(f'Episode: {episode} {reward}  Epsilon: {agent.epsilon:0.2f}  Mean Rewards {mean_rewards:0.1f}')
    draw_summary_results(env, rewards_per_episode, environment_id)


def play(env, agent, times=20):
    successed = 0
    failed = 0
    draw = 0
    for _ in range(times):
        obs, _ = env.reset()
        terminated = False

        # play one episode
        while not terminated:
            action = agent.get_action_from_q_result(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            obs = next_obs

        if reward >= 1.0:
            successed += 1
        elif reward == 0:
            draw += 1
        else:
            failed += 1
        print("reward = ", reward)
    print(f"#successed = {successed} / #draw = {draw} / #failed = {failed}")


RL_traning = False
if len(sys.argv) > 1:
    RL_traning = True

environment_id = "Blackjack-v1"
filename = f'{environment_id}.pkl'

env = gym.make(environment_id,
               render_mode= "human" if not RL_traning else None, 
               sab=True) #natural=False, sab=False

if RL_traning:
    learning_rate = 0.01
    n_episodes = 1000_000
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 1.5)  # reduce the exploration over time
    final_epsilon = 0.1
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
    play(env, agent, times=100)



