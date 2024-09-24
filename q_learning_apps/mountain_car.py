import sys
sys.path.append('..')   
import numpy as np
import gymnasium as gym
from agents import QLearningAgent
from gymnasium.wrappers import RecordEpisodeStatistics
from lib import draw_summary_results


def training(env, agent, n_episodes, environment_id):
    global pos_space, vel_space
    rewards_per_episode = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        
        state = tuple([
            np.digitize(obs[0], pos_space),
            np.digitize(obs[1], vel_space)
        ])
        total_rewards = 0
        # play one episode
        while not done:
            action = agent.get_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = tuple([
                np.digitize(next_obs[0], pos_space),
                np.digitize(next_obs[1], vel_space)
            ])

            # update the agent
            agent.update(
                state, 
                action, 
                reward, 
                False, # terminated is set to False always, because finish state is in a bucket with others
                next_state)

            # update if the environment is done and the current obs
            done = terminated or total_rewards <= -1000

            state = next_state
            total_rewards += reward

        agent.decay_epsilon()
        rewards_per_episode.append(total_rewards)
        mean_rewards = np.mean(rewards_per_episode[len(rewards_per_episode)-100:])
        if episode > 0 and episode % 100==0:
            print(f'Episode: {episode} {reward}  Epsilon: {agent.epsilon:0.2f}  Mean Rewards {mean_rewards:0.1f}')
    draw_summary_results(env, rewards_per_episode, environment_id)


def play(env, agent, times=100):
    global pos_space, vel_space
    
    successed = 0
    failed = 0
    draw = 0
    for _ in range(times):
        obs, _ = env.reset()
        done = False

        # play one episode
        total_rewards = 0
        terminated = False
        while not terminated:
            state = tuple([
                np.digitize(obs[0], pos_space),
                np.digitize(obs[1], vel_space)
            ])
            action = agent.get_action_from_q_result(state)
            next_obs, reward, terminated, _, _ = env.step(action)

            obs = next_obs
            total_rewards += reward
        if total_rewards >= 1.0:
            successed += 1
        elif total_rewards == 0:
            draw += 1
        else:
            failed += 1
        print("total_rewards = ", total_rewards)
    print(f"#successed = {successed} / #draw = {draw} / #failed = {failed}")


RL_traning = False
if len(sys.argv) > 1:
    RL_traning = True

environment_id = "MountainCar-v0"
filename = f'{environment_id}.pkl'

env = gym.make(environment_id,
               render_mode= "human" if not RL_traning else None)
pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)

if RL_traning:
    env = gym.make(environment_id)
    
    learning_rate = 0.9
    n_episodes = 100_00
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 1.5)  # reduce the exploration over time
    final_epsilon = 0
    agent = QLearningAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=0.9
    )

    env = gym.wrappers.RecordEpisodeStatistics(env)
    training(env, agent, n_episodes, environment_id)
    agent.save_to_file(filename)
else:
    agent = QLearningAgent(env)
    agent.load_from_file(filename)
    play(env, agent, times=1000)