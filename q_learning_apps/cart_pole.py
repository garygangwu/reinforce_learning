import sys
sys.path.append('..')   
import numpy as np
import gymnasium as gym
from agents import QLearningAgent
from gymnasium.wrappers import RecordEpisodeStatistics
from lib import draw_summary_results


def training(env, agent, n_episodes, environment_id):
    global pos_space, vel_space, ang_space, ang_vel_space
    rewards_per_episode = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        
        state = tuple([
            np.digitize(obs[0], pos_space),
            np.digitize(obs[1], vel_space),
            np.digitize(obs[2], ang_space),
            np.digitize(obs[3], ang_vel_space)
        ])
        total_rewards = 0
        # play one episode
        while not done and total_rewards < 10000:
            action = agent.get_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            next_state = tuple([
                np.digitize(next_obs[0], pos_space),
                np.digitize(next_obs[1], vel_space),
                np.digitize(next_obs[2], ang_space),
                np.digitize(next_obs[3], ang_vel_space)
            ])

            # update the agent
            agent.update(
                state, 
                action, 
                reward, 
                False, # terminated is set to False always, because finish state is in a bucket with others
                next_state)

            # update if the environment is done and the current obs
            done = terminated

            state = next_state
            total_rewards += reward

        agent.decay_epsilon()
        rewards_per_episode.append(total_rewards)
        mean_rewards = np.mean(rewards_per_episode[len(rewards_per_episode)-1000:])

        if episode % 1000==0:
            print(f'Episode: {episode} {total_rewards}  Epsilon: {agent.epsilon:0.2f}  Mean Rewards {mean_rewards:0.1f}')
    draw_summary_results(env, rewards_per_episode, environment_id)


def play(env, agent, times=100):
    global pos_space, vel_space, ang_space, ang_vel_space
    
    successed = 0
    failed = 0
    draw = 0
    for _ in range(times):
        obs, _ = env.reset()
        done = False

        # play one episode
        total_rewards = 0
        while not done and total_rewards < 10000:
            state = tuple([
                np.digitize(obs[0], pos_space),
                np.digitize(obs[1], vel_space),
                np.digitize(obs[2], ang_space),
                np.digitize(obs[3], ang_vel_space)
            ])
            action = agent.get_action_from_q_result(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            # update if the environment is done and the current obs
            done = terminated
            obs = next_obs
            total_rewards += reward

        if reward >= 10000:
            successed += 1
        elif reward == 0:
            draw += 1
        else:
            failed += 1
        print("reward = ", total_rewards)
    print(f"#successed = {successed} / #draw = {draw} / #failed = {failed}")


RL_traning = False
if len(sys.argv) > 1:
    RL_traning = True

environment_id = "CartPole-v1"
filename = f'{environment_id}.pkl'

env = gym.make(environment_id,
               render_mode= "human" if not RL_traning else None)
cart_position_space = 10
cart_velocity_space = 10
pole_angle_space = 10
pole_angular_velocity = 10
pos_space = np.linspace(-2.4, 2.4, cart_position_space)
vel_space = np.linspace(-4, 4, cart_velocity_space)
ang_space = np.linspace(-0.2095, 0.2095, pole_angle_space)
ang_vel_space = np.linspace(-4, 4, pole_angle_space)

if RL_traning:
    learning_rate = 0.1
    n_episodes = 100000
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes/1.2)  # reduce the exploration over time
    final_epsilon = 0.001
    agent = QLearningAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=0.99
    )

    env = gym.wrappers.RecordEpisodeStatistics(env)
    training(env, agent, n_episodes, environment_id)
    agent.save_to_file(filename)
else:
    agent = QLearningAgent(env)
    agent.load_from_file(filename)
    play(env, agent, times=2)