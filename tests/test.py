import gymnasium as gym
import numpy as np
import time

def query_environment(name, **args):
    env = gym.make(name, **args)
    spec = gym.spec(name)
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Max Episode Steps: {spec.max_episode_steps}")
    print(f"Nondeterministic: {spec.nondeterministic}")
    print(f"Reward Threshold:  {spec.reward_threshold}")
    return env

# game_name = "FrozenLake-v1"
#game_name = "MountainCarContinuous-v0"
game_name = "CartPole-v1"
env = query_environment(game_name, render_mode="human")
time.sleep(10)
observation, info = env.reset()

for _ in range(10):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    print("pre-observation: ", observation)
    print("action, ", action)
    observation, reward, terminated, truncated, info = env.step(action)
    
    print("observation: ", observation)
    print("reward: ", reward)
    print("info: ", info)
    print("terminated: ", terminated)
    print("truncated: ", truncated)

    if terminated or truncated:
        observation, info = env.reset()

time.sleep(10)
env.close()
