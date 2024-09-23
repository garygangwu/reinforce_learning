import sys
sys.path.append('..')   
import numpy as np
import gymnasium as gym
from agents import QLearningAgent
from gymnasium.wrappers import RecordEpisodeStatistics
from lib import draw_summary_results


class ContinuousActionQLearningAgent(QLearningAgent):
    def default_value(self):
        return np.zeros(num_action_space)
    
    def get_action(self, obs) -> float:
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return self.get_action_exploration(obs)
   
    # Always taking the best option could lead to local optimization,
    # choosing sub-optimal solutoins could result in the global optimization
    def get_action_exploration(self, obs) -> float:
        arr = np.argsort(self.q_values[obs])[::-1] # from max to min
        i = arr[0]
        if len(arr) >= 3:
            r = np.random.random()
            if r < 0.5:
                i = arr[0]
            elif r < 0.8:
                i = arr[1]
            else:
                i = arr[2]
        return self.get_action_value_from_id(i)
        
    def get_action_value_from_id(self, id) -> float:
        assert(id>=0)
        assert(id<num_action_space)
        if id <= num_action_space - 2:
            return [(action_space[id] + action_space[id+1]) / 2]
        else:
            return [action_space[num_action_space-1]]


def training(env, agent, n_episodes, game_name):
    global pos_space, vel_space, action_space
    
    rewards_per_episode = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        
        state = tuple([
            np.digitize(obs[0], pos_space),
            np.digitize(obs[1], vel_space)
        ])
        total_rewards = 0
        terminated = False          # True when reached goal
        # play one episode
        while not terminated and total_rewards > -1000:
            action_float = agent.get_action(state)
            #print("action_float = ", action_float)
            next_obs, reward, terminated, truncated, _ = env.step(action_float)
            next_state = tuple([
                np.digitize(next_obs[0], pos_space),
                np.digitize(next_obs[1], vel_space)
            ])
            action_bucket = np.digitize(action_float[0], action_space) - 1
            assert(action_bucket >= 0)
            assert(action_bucket < num_action_space)

            # update the agent
            agent.update(
                state, 
                action_bucket, 
                reward, 
                False, # terminated is set to False always, because finish state could be in a bucket with others
                next_state)

            state = next_state
            total_rewards += reward

        agent.decay_epsilon()
        rewards_per_episode.append(total_rewards)
        
        mean_rewards = np.mean(rewards_per_episode[len(rewards_per_episode)-100:])
        if episode > 0 and episode % 100==0 or total_rewards > 10:
            print(f'Episode: {episode} {total_rewards}  Epsilon: {agent.epsilon:0.2f}  Mean Rewards {mean_rewards:0.1f}')
    draw_summary_results(env, rewards_per_episode, game_name)


def play(env, agent, times=100):
    global pos_space, vel_space, action_space
    
    successed = 0
    failed = 0
    draw = 0
    for _ in range(times):
        obs, _ = env.reset()
        terminated = False

        # play one episode
        total_rewards = 0
        while not terminated and total_rewards > -1000:
            state = tuple([
                np.digitize(obs[0], pos_space),
                np.digitize(obs[1], vel_space)
            ])
            action = agent.get_action_exploration(state)
            
            next_obs, reward, terminated, _, _ = env.step(action)
            
            # update if the environment is done and the current obs
            obs = next_obs
            total_rewards += reward

        if reward >= 1.0:
            successed += 1
        elif reward < 0:
            failed += 1
        else:
            draw += 1
        print("reward = ", total_rewards)
    print(f"#successed = {successed} / #draw = {draw} / #failed = {failed}")


RL_traning = False
if len(sys.argv) > 1:
    RL_traning = True

game_name = "MountainCarContinuous-v0"
filename = f'{game_name}.pkl'

env = gym.make(game_name,
               render_mode= "human" if not RL_traning else None)

num_spaces = 20
num_action_space = 10
pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num_spaces)
vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num_spaces)
action_space = np.linspace(env.action_space.low[0], env.action_space.high[0], num_action_space)

if RL_traning:
    env = gym.make(game_name)
    learning_rate = 0.9
    n_episodes = 8000
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes/1.5)  # reduce the exploration over time
    final_epsilon = 0.01
    agent = ContinuousActionQLearningAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=0.9
    )

    env = gym.wrappers.RecordEpisodeStatistics(env)
    training(env, agent, n_episodes, game_name)
    agent.save_to_file(filename)
else:
    agent = ContinuousActionQLearningAgent(env)
    agent.load_from_file(filename)
    play(env, agent, times=1000)