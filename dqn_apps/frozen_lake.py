import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import math

# FrozeLake Deep Q-Learning
class FrozenLakeDQL():
    env = None
    # Hyperparameters (adjustable)
    learning_rate_a = 0.001         # learning rate (alpha)
    discount_factor_g = 0.9         # discount rate (gamma)
    network_sync_rate = 10          # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 1000       # size of replay memory
    mini_batch_size = 128           # size of the training data set sampled from the replay memory

    # Neural Network
    loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
    optimizer = None                # NN Optimizer. Initialize later.

    ACTIONS = ['L','D','R','U']     # for printing 0,1,2,3 => L(eft),D(own),R(ight),U(p)

    training_counter = 0

    def get_nn_model(self, num_states, num_actions):
        return nn.Sequential(
            nn.Linear(num_states, num_states),
            nn.ReLU(),
            nn.Linear(num_states, num_states),
            nn.ReLU(),
            nn.Linear(num_states, num_actions)
        )

    def pretrain(self, policy_dqn_model, target_dqn_model):
        memory = deque([])

        num_train_data = 400
        num_positive = num_negative = 0
        while num_positive <= num_train_data or num_negative <= num_train_data:
            state = self.env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions

            local_memory = deque([])
            # Agent navigates map until it falls into hole/reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and not truncated):
                action = self.env.action_space.sample() # actions: 0=left,1=down,2=right,3=up

                # Execute action
                new_state,reward,terminated,truncated,_ = self.env.step(action)

                # Save experience into memory
                local_memory.append((state, action, new_state, reward, terminated))

                # Move to the next state
                state = new_state

            if truncated:
                continue

            if reward > 0:
                num_positive += 1
                if num_positive <= num_train_data:
                    memory.extend(local_memory)
                    print(f"pre-train - positive({num_positive}) - mem size: {len(memory)}")
            else:
                num_negative += 1
                if num_negative <= num_train_data:
                    memory.extend(local_memory)
                    print(f"pre-train - negative({num_negative}) - mem size: {len(memory)}")


        i = 0
        while i < 3000:
            sample_batch = random.sample(memory, self.mini_batch_size)
            self.optimize(i, sample_batch, policy_dqn_model, target_dqn_model)
            # Copy policy network to target network after a certain number of steps
            target_dqn_model.load_state_dict(policy_dqn_model.state_dict())
            i += 1
            if i % 100 == 0:
                self.evaluate(100, policy_dqn_model)
        print('Policy (random, post pre-training):')
        self.print_dqn(policy_dqn_model)

    # Train the FrozeLake environment
    def train(self, episodes, map_name="4x4", render=False, is_slippery=False, pre_train=False):
        # Create FrozenLake instance
        self.env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=is_slippery, render_mode='human' if render else None)
        num_states = self.env.observation_space.n
        num_actions = self.env.action_space.n

        memory = deque([], maxlen=self.replay_memory_size)

        policy_dqn_model = self.get_nn_model(num_states, num_actions)
        target_dqn_model = self.get_nn_model(num_states, num_actions)

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        target_dqn_model.load_state_dict(policy_dqn_model.state_dict())

        print('Policy (random, before training):')
        self.print_dqn(policy_dqn_model)

        # Policy network optimizer. "Adam" optimizer can be swapped to something else.
        self.optimizer = torch.optim.Adam(policy_dqn_model.parameters(), lr=self.learning_rate_a)

        initial_epsilon = 1.0
        if pre_train:
            initial_epsilon = 0.5
            self.pretrain(policy_dqn_model, target_dqn_model)
        epsilon = initial_epsilon # 1 = 100% random actions

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = np.zeros(episodes)

        # List to keep track of epsilon decay
        epsilon_history = []

        i = 0
        max_success_rate = 0.45
        while i < episodes:
            state = self.env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions

            local_memory = deque([])
            # Agent navigates map until it falls into hole/reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and not truncated):

                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    action = self.env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
                else:
                    # select best action
                    with torch.no_grad():
                        action = policy_dqn_model(self.state_to_dqn_input(state, num_states)).argmax().item()

                # Execute action
                new_state,reward,terminated,truncated,_ = self.env.step(action)

                # Save experience into memory
                local_memory.append((state, action, new_state, reward, terminated))

                # Move to the next state
                state = new_state

            memory.extend(local_memory)

            # Keep track of the rewards collected per episode.
            if reward == 1:
                rewards_per_episode[i] = 1
                print(f"-- {i} - {np.sum(rewards_per_episode)}-- mem size: {len(memory)}")

            if len(memory) > self.mini_batch_size and np.sum(rewards_per_episode) >= 1:
                sample_batch = random.sample(memory, self.mini_batch_size)
                self.optimize(i, sample_batch, policy_dqn_model, target_dqn_model)
                    # Copy policy network to target network after a certain number of steps
                target_dqn_model.load_state_dict(policy_dqn_model.state_dict())
                    # if step_count > self.network_sync_rate:
                    #     target_dqn_model.load_state_dict(policy_dqn_model.state_dict())
                    #     step_count=0
                # Decay epsilon
                epsilon = max(epsilon - initial_epsilon/episodes, 0)
                epsilon_history.append(epsilon)
                i+=1

            if i != 0 and i % 100 == 0:
                num = 1000
                num_success = self.evaluate(num, policy_dqn_model)
                if (num_success * 1.0 / num > max_success_rate):
                    max_success_rate = num_success * 1.0 / num
                    # Save policy
                    torch.save(policy_dqn_model.state_dict(), f"frozen_lake_max_{num_success}_dql.pt")
                    print(f"max_success_rate = {max_success_rate}")


        # Close environment
        self.env.close()

        # Save policy
        torch.save(policy_dqn_model.state_dict(), "frozen_lake_dql.pt")

        # Create new graph
        plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        plt.plot(sum_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        plt.plot(epsilon_history)

        # Save plots
        plt.savefig('frozen_lake_dql_training_results.png')

    # Optimize policy network
    def optimize(self, i, mini_batch, policy_dqn_model, target_dqn_model):

        # Get number of input nodes
        num_states = policy_dqn_model[0].in_features

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:

            if terminated:
                # Agent either reached goal (reward=1) or fell into hole (reward=0)
                # When in a terminated state, target q value should be set to the reward.
                target = torch.FloatTensor([reward])
            else:
                # Calculate target q value
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + \
                        self.discount_factor_g * target_dqn_model(self.state_to_dqn_input(new_state, num_states)).max()
                    )
            #target = target - target_dqn_model(self.state_to_dqn_input(state, num_states))[action]

            # Get the current set of Q values
            current_q = policy_dqn_model(self.state_to_dqn_input(state, num_states))
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_dqn_model(self.state_to_dqn_input(state, num_states))
            # Adjust the specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)

        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.training_counter % 100 == 0:
            print(f"{i}, {self.training_counter}: loss {loss.item()}, batch size: {len(mini_batch)}")
            self.print_dqn(policy_dqn_model)
        self.training_counter += 1

    '''
    Converts an state (int) to a tensor representation.
    For example, the FrozenLake 4x4 map has 4x4=16 states numbered from 0 to 15.

    Parameters: state=1, num_states=16
    Return: tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    '''
    def state_to_dqn_input(self, state:int, num_states:int)->torch.Tensor:
        input_tensor = torch.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor

    def evaluate(self, episodes, policy_dqn_model):
        successed = 0
        failed = 0
        draw = 0
        num_states = policy_dqn_model[0].in_features
        for i in range(episodes):
            state = self.env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions

            # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
            while (not terminated and not truncated):
                # Select best action
                with torch.no_grad():
                    #action = policy_dqn_model(self.state_to_dqn_input(state, num_states)).argmax().item()
                    ordered_actions = policy_dqn_model(self.state_to_dqn_input(state, num_states)).argsort()
                    action = ordered_actions[-1].item()

                # Execute action
                state,reward,terminated,truncated,_ = self.env.step(action)
            if truncated:
                draw += 1
            elif terminated and reward > 0:
                successed += 1
            else:
                failed += 1
        print(f"#successed = {successed} / #draw = {draw} / #failed = {failed}")
        return successed

    # Run the FrozeLake environment with the learned policy
    def test(self, episodes, map_name="4x4", is_slippery=False):
        # Create FrozenLake instance
        self.env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=is_slippery, render_mode=None) # "human"
        num_states = self.env.observation_space.n
        num_actions = self.env.action_space.n

        policy_dqn_model = self.get_nn_model(num_states, num_actions)
        policy_dqn_model.load_state_dict(torch.load("frozen_lake_dql.pt"))
        policy_dqn_model.eval()

        print('Policy (trained):')
        self.print_dqn(policy_dqn_model, simple=True)
        self.evaluate(episodes, policy_dqn_model)

        self.env.close()


    # # Print DQN: state, best action, q values
    def print_dqn(self, dqn, simple=False):
        # Get number of input nodes
        num_states = dqn[0].in_features
        num_cols = int(math.sqrt(num_states + 1))

        # Loop each state and print policy to console
        for s in range(num_states):
            #  Format q values for printing
            q_values = ''
            for q in dqn(self.state_to_dqn_input(s, num_states)).tolist():
                q_values += "{:+.2f}".format(q)+' '  # Concatenate q values, format to 2 decimals
            q_values=q_values.rstrip()              # Remove space at the end

            # Map the best action to L D R U
            best_action = self.ACTIONS[dqn(self.state_to_dqn_input(s, num_states)).argmax()]

            # Print policy in the format of: state, action, q values
            # The printed layout matches the FrozenLake map.
            if simple:
                print(f'{s:02},{best_action}', end=' ')
            else:
                print(f'{s:02},{best_action},[{q_values}]', end=' ')
            if (s+1)%num_cols==0:
                print() # Print a newline every 4 states

if __name__ == '__main__':

    frozen_lake = FrozenLakeDQL()
    is_slippery = True
    map_name = "8x8"
    #frozen_lake.train(5000, map_name=map_name, is_slippery=is_slippery, pre_train=True)
    frozen_lake.test(1000, map_name=map_name, is_slippery=is_slippery)