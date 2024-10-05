import numpy as np
import torch
import torch.nn as nn
import gym
from collections import deque
import random

from click.core import batch

from FrozenLake import EPSILON_START, LEARNING_RATE, EPSILON_DECAY, EPSILON_MIN, BATCH_SIZE, TARGET_UPDATE_INTERVAL, \
    MEMORY_SIZE

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Set the global parameters
ENV_NAME = "CartPole-v1"
EPSILON_START = 1.0
LEARNING_RATE = 1e-3
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
GAMMA = 0.995
BATCH_SIZE = 64
TARGET_UPDATE_INTERVAL = 10 # How often to update the target network
NUM_EPISODES = 500
MAX_STEPS = 500 # Maximum steps per episodes
SEED = 42

# Neural network for the DQN
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer ,batch_size)

    def size(self):
        return len(self.buffer)

# Define the DoubleDQN Agent
class DoubleDQNAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = EPSILON_START
        self.memory = ReplayBuffer(MEMORY_SIZE)

        # Define the online network and the target network
        self.online_network = DQN(state_size, action_size).to(device)
        self.target_network = DQN(state_size, action_size).to(device)
        self.update_target_network()

        # Define optimizer
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=1e-3)

    def update_target_network(self):
        """ Copy weights from the online network to the target network """
        self.target_network.load_state_dict(self.online_network.state_dict())

    def select_action(self, state):
        """ Select the action to perform in the environment """
        if self.epsilon > EPSILON_MIN:
            return random.randrange(self.action_size)
        else:
            state = torch.FloatTensor(np.array(state)).to(device)
            with torch.no_grad():
                q_values = self.online_network(state)
            return q_values.argmax().item()

    def store_experience(self, state, action, reward, next_state, done):
        """ Store experience in the memory buffer """
        self.memory.add((state, action, reward, next_state, done))

    def update_epsilon(self):
        """ Decay epsilon """
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def train_step(self):
        if self.memory.size() < BATCH_SIZE:
            return

        # Sample a batch of experience
        batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert them into tensors
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones)).to(device)

        # Compute current Q values from online network
        current_q_values = self.online_network(states).gather(1, actions.unsqueeze(1)).to(device)

        # Compute next actions from the environment network
        next_actions = self.target_network(next_states).argmax(1).unsqueeze(1)

        # Compute next q values from the target network
        next_q_values = self.online_network(next_states).gather(1, next_actions).squeeze(1)

        # Compute target Q values
        target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values.detach())

        # Optimize the target q network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# Training function
def train_agent(env, agent, num_episodes=NUM_EPISODES):
    scores =  []
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        for t in range(MAX_STEPS):
            # Select and perform an action
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            # Modify the reward to encourage the agent to balance the pole longer
            if done and t < 199:
                reward = -1.0
            else:
                reward = 0.1

            # Store the transition in memory
            agent.store_experience(state, action, reward, next_state, done)

            # Perform one step of the optimization
            agent.train_step()

            state = next_state
            total_reward += reward

            if done:
                break

            # Update epsilon
            agent.update_epsilon()

            # Update the target network periodically
            if episode % TARGET_UPDATE_INTERVAL == 0:
                agent.update_target_network()

            scores.append(total_reward)

            # Print progress
            if episode % 10 == 0:
                avg_score = np.mean(scores[-10:])
                print(f"Episodes: {episode}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f} ")

    return agent

# Testing function
def test_agent(env, agent, num_episodes=20, render=False):
    total_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            if render:
                env.render()

            action = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward

        total_rewards.append(total_reward)
        print(f"Test episode: {episode}, Total Reward: {total_reward}")

    avg_reward = np.mean(total_rewards)
    print(f"Average test reward over {num_episodes} episodes: {avg_reward}")

# Main function
if __name__ == "__main__":
    # Create the environment
    env = gym.make(ENV_NAME)
    env.seed(SEED)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Initialize the agent
    agent = DoubleDQNAgent(state_size, action_size)

    # Train the agent
    trained_agent = train_agent(env, agent, NUM_EPISODES)

    # Test the agent
    tested_agent = test_agent(env, trained_agent, num_episodes=20, render=False)