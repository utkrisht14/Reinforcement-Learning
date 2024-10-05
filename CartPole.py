import gym
import torch
import torch.nn as nn
import random
import numpy as np
from collections import deque


from FrozenLake import MEMORY_SIZE, EPSILON_DECAY, TARGET_UPDATE_INTERVAL

# Set up the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Constants for the environment
ENV_NAME = "CartPole-v1"
SEED = 42

# Hyper-parameters
GAMMA = 0.99
LEARNING_RATE = 1e-3
MEMORY_SIZE =  10000
BATCH_SIZE = 64
EPSILON_MIN = 0.01
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
TARGET_UPDATE_INTERVAL = 10

# Neural Network for the DQN
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

# Experience the Replay Buffer Memory
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

# Define the DQN agent
class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = EPSILON_START
        self.memory = ReplayBuffer(MEMORY_SIZE)

        # Online and target network parameters
        self.online_network = DQN(state_size, action_size).to(device)
        self.target_network = DQN(state_size, action_size).to(device)
        self.update_target_network()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=LEARNING_RATE)

    def update_target_network(self):
        """Copy the weights from the online network to the target network"""
        self.target_network.load_state_dict(self.online_network.state_dict())

    def select_action(self, state):
        """ Choose the action to perform on the environment """
        if self.epsilon > EPSILON_MIN:
            return np.random.randint(0, self.action_size) # Exploration
        else:
            state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(device)
            with torch.no_grad():
                return self.online_network(state).argmax(dim=1).to(device)

    def select_action_test(self, state):
        """ Select action greedily for testing """
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            return self.online_network(state).argmax(dim=1).item()

    def store_experience(self, state, action, reward, next_state, done):
        """ Store the agent experience """
        self.memory.add((state, action, reward, next_state, done))

    def update_epsilon(self):
        """ Perform a training epsilon """
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def train_step(self):
        """ Perform a training step using a random batch from the replay buffer """
        if self.memory.size() < BATCH_SIZE:
            return

        batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones)).to(device)

        # Compute Q-values
        current_q_values = self.online_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_states).max(dim=1)[0]
        target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # Backpropagation
        self.online_network.zero_grad()
        loss.backward()
        self.optimizer.step()

# Define the training step
def train_agent(episodes=1000):
    # Set up the environment
    env = gym.make(ENV_NAME)
    env.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    # Get the size of the environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    scores = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # Select an action
            action = agent.select_action(state)

            # Take the action in the environment
            next_state, reward, done, _ = env.step(action)

            # Store the experience
            agent.store_experience(state, action, reward, next_state, done)

            # Train the agent
            agent.train_step()

            state = next_state
            total_reward += reward

        # Update epsilon
        agent.update_epsilon()

        # Update target network periodically
        if episode % TARGET_UPDATE_INTERVAL == 0:
            agent.update_target_network()

        scores.append(total_reward)

        # Print progress
        if episode % 10 == 0:
            average_score = np.mean(scores[-10:])
            print(f"Episode: {episode}, Average Score: {average_score:.2f}, Epsilon: {agent.epsilon:.2f}")

        # Check for the early stopping condition
        if np.mean(scores[-100:]) >= 195.0:
            print(f"Solved after {episode} episodes!")
            break

    env.close()
    return agent

def test_agent(agent, episodes=20):
    env = gym.make(ENV_NAME)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.select_action_test(state)

            # Take the action in the environment
            next_state, reward, done, _ = env.step(action)

            state = next_state
            total_reward += reward

        total_rewards.append(total_reward)
        print(f"Test Episode: {episode}, Total Reward: {total_reward}")

    env.close()

    # Calculate total average reward
    average_reward = np.mean(total_rewards)
    print(f"Average Test Reward over {episodes} episodes: {average_reward}")

if __name__ == "__main__":
    trained_agent = train_agent(episodes=500)
    test_agent(trained_agent, episodes=20)