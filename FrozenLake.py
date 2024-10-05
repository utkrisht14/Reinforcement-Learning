import torch
import torch.nn as nn
import numpy as np
import random
import gym
from pydantic.json import deque

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Define the constants for the environments
ENV_NAME = "FrozenLake-v1"
RENDER_VIDEO = True
VIDEO_PATH = "./videos/"
SEED = 42

# Define the Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 1e-3
MEMORY_SIZE = 10000
BATCH_SIZE = 64
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_INTERVAL =10

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

# Define the Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

# Define the DQN Agent
class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = EPSILON_START
        self.memory = ReplayBuffer(MEMORY_SIZE)

        # Online and target network
        self.online_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)

        # Update the target network
        self.update_target_network()

        # Define the optimizer
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=LEARNING_RATE)

    def update_target_network(self):
        """ Copy the weights from the online network to the target network """
        return self.target_network.load_state_dict(self.online_network.state_dict())

    def select_action(self, state):
        """ Choose the best action possible """
        # epsilon-greedy policy
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_size)

        else:
            state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(device)
            with torch.no_grad():
                return self.online_network(state).argmax(dim=1).item()

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def select_action_test(self, state):
        """ Select action greedily for testing """
        state = torch.FLoatTensor(np.array(state)).unsqueeze(0).to(device)
        with torch.no_grad():
            return self.online_network(state).argmax(dim=1).item()

    def update_epsilon(self):
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def train_step(self):
        """ Perform a training step using a random batch from the replay buffer """
        if self.memory.size() < BATCH_SIZE:
            return

        batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to the tensors
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones)).to(device)

        # Compute Q values
        current_q_values = self.online_network(states).gather(1, actions.unsqueeze(1)).suqueeze(1)
        next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # Backpropogataion
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Now train the agent
def train_agent(episodes=1000):
    # Set up the environment
    env = gym.make(ENV_NAME)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Get the size of the environment and action_space
    state_size = env.observation_space.n
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    for episode in range(episodes):
        state, info = env.reset(seed=SEED)
        done = False
        total_reward = 0

        while not done:
            state_one_hot = np.eye(state_size)[state]

            # Select an action
            action = agent.select_action(state_one_hot)

            #  Take an action in the environment
            next_state, action, reward, terminated, truncated, _ = env.step(action)
            next_state_one_hot = np.eye(state_size)[next_state]
            done = terminated or truncated

            # Store the experience
            agent.store_experience((state_one_hot, action, reward, next_state_one_hot, done))

            # Train the agent
            agent.train_step()

            state = next_state
            total_reward += reward

        # Update epsilon
        agent.update_epsilon()

        # Update target network periodically
        if episode % TARGET_UPDATE_INTERVAL == 0:
            agent.update_target_network()

        # Update target network periodically
        if episode % 100 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}, epsilon: {agent.epsilon}")

    env.close()
    return agent

def test_agent(agent, episodes=100):
    env = gym.make(ENV_NAME)
    state_size = env.observation_space.n
    action_size = env.action_space.n

    total_rewards = []

    for episode in range(episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Convert state to one-hot encoding
            state_one_hot = np.eye(state_size)[state]

            # Select action greedily (epsilon=0)
            action = agent.select_action_test(state_one_hot)

            # Take the action in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            total_reward += reward

            # Print progress
            if episode % 10 == 0:
                print(f"Test Episode: {episode}, Total Reward: {total_reward}")

        env.close()

        # Calculate average reward and the success rate
        average_reward = np.mean(total_rewards)
        success_rate = (np.sum(total_rewards) / episodes) * 100
        print(f"Success Rate: {success_rate}%")

if __name__ == "__main__":
    trained_agent = train_agent(episodes= 1000)
    test_agent(trained_agent, episodes=100)

