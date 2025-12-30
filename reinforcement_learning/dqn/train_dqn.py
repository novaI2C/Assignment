import gymnasium as gym
import numpy as np
import random
from collections import deque
import time

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

# =========================
# DQN NETWORK
# =========================
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# =========================
# REPLAY BUFFER
# =========================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions).unsqueeze(1),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)

# =========================
# HYPERPARAMETERS
# =========================
ENV_NAME = "CartPole-v1"
GAMMA = 0.99
LR = 5e-4
BATCH_SIZE = 64
MEMORY_SIZE = 10000
NUM_EPISODES = 1000

EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 500

TARGET_UPDATE_STEPS = 1000
RENDER_EVERY = 100     # render every N episodes

# =========================
# ENVIRONMENTS
# =========================
env = gym.make(ENV_NAME)
render_env = gym.make(ENV_NAME, render_mode="human")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# =========================
# NETWORKS
# =========================
policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayBuffer(MEMORY_SIZE)

# =========================
# EPSILON-GREEDY ACTION
# =========================
def select_action(state, step):
    epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-step / EPS_DECAY)
    if random.random() < epsilon:
        return env.action_space.sample()

    with torch.no_grad():
        state = torch.FloatTensor(state).unsqueeze(0)
        return policy_net(state).argmax().item()

# =========================
# TRAIN STEP
# =========================
def train_step():
    if len(memory) < BATCH_SIZE:
        return

    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

    current_q = policy_net(states).gather(1, actions)

    with torch.no_grad():
        max_next_q = target_net(next_states).max(1, keepdim=True)[0]
        target_q = rewards + GAMMA * max_next_q * (1 - dones)

    loss = nn.SmoothL1Loss()(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()

# =========================
# TRAIN LOOP
# =========================
step_count = 0
best_reward = 0
reward_history = []

for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    total_reward = 0

    while True:
        action = select_action(state, step_count)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        memory.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        step_count += 1

        train_step()

        if step_count % TARGET_UPDATE_STEPS == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            break

    reward_history.append(total_reward)

    # save best model
    if total_reward > best_reward:
        best_reward = total_reward
        torch.save(policy_net.state_dict(), "best_dqn_cartpole.pth")

    print(f"Episode {episode:4d} | Reward: {total_reward:4.0f}")

    # =========================
    # SAFE RENDER DURING TRAINING
    # =========================
    if episode % RENDER_EVERY == 0 and episode > 0:
        state_r, _ = render_env.reset()
        done_r = False

        while not done_r:
            with torch.no_grad():
                action = policy_net(
                    torch.FloatTensor(state_r).unsqueeze(0)
                ).argmax().item()

            state_r, _, terminated_r, truncated_r, _ = render_env.step(action)
            done_r = terminated_r or truncated_r
            time.sleep(0.02)

env.close()
render_env.close()

# =========================
# TRAINING CURVES
# =========================
def moving_average(x, window=50):
    return np.convolve(x, np.ones(window) / window, mode="valid")

plt.figure(figsize=(9, 5))
plt.plot(reward_history, alpha=0.3, label="Episode Reward")
plt.plot(moving_average(reward_history), linewidth=2, label="50-Episode Moving Avg")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN CartPole Training Curve")
plt.legend()
plt.grid(True)
plt.show()

print("Training finished.")
print("Best model saved as best_dqn_cartpole.pth")
