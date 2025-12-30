import gymnasium as gym
import torch
import torch.nn as nn
import time

# ======================
# DQN NETWORK (SAME AS TRAIN)
# ======================
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

# ======================
# LOAD ENV & MODEL
# ======================
env = gym.make("CartPole-v1", render_mode="human")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim)
policy_net.load_state_dict(torch.load("best_dqn_cartpole.pth"))
policy_net.eval()

# ======================
# RUN EPISODE
# ======================
state, _ = env.reset()
done = False

while not done:
    with torch.no_grad():
        action = policy_net(torch.FloatTensor(state).unsqueeze(0)).argmax().item()

    state, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    time.sleep(0.02)

env.close()
