# evaluate.py

import gymnasium as gym
import time
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.results_plotter import load_results, ts2xy


# =========================
# Load trained model
# =========================
model = PPO.load("ppo_cartpole")

# =========================
# Render environment
# =========================
env = gym.make("CartPole-v1", render_mode="human")

obs, _ = env.reset()
done = False

print("🎥 Rendering trained policy...")

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    time.sleep(0.02)

env.close()

# =========================
# Plot training curve
# =========================
LOG_DIR = "./ppo_logs/"

x, y = ts2xy(load_results(LOG_DIR), "timesteps")

def moving_average(data, window=50):
    return np.convolve(data, np.ones(window) / window, mode="valid")

y_smooth = moving_average(y)
x_smooth = x[len(x) - len(y_smooth):]

plt.figure(figsize=(10, 5))
plt.plot(x, y, alpha=0.3, label="Episode Reward")
plt.plot(x_smooth, y_smooth, linewidth=2, label="Smoothed Reward")
plt.xlabel("Timesteps")
plt.ylabel("Reward")
plt.title("PPO Training Curve – CartPole")
plt.legend()
plt.grid(True)
plt.show()
