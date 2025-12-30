import gymnasium as gym
import time
import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy


# =========================
# Custom callback to render during training
# =========================
class TrainingRenderCallback(BaseCallback):
    def __init__(self, render_freq=10_000, verbose=1):
        super().__init__(verbose)
        self.render_freq = render_freq
        self.render_env = gym.make("CartPole-v1", render_mode="human")

    def _on_step(self) -> bool:
        if self.num_timesteps % self.render_freq == 0:
            print(f"\n🎥 Rendering at step {self.num_timesteps}")

            obs, _ = self.render_env.reset()
            done = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=False)
                obs, _, terminated, truncated, _ = self.render_env.step(action)
                done = terminated or truncated
                time.sleep(0.02)

        return True

    def _on_training_end(self):
        self.render_env.close()


# =========================
# Logging directory
# =========================
log_dir = "./ppo_logs/"
os.makedirs(log_dir, exist_ok=True)

# =========================
# Training environment (NO RENDER)
# =========================
train_env = Monitor(gym.make("CartPole-v1"), log_dir)

# =========================
# PPO model
# =========================
model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    learning_rate=3e-4,
    gamma=0.99,
    n_steps=2048,
    batch_size=64
)

# =========================
# Train with rendering callback
# =========================
callback = TrainingRenderCallback(render_freq=15_000)

model.learn(
    total_timesteps=100_000,
    callback=callback
)

model.save("ppo_cartpole")

# =========================
# Plot training curve
# =========================
x, y = ts2xy(load_results(log_dir), "timesteps")

def moving_average(data, window=50):
    return np.convolve(data, np.ones(window)/window, mode="valid")

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
