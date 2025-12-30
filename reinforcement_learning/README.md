# Reinforcement Learning: CartPole-v1 (DQN & PPO)

## Overview
This folder contains implementations of two reinforcement learning algorithms — **Deep Q-Network (DQN)** and **Proximal Policy Optimization (PPO)** — for the `CartPole-v1` environment (Gymnasium). The aim is to train agents to balance the pole, compare learning behavior, and produce evaluation videos.

## Repository structure
```
reinforcement_learning/
├── train_dqn.py
├── evaluate_dqn.py
├── best_dqn_cartpole.pth
├── train_ppo.py
├── evaluate_ppo.py
├── ppo_cartpole/ # saved PPO model directory
├── ppo_logs/ # PPO training logs (tsv/json)
└── videos/ # rendered videos
└── images/ #graphs
```

## Dependencies
- `Python 3.8+`
- `gymnasium`
- `torch`
- `numpy`
- `matplotlib`
- `stable-baselines3` (for PPO)

Install with:
```bash
pip install gymnasium torch numpy matplotlib stable-baselines3
```

(Install torch following the official PyTorch instructions for your OS if needed.)
# How to run
# Train DQN
```bash
python train_dqn.py
```


Trains a DQN agent using an MLP network, replay buffer, target network.

Saves best model to `best_dqn_cartpole.pth`.

Produces a training plot (episode reward + moving average).

# Evaluate DQN (rendered)
```bash
python evaluate_dqn.py
```


Loads best_dqn_cartpole.pth.

Runs a deterministic rollout with `render_mode="human"` for demo/video recording.

# Train PPO
```bash
python train_ppo.py
```


Uses Stable-Baselines3 PPO("MlpPolicy").

Logs to `./ppo_logs/` and saves the final model as ppo_cartpole.

A callback renders an episode periodically during training.

# Evaluate PPO (rendered)
```bash
python evaluate_ppo.py
```

Loads ppo_cartpole and runs a deterministic human-rendered rollout.

Plots training curves from `./ppo_logs/.`

# Outputs

Training curves (plots)

Saved models: `best_dqn_cartpole.pth`,`ppo_cartpole`

Rendered evaluation videos (store under videos/ if created)

# Evaluation criteria (what to look for)

Learning curves (reward vs timesteps/episodes)

Average reward per episode

Training stability (variance of rewards)

Qualitative behavior in rendered demos

# Notes

PPO generally shows smoother, more stable training curves in these scripts; DQN is a baseline with replay buffer.

Scripts are CPU-compatible and tested on standard x86_64 environments. ARM should work with standard Python packages.
