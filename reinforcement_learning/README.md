Reinforcement Learning: CartPole-v1 (DQN & PPO)
Overview

This module implements two reinforcement learning algorithms—Deep Q-Network (DQN) and Proximal Policy Optimization (PPO)—to solve the CartPole-v1 environment from OpenAI Gymnasium.
The goal is to train agents to balance a pole on a moving cart and compare their learning behavior and stability.

Algorithms Implemented

Deep Q-Network (DQN) – value-based, off-policy method

Proximal Policy Optimization (PPO) – policy-gradient, on-policy method

Both agents are trained and evaluated independently.

Environment

Environment: CartPole-v1

State space: 4-dimensional continuous

Action space: 2 discrete actions

Repository Structure
rl/
├── train_dqn.py
├── evaluate_dqn.py
├── best_dqn_cartpole.pth
├── train_ppo.py
├── evaluate_ppo.py
├── ppo_cartpole/
├── ppo_logs/
└── videos/

Dependencies

Python 3.8+

gymnasium

torch

numpy

matplotlib

stable-baselines3 (for PPO)

Install using:

pip install gymnasium torch numpy matplotlib stable-baselines3

How to Run
Train DQN
python train_dqn.py

Evaluate DQN (rendered)
python evaluate_dqn.py

Train PPO
python train_ppo.py

Evaluate PPO (rendered)
python evaluate_ppo.py

Outputs

Training curves (reward vs episodes/timesteps)

Saved models

best_dqn_cartpole.pth

ppo_cartpole

Rendered evaluation videos (screen-recorded and stored in videos/)

Evaluation Criteria

Agents are evaluated based on:

Learning curves

Average episode reward

Training stability

Qualitative behavior during rendered evaluation

Platform Compatibility

Runs on x86_64 and ARM

CPU-only execution supported

Notes

PPO demonstrates smoother and more stable learning compared to DQN.

DQN serves as a strong baseline with replay-buffer-based learning.
