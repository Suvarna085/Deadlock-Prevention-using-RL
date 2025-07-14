# Deadlock Detection using Reinforcement Learning

This project uses a Deep Q-Network (DQN) to detect and prevent deadlocks in a simulated process-resource environment.

## 🔍 Overview

- Custom environment simulating process requests & resource allocation
- RL agent learns to avoid deadlocks via reward signals
- Metrics tracking for performance (rewards & deadlocks avoided)

- 
## 🧠 How it Works

- **State**: Encoded from held/requested resource matrices
- **Action**: Select which resource to allocate
- **Reward**: Positive for avoiding deadlock, negative otherwise

## ▶️ Run

```bash
python rl_model_trainer.py

Outputs:

Model: model/dqn_agent_episode_*.h5

Metrics: logs/metrics.json

Plot: logs/performance_plot.png
