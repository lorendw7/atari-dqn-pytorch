# Atari DQN with PyTorch

A clean PyTorch implementation of Deep Q-Network (DQN) for Atari games, following the original DeepMind paper:
> [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — Mnih et al., 2013

## Overview

DQN is the foundational algorithm that demonstrated a single neural network agent could learn to play 49 Atari games at human-level performance, using only raw pixel input and game score as feedback.

**Key techniques implemented:**
- **Experience Replay** — stores past transitions in a buffer and samples random mini-batches to break correlation between consecutive frames
- **Target Network** — a periodically-updated copy of the Q-network to stabilize training targets
- **Frame Stacking** — stacks 4 consecutive grayscale frames so the agent can perceive motion
- **Frame Skipping** — repeats each action for 4 frames to reduce computation and smooth control

## Results

| Game      | Training Steps | Mean Score |
|-----------|---------------|------------|
| Breakout  | 10M           | ~300       |
| Pong      | 10M           | ~20        |

## Project Structure

```
atari-dqn-pytorch/
├── config.py           # Hyperparameters and training settings
├── env_wrapper.py      # Atari preprocessing wrappers
├── replay_buffer.py    # Experience replay memory
├── model.py            # CNN Q-network architecture
├── agent.py            # DQN agent (action selection + learning)
├── train.py            # Main training loop
└── environment.yml     # Conda environment definition
```

## Requirements

- Python 3.11+
- PyTorch 2.7+ (CUDA 12.8 recommended for Blackwell GPUs)
- GPU with 12GB VRAM (tested on RTX 5070 Ti Laptop)

## Setup

**1. Create the conda environment:**
```bash
conda env create -f environment.yml
conda activate rl-general
```

**2. Install PyTorch with CUDA support** (must be done separately — conda does not pass `--index-url`):
```bash
# For NVIDIA Blackwell (RTX 50xx) and Ampere/Ada (RTX 30xx/40xx) with CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**3. Download Atari ROMs:**
```bash
autorom --accept-license
```

**4. Verify your installation:**
```bash
python verify_env.py
```

> **Note for gymnasium >= 1.0:** ALE environments must be registered explicitly before use:
> ```python
> import ale_py, gymnasium as gym
> gym.register_envs(ale_py)
> env = gym.make("ALE/Breakout-v5")
> ```

## Training

```bash
# Train on Breakout (default)
python train.py

# Train on a different game
python train.py --env ALE/Pong-v5

# Resume from a checkpoint
python train.py --resume checkpoints/breakout_step_1000000.pt
```

Training logs are written to `runs/` and can be viewed with TensorBoard:
```bash
tensorboard --logdir runs/
```

## How It Works

### The Q-Function

The core idea of DQN is to learn a function **Q(s, a)** — the expected total future reward when taking action **a** in state **s** and then playing optimally.

```
Q(s, a) = r + γ · max_a' Q(s', a')
          └──┘   └────────────────┘
        immediate    future value
         reward    (discounted by γ)
```

This is called the **Bellman equation**. The network is trained to minimize the difference between its current prediction and this target.

### Training Loop

```
For each step:
  1. Observe current state s (stack of 4 frames)
  2. Choose action a:
       - with probability ε → random action  (exploration)
       - otherwise         → argmax Q(s, a)  (exploitation)
  3. Execute action, receive reward r and next state s'
  4. Store (s, a, r, s', done) in replay buffer
  5. Sample random mini-batch from buffer
  6. Compute target: y = r + γ · max_a' Q_target(s', a')
  7. Update Q-network: minimize (Q(s,a) - y)²
  8. Every N steps: copy Q-network weights → target network
```

### Network Architecture

```
Input: (4, 84, 84) — 4 stacked grayscale frames
  ↓
Conv2d(4→32, kernel=8, stride=4)   → (32, 20, 20)  + ReLU
  ↓
Conv2d(32→64, kernel=4, stride=2)  → (64, 9, 9)    + ReLU
  ↓
Conv2d(64→64, kernel=3, stride=1)  → (64, 7, 7)    + ReLU
  ↓
Flatten → Linear(3136→512)         + ReLU
  ↓
Linear(512→n_actions)
  ↓
Output: Q-value for each action
```

## Hyperparameters

Key parameters (defined in `config.py`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `replay_buffer_size` | 100,000 | Number of transitions stored |
| `batch_size` | 32 | Mini-batch size for each update |
| `learning_rate` | 1e-4 | Adam optimizer LR |
| `gamma` | 0.99 | Discount factor |
| `epsilon_start` | 1.0 | Initial exploration rate |
| `epsilon_end` | 0.01 | Final exploration rate |
| `epsilon_decay_steps` | 1,000,000 | Steps to anneal epsilon |
| `target_update_freq` | 10,000 | Steps between target network syncs |
| `train_start` | 50,000 | Steps before training begins |

## References

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) (Mnih et al., 2013)
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) (Mnih et al., 2015, Nature)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
