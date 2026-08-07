# SafeRL-IFT-7201

A comparative study of reinforcement learning approaches (Q-learning, PPO, and P30) for learning safe policies in stochastic environments.

## Project Structure

```
SafeRL-IFT-7201/
├── environments.py                             # Definition of wrapped FrozenLake environments
├── utils.py                                    # Utility functions
├── train_ppo.py                                # PPO training script (Proximal Policy Optimization)
├── train_q_learning.py                         # Q-learning training script
├── train_p3o.py                                # P3O training script (Penalized Proximal Policy Optimization)
├── model_p3o.py                                # Implementation of the P3O algorithm based on the Stable-Baselines3 architecture
├── generate_train_plots.py                     # Script for generating training plots
├── generate_train_plots_param_search.py        # Script for generating hyperparameter search plots
├── Visualize_environments_and_policies.ipynb   # Notebook for visualizing the case studies and learned policies
├── pyproject.toml                              # Dependency configuration (uses `uv`)
├── README.md                                   # This file
```

## Installation

This project uses **`uv`** for dependency management. Make sure `uv` is installed before following the steps below.

### Requirements

- Python ≥ 3.12
- `uv` installed ([documentation](https://docs.astral.sh/uv/))

### Install dependencies

```bash
# Install the project dependencies in the virtual environment
uv sync
```

## Usage

### Train a PPO / Q-learning / P3O model

```bash
uv run train_ppo.py
uv run train_q_learning.py
uv run train_p3o.py
```

### View TensorBoard logs

```bash
# Launch TensorBoard to visualize the training metrics
tensorboard --logdir ./<algorithm>_<env_type>/

# Example:
tensorboard --logdir ./ppo_8x8_default/
```
