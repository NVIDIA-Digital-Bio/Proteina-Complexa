"""Reward models for protein design optimization.

All reward models inherit from BaseRewardModel and implement score() and extract_results().
Use CompositeRewardModel to combine multiple models with configurable weights.

Utilities:
- ensure_tensor: Convert scalars/arrays to torch.Tensor
- standardize_reward: Build standardized score return dict
"""

from proteinfoundation.rewards.base_reward import (
    GRAD_KEY,
    REWARD_KEY,
    TOTAL_REWARD_KEY,
    BaseRewardModel,
    CompositeRewardModel,
    ensure_tensor,
    standardize_reward,
)
from proteinfoundation.rewards.reward_utils import compute_reward_from_samples, initialize_reward_model

__all__ = [
    "GRAD_KEY",
    "REWARD_KEY",
    "TOTAL_REWARD_KEY",
    "BaseRewardModel",
    "CompositeRewardModel",
    "compute_reward_from_samples",
    "ensure_tensor",
    "initialize_reward_model",
    "standardize_reward",
]
