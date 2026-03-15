"""
Monte Carlo Tree Search utilities for protein generation.

This module contains the MCTS state representation and utility functions
for implementing Monte Carlo Tree Search in protein generation tasks.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from loguru import logger


@dataclass
class MCTSState:
    """
    Represents a state in the MCTS tree for protein generation.

    Attributes:
        current_step: Current simulation step
        x_t: Current state tensor (dictionary of tensors for different data modes)
        x_1_pred: Predicted clean sample (dictionary of tensors)
        visits: Number of times this state has been visited
        cumulative_reward: Sum of all rewards received from this state
        parent: Parent state (None for root)
        children: List of child states
        is_fully_expanded: Whether all n_branch children have been expanded
        sample_idx: Index of the sample in the batch this state belongs to
        branch_idx: Index of the branch for this state
    """

    current_step: int
    x_t: dict[str, torch.Tensor]
    x_1_pred: dict[str, torch.Tensor] | None
    visits: int = 0
    cumulative_reward: float = 0.0
    parent: Optional["MCTSState"] = None
    children: list["MCTSState"] = None
    is_fully_expanded: bool = False
    sample_idx: int = 0
    branch_idx: int = 0

    def __post_init__(self):
        if self.children is None:
            self.children = []

    @property
    def average_reward(self) -> float:
        """Calculate average reward for this state."""
        return self.cumulative_reward / max(self.visits, 1)

    def ucb_score(self, exploration_constant: float = 1.414) -> float:
        """
        Calculate UCB score for this state.

        Args:
            exploration_constant: Exploration parameter (typically sqrt(2))

        Returns:
            UCB score for node selection
        """
        if self.visits == 0:
            return float("inf")

        if self.parent is None:
            return self.average_reward

        exploitation = self.average_reward
        exploration = exploration_constant * np.sqrt(np.log(self.parent.visits) / self.visits)
        return exploitation + exploration


def backpropagate_reward(state: MCTSState, reward: float):
    """
    Backpropagate reward up the tree.

    Args:
        state: State to start backpropagation from
        reward: Reward value to propagate
    """
    current = state
    while current is not None:
        current.visits += 1
        current.cumulative_reward += reward
        current = current.parent


def get_tree_statistics(root_state: MCTSState) -> dict[str, float]:
    """
    Get statistics about the MCTS tree.

    Args:
        root_state: Root state of the tree

    Returns:
        Dictionary with tree statistics
    """

    def count_nodes(state: MCTSState) -> int:
        count = 1
        for child in state.children:
            count += count_nodes(child)
        return count

    def get_max_depth(state: MCTSState, current_depth: int = 0) -> int:
        if not state.children:
            return current_depth
        return max(get_max_depth(child, current_depth + 1) for child in state.children)

    def get_total_visits(state: MCTSState) -> int:
        total = state.visits
        for child in state.children:
            total += get_total_visits(child)
        return total

    return {
        "total_nodes": count_nodes(root_state),
        "max_depth": get_max_depth(root_state),
        "total_visits": get_total_visits(root_state),
        "root_visits": root_state.visits,
        "root_average_reward": root_state.average_reward,
    }


def print_tree_structure(
    root_state: MCTSState,
    max_depth: int = 3,
    current_depth: int = 0,
    exploration_constant: float = 1.414,
):
    """
    Print the structure of the MCTS tree for debugging with UCB scores.

    Args:
        root_state: Root state of the tree
        max_depth: Maximum depth to print
        current_depth: Current depth (for recursion)
        exploration_constant: Exploration constant for UCB calculation
    """
    if current_depth > max_depth:
        return

    indent = "  " * current_depth
    ucb_score = root_state.ucb_score(exploration_constant)

    logger.debug(
        f"{indent}State: step={root_state.current_step}, visits={root_state.visits}, "
        f"ucb_score={ucb_score:.4f}, children={len(root_state.children)}"
    )

    for i, child in enumerate(root_state.children):
        child_ucb = child.ucb_score(exploration_constant)
        logger.debug(
            f"{indent}  Child {i}: visits={child.visits}, avg_reward={child.average_reward:.4f}, ucb_score={child_ucb:.4f}"
        )
        print_tree_structure(child, max_depth, current_depth + 1, exploration_constant)
