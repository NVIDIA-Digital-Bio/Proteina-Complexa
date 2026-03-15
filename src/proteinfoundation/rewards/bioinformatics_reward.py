"""Bioinformatics reward model for protein interface evaluation.

This module provides a reward model based on bioinformatics metrics
for evaluating protein-protein interfaces, including:
- Surface hydrophobicity
- Interface shape complementarity (SC)
- Interface buried surface area (dSASA)
- Interface fraction
- Interface hydrophobicity
- Interface number of residues
"""

import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)

from proteinfoundation.rewards.base_reward import BaseRewardModel, standardize_reward

# PR Alternative (bioinformatics interface scoring) - may have missing dependencies
PR_ALTERNATIVE_AVAILABLE = False
try:
    from proteinfoundation.utils.pr_alternative_utils import pr_alternative_score_interface

    PR_ALTERNATIVE_AVAILABLE = True
except (ImportError, Exception) as e:
    logger.warning(f"PR Alternative import failed: {e}. Bioinformatics metrics will return NaN.")


class BioinformaticsRewardModel(BaseRewardModel):
    """Bioinformatics-based reward model for protein interface evaluation.

    This class implements a reward model that uses bioinformatics metrics
    to evaluate protein-protein interfaces, providing scores based on
    interface properties like shape complementarity, buried surface area, etc.
    """

    IS_FOLDING_MODEL = False
    SUPPORTS_GRAD = False
    SUPPORTS_SAVE_PDB = False

    def __init__(
        self,
        reward_weights: dict[str, float],
        reward_thresholds: dict[str, float | None] | None = None,
        reward_threshold_modes: dict[str, str] | None = None,
        structure_source: str | None = None,
    ) -> None:
        """Initialize the Bioinformatics reward model.

        Args:
            reward_weights: Dictionary mapping metric names to their weights
            reward_thresholds: Dictionary mapping metric names to threshold values (None = no threshold)
            reward_threshold_modes: Dictionary mapping metric names to threshold mode ("max" or "min")
            structure_source: Key of folding model whose predicted structure to use.
                None = use generated structure.
        """
        super().__init__()
        self.reward_weights = reward_weights
        self.reward_thresholds = reward_thresholds or {}
        self.reward_threshold_modes = reward_threshold_modes or {}
        self.sc_bin = os.environ.get("SC_EXEC", "./env/docker/internal/sc")
        self.structure_source = structure_source

        # Validate threshold modes
        for metric, mode in self.reward_threshold_modes.items():
            if mode not in ["max", "min"]:
                raise ValueError(f"Threshold mode must be 'max' or 'min', got '{mode}' for metric '{metric}'")

    def score(
        self,
        pdb_path: str,
        requires_grad: bool = False,
        sequence: torch.Tensor | None = None,
        structure: torch.Tensor | None = None,
        binder_chain: str | None = None,
        target_chain: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Calculate bioinformatics reward for a given structure.

        Args:
            pdb_path: Path to PDB file to analyze (CompositeRewardModel passes
                generated or refolded path as appropriate)
            requires_grad: Whether to calculate gradients (not supported, raises error if True)
            sequence: Not used (required by interface)
            structure: Not used (required by interface)
            binder_chain: Binder chain identifier (required)
            target_chain: Target chain identifier(s) (required)
            **kwargs: Additional arguments

        Returns:
            Dictionary with reward components:
                reward: Dict[str, torch.Tensor]  # Individual metric rewards
                total_reward: torch.Tensor  # Combined total reward value
                grad: Dict[str, torch.Tensor]  # Empty (no gradients)
                interface_scores: Dict[str, float]  # Raw interface scores
        """
        self._check_capabilities(requires_grad=requires_grad)
        if target_chain is None or binder_chain is None:
            raise ValueError("target_chain and binder_chain are required for BioinformaticsRewardModel")
        if not PR_ALTERNATIVE_AVAILABLE:
            raise ValueError("PR Alternative is not available.")

        # Compute interface scores on pdb_path (CompositeRewardModel passes the requested structure)
        interface_scores, interface_AA, interface_residues_pdb_ids_str = pr_alternative_score_interface(
            pdb_path,
            binder_chain=binder_chain,
            target_chain=target_chain,
            sasa_engine="auto",
            sc_bin=self.sc_bin,
        )

        # Round scores
        interface_scores = {k: round(v, 2) if isinstance(v, float) else v for k, v in interface_scores.items()}

        # Compute reward based on weights and thresholds
        reward_components = {}
        total_reward = 0.0

        for metric_name, metric_value in interface_scores.items():
            if metric_name in self.reward_weights:
                weight = self.reward_weights[metric_name]

                # Check if threshold is set
                threshold = self.reward_thresholds.get(metric_name, None)

                if threshold is not None:
                    # Use threshold-based reward
                    threshold_mode = self.reward_threshold_modes.get(metric_name, "max")
                    if threshold_mode == "max":
                        metric_reward = weight if metric_value >= threshold else 0.0
                    else:  # min
                        metric_reward = weight if metric_value <= threshold else 0.0
                else:
                    # Use weighted value
                    metric_reward = weight * metric_value

                reward_components[metric_name] = torch.tensor(metric_reward, dtype=torch.float32)
                total_reward += metric_reward

        # Log scalar results
        logger.info("BioinformaticsRewardModel score results:")
        for k, v in reward_components.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                logger.info("  %s: %.4f", k, v.item())
        logger.info("total_reward: %.4f", total_reward)
        logger.info("--------------------------------")

        return standardize_reward(
            reward=reward_components,
            total_reward=total_reward,
            interface_scores=interface_scores,
            interface_AA=interface_AA,
            interface_residues_pdb_ids_str=interface_residues_pdb_ids_str,
        )
