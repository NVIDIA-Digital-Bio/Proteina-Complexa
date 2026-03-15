"""Base reward model class for protein design optimization.

This module provides an abstract base class for all reward models, ensuring
a consistent interface for evaluating and optimizing protein sequences and structures.

Reward models are classified into:
- **Folding models** (af2, boltz2, ptx, rf3): Predict structures and can save PDB output.
- **Non-folding models** (bioinformatics, tmol): Compute metrics over structures.
  Non-folding models can use generated structures (pdb_path) or refolded structures from
  a specific folding model via the structure_source parameter (model key in CompositeRewardModel).
"""

import inspect
import logging
import os
import warnings
from abc import ABC, abstractmethod
from typing import Any

import torch

# Standard keys for reward model output. All models must return a dict with these keys.
REWARD_KEY = "reward"
GRAD_KEY = "grad"
TOTAL_REWARD_KEY = "total_reward"


def ensure_tensor(value: float | int | torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert scalar or array to torch.Tensor if needed.

    Args:
        value: Scalar, numpy-like value, or existing tensor.
        dtype: Target dtype.

    Returns:
        torch.Tensor with given dtype.
    """
    if isinstance(value, torch.Tensor):
        return value.to(dtype) if value.dtype != dtype else value
    return torch.tensor(value, dtype=dtype)


def standardize_reward(
    reward: dict[str, torch.Tensor],
    grad: dict[str, torch.Tensor] | None = None,
    total_reward: float | torch.Tensor | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build a standardized result dict for score.

    Args:
        reward: Dict of reward component names to tensors.
        grad: Dict of gradient names to tensors. Defaults to {}.
        total_reward: Scalar total reward. Defaults to tensor(0.0) if None.
        **extra: Additional keys (e.g. plddt, pae, model-specific fields).

    Returns:
        Dict conforming to BaseRewardModel.score return format.
    """
    result: dict[str, Any] = {
        REWARD_KEY: reward,
        GRAD_KEY: grad if grad is not None else {},
        TOTAL_REWARD_KEY: ensure_tensor(total_reward if total_reward is not None else 0.0),
    }
    result.update(extra)
    return result


def _is_folding_model(model: "BaseRewardModel") -> bool:
    """Check if a reward model is a folding model (predicts and can save structures)."""
    return getattr(model, "IS_FOLDING_MODEL", False)


logger = logging.getLogger(__name__)


class BaseRewardModel(ABC):
    """Abstract base class for reward models.

    All reward models should inherit from this class and implement ``score()``.
    ``extract_results()`` has a default pass-through — override only when needed.

    Class attributes:
        IS_FOLDING_MODEL: True for folding models (AF2, Boltz2, PTX, RF3) that
            predict structures and can save PDB output.
        SUPPORTS_GRAD: True if the model supports gradient computation.
        SUPPORTS_SAVE_PDB: True if the model can save a predicted PDB file.
    """

    IS_FOLDING_MODEL = False
    SUPPORTS_GRAD = False
    SUPPORTS_SAVE_PDB = False

    _REQUIRED_CAPABILITY_ATTRS = (
        "IS_FOLDING_MODEL",
        "SUPPORTS_GRAD",
        "SUPPORTS_SAVE_PDB",
    )

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return
        for attr in cls._REQUIRED_CAPABILITY_ATTRS:
            if attr not in cls.__dict__:
                raise TypeError(
                    f"{cls.__name__} must explicitly set class attribute '{attr}'. "
                    f"See BaseRewardModel docstring for details."
                )

    def _check_capabilities(self, requires_grad: bool = False, save_pdb: bool = False) -> None:
        """Validate that the requested capabilities are supported by this model.

        Call at the top of ``score()`` to replace per-model boilerplate guards.

        Raises:
            ValueError: If an unsupported capability is requested.
        """
        if requires_grad and not self.SUPPORTS_GRAD:
            raise ValueError(
                f"{type(self).__name__} does not support gradient computation. "
                f"Set requires_grad=False or use a model with SUPPORTS_GRAD=True."
            )
        if save_pdb and not self.SUPPORTS_SAVE_PDB:
            raise ValueError(
                f"{type(self).__name__} does not support save_pdb. "
                f"Set save_pdb=False or use a model with SUPPORTS_SAVE_PDB=True."
            )

    @staticmethod
    def _validate_score_output(result: dict[str, Any]) -> None:
        """Validate that a score() return dict has the required structure.

        Raises:
            TypeError/ValueError with a clear message on misimplementation.
        """
        if not isinstance(result, dict):
            raise TypeError(f"score() must return a dict, got {type(result).__name__}")
        missing = {REWARD_KEY, GRAD_KEY, TOTAL_REWARD_KEY} - set(result.keys())
        if missing:
            raise ValueError(f"score() result missing required keys: {missing}")
        if not isinstance(result[REWARD_KEY], dict):
            raise TypeError(f"result['{REWARD_KEY}'] must be a dict, got {type(result[REWARD_KEY]).__name__}")
        if not isinstance(result[GRAD_KEY], dict):
            raise TypeError(f"result['{GRAD_KEY}'] must be a dict, got {type(result[GRAD_KEY]).__name__}")
        if not isinstance(result[TOTAL_REWARD_KEY], torch.Tensor):
            raise TypeError(
                f"result['{TOTAL_REWARD_KEY}'] must be a torch.Tensor, got {type(result[TOTAL_REWARD_KEY]).__name__}"
            )

    @abstractmethod
    def score(self, pdb_path: str, requires_grad: bool = False, **kwargs) -> dict[str, Any]:
        """Calculate reward and gradients for a given sequence and structure.

        Args:
            pdb_path: Path to PDB file (required)
            requires_grad: Whether to calculate gradients
            **kwargs: Model-specific arguments, including:
                - sequence: Optional input sequence tensor (logits or one-hot encoded)
                - structure: Optional generated coordinates
                - binder_chain: Optional binder chain (for binder protocol)
                - target_chain: Optional target chain (for binder protocol)
                - save_pdb: Optional flag to save PDB file (only for folding models)
                - output_pdb_path: Path for folding models to save predicted structure
                - For non-folding models in CompositeRewardModel: pdb_path is the
                  resolved structure (generated or refolded per structure_source)
                - Additional model-specific arguments

        Returns:
            Dictionary with reward components and gradients:
                reward: Dict[str, torch.Tensor]  # Reward values for each reward function
                grad: Dict[str, torch.Tensor]  # Gradients for sequence and structure
                total_reward: torch.Tensor  # Total reward value
                Additional model-specific fields (plddt, pae, ptm, etc.)
        """

    def extract_results(self, aux: dict[str, Any]) -> dict[str, Any]:
        """Extract reward dictionary and scores from model auxiliary output.

        Default implementation is a pass-through. Override only when the model
        needs to transform raw auxiliary output (e.g. AF2's JAX aux dict).

        Args:
            aux: Auxiliary output from the reward model

        Returns:
            Dictionary with reward components and gradients
        """
        return aux

    def cleanup(self) -> None:
        """Explicit cleanup of model memory.

        Call this method to free up GPU memory and clear state.
        Subclasses can override this for model-specific cleanup.
        """

    def __del__(self) -> None:
        """Ensure cleanup happens on deletion."""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up {self.__class__.__name__}: {e}")


class CompositeRewardModel(BaseRewardModel):
    """Composite reward model that combines multiple reward models from configuration.

    This class allows combining different reward models (e.g., AF2, PTX, RF3, TMOL,
    Bioinformatics) into a single reward model with configurable weights for
    each component. It can be instantiated directly from Hydra configuration.

    Execution order: All folding models run first (and optionally save predicted structures),
    then all non-folding models run. Non-folding models can specify structure_source
    (a folding model key) to use that model's predicted structure instead of the
    generated structure.

    Capability flags are dynamic properties computed from child models.
    """

    @property
    def IS_FOLDING_MODEL(self):
        return any(m.IS_FOLDING_MODEL for m in self.reward_models.values())

    @property
    def SUPPORTS_GRAD(self):
        return any(m.SUPPORTS_GRAD for m in self.reward_models.values())

    @property
    def SUPPORTS_SAVE_PDB(self):
        return any(m.SUPPORTS_SAVE_PDB for m in self.reward_models.values())

    def __init__(
        self,
        reward_models: dict[str, BaseRewardModel],
        weights: dict[str, float] | None = None,
    ):
        """Initialize the composite reward model.

        Args:
            reward_models: Dictionary mapping reward model names to reward model instances
            weights: Dictionary mapping reward model names to their weights (default: 1.0 for all)
        """
        super().__init__()
        self.reward_models = reward_models
        self.weights = weights or {}

        # Set default weight of 1.0 for models without specified weights
        for name in reward_models:
            if name not in self.weights:
                self.weights[name] = 1.0

        # Validate that all weights correspond to reward models
        for name in self.weights:
            if name not in reward_models:
                raise ValueError(f"Weight specified for '{name}' but no corresponding reward model found")

    def _partition_models(
        self,
    ) -> tuple[list[tuple[str, BaseRewardModel]], list[tuple[str, BaseRewardModel]]]:
        """Partition reward models into folding and non-folding."""
        folding: list[tuple[str, BaseRewardModel]] = []
        non_folding: list[tuple[str, BaseRewardModel]] = []
        for name, model in self.reward_models.items():
            if _is_folding_model(model):
                folding.append((name, model))
            else:
                non_folding.append((name, model))
        return folding, non_folding

    def _get_structure_source(self, model: BaseRewardModel) -> str | None:
        """Get structure_source for non-folding model (folding model key to use for structure)."""
        return getattr(model, "structure_source", None)

    @staticmethod
    def _accumulate_result(
        name: str,
        weight: float,
        result: dict[str, Any],
        combined_reward_dict: dict[str, torch.Tensor],
        combined_grad_dict: dict[str, torch.Tensor],
        total_reward: torch.Tensor,
    ) -> torch.Tensor:
        """Accumulate a single model's result into the combined dicts and total reward."""
        if TOTAL_REWARD_KEY in result:
            r = result[TOTAL_REWARD_KEY]
            total_reward = total_reward + weight * (
                r if isinstance(r, torch.Tensor) else torch.tensor(r, dtype=torch.float32)
            )
        if REWARD_KEY in result:
            for key, value in result[REWARD_KEY].items():
                v = value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.float32)
                combined_reward_dict[f"{name}_{key}"] = weight * v
        if GRAD_KEY in result:
            for key, value in result[GRAD_KEY].items():
                if value is not None:
                    v = value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.float32)
                    combined_grad_dict[f"{name}_{key}"] = weight * v
        return total_reward

    def score(self, pdb_path: str, requires_grad: bool = False, **kwargs) -> dict[str, Any]:
        """Calculate combined reward from all reward models.

        Runs all folding models first (each saving to pdb_path_refolded_{key}.pdb when
        a non-folding model needs its structure), then all non-folding models.
        Non-folding models use structure_source to select which structure to analyze.

        Args:
            pdb_path: Path to PDB file (required)
            requires_grad: Whether to calculate gradients
            **kwargs: Arguments passed to individual reward models

        Returns:
            Dictionary with combined reward components
        """
        model_results = {}
        total_reward = torch.tensor(0.0, dtype=torch.float32)
        combined_reward_dict = {}
        combined_grad_dict = {}
        base_path = pdb_path.replace(".pdb", "")

        folding, non_folding = self._partition_models()

        # Collect which folding models need to save (requested by non-folding models).
        folding_keys_to_save: set = set()
        for name, model in non_folding:
            src = self._get_structure_source(model)
            if src and src in self.reward_models and getattr(self.reward_models[src], "SUPPORTS_SAVE_PDB", False):
                folding_keys_to_save.add(src)

        refolded_pdb_paths: dict[str, str] = {}

        # Phase 1: Run all folding models
        for name, model in folding:
            weight = self.weights.get(name, 1.0)
            model_kwargs = kwargs.copy()
            output_path = f"{base_path}_refolded_{name}.pdb"
            if name in folding_keys_to_save:
                model_kwargs["save_pdb"] = True
                model_kwargs["output_pdb_path"] = output_path
            try:
                result = model.score(pdb_path=pdb_path, requires_grad=requires_grad, **model_kwargs)
                model_results[name] = result
                if name in folding_keys_to_save and os.path.exists(output_path):
                    refolded_pdb_paths[name] = output_path
                total_reward = self._accumulate_result(
                    name,
                    weight,
                    result,
                    combined_reward_dict,
                    combined_grad_dict,
                    total_reward,
                )
            except Exception as e:
                warnings.warn(f"Error computing reward from folding model '{name}': {e}")

        # Phase 2: Run all non-folding models
        for name, model in non_folding:
            weight = self.weights.get(name, 1.0)
            model_kwargs = kwargs.copy()
            structure_source = self._get_structure_source(model)
            if structure_source and structure_source in refolded_pdb_paths:
                non_folding_pdb_path = refolded_pdb_paths[structure_source]
            elif structure_source and structure_source in self.reward_models:
                folding_model = self.reward_models[structure_source]
                if not getattr(folding_model, "SUPPORTS_SAVE_PDB", False):
                    raise ValueError(
                        f"Non-folding model '{name}' requested structure_source='{structure_source}', "
                        f"but that folding model does not support save_pdb."
                    )
                non_folding_pdb_path = pdb_path
            else:
                non_folding_pdb_path = pdb_path
            try:
                result = model.score(
                    pdb_path=non_folding_pdb_path,
                    requires_grad=requires_grad,
                    **model_kwargs,
                )
                model_results[name] = result
                total_reward = self._accumulate_result(
                    name,
                    weight,
                    result,
                    combined_reward_dict,
                    combined_grad_dict,
                    total_reward,
                )
            except Exception as e:
                warnings.warn(f"Error computing reward from model '{name}': {e}")

        return {
            REWARD_KEY: combined_reward_dict,
            GRAD_KEY: combined_grad_dict,
            TOTAL_REWARD_KEY: total_reward,
            "model_rewards": model_results,
        }

    def cleanup(self) -> None:
        """Cleanup all reward models."""
        for model in self.reward_models.values():
            if hasattr(model, "cleanup"):
                model.cleanup()
