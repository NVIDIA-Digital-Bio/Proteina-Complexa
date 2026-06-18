"""
Reward model initialization and sample evaluation utilities.
Extracted from Proteina class for separation of concerns.
"""

import os
import shutil
import tempfile
from datetime import datetime
from typing import Any

import hydra
import torch
from loguru import logger

from proteinfoundation.rewards.base_reward import REWARD_KEY, TOTAL_REWARD_KEY
from proteinfoundation.utils.pdb_utils import get_chain_ids_from_pdb, write_prot_ligand_to_pdb, write_prot_to_pdb


class RewardCache:
    """LRU sequence-keyed cache for reward model outputs.

    Attach to a reward model via ``reward_model.enable_cache(max_size)`` to
    avoid redundant scoring of identical sequences (common in beam search
    where siblings share nearly identical sequences).
    """

    def __init__(self, max_size: int = 5000):
        self._cache: dict[bytes, dict[str, Any]] = {}
        self._order: list[bytes] = []
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, key: bytes) -> dict[str, Any] | None:
        if key in self._cache:
            self._order.remove(key)
            self._order.append(key)
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, key: bytes, value: dict[str, Any]) -> None:
        if key in self._cache:
            self._order.remove(key)
        elif len(self._cache) >= self.max_size:
            oldest = self._order.pop(0)
            del self._cache[oldest]
        self._cache[key] = value
        self._order.append(key)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def __len__(self) -> int:
        return len(self._cache)


def _sequence_cache_key(residue_type: torch.Tensor) -> bytes:
    return residue_type.detach().cpu().numpy().tobytes()


def initialize_reward_model(inf_cfg: Any) -> Any | None:
    """Initialize reward model from configuration.

    Reward model is independent of search algorithm. It can be initialized
    regardless of whether search (beam-search, mcts, etc.) is used.

    Args:
        inf_cfg: Inference config with reward_model at inf_cfg.reward_model.

    Returns:
        Reward model instance or None if no reward model configured.
    """
    if hasattr(inf_cfg, "reward_model") and inf_cfg.reward_model is not None:
        reward_model_cfg = inf_cfg.reward_model
        reward_model = hydra.utils.instantiate(reward_model_cfg)
        logger.info(f"Reward model initialized: {type(reward_model).__name__}")
        return reward_model

    logger.warning("No reward model configured.")
    return None


def _extract_reward_components(reward_dict: dict[str, Any]) -> dict[str, float]:
    """Extract reward components from reward_dict for CSV storage.

    Takes all entries from reward_dict[REWARD_KEY] (e.g. AF2's plddt, pae, i_pae, etc.).
    Does not use any other keys from the top-level reward_dict.

    Args:
        reward_dict: Result from reward_model.score().

    Returns:
        Dict mapping component name to scalar value (e.g. {"af2_pae": 0.5, "af2_plddt": 0.8}).
    """
    components = {}
    reward_subdict = reward_dict.get(REWARD_KEY, {})
    for key, value in reward_subdict.items():
        if isinstance(value, torch.Tensor):
            components[key] = value.item() if value.numel() == 1 else float(value.mean().item())
        else:
            components[key] = float(value)
    return components


def compute_reward_from_samples(
    reward_model: Any | None,
    sample_prots: dict[str, torch.Tensor],
    target_hotspot_mask: torch.Tensor | None = None,
    ligand: Any | None = None,
) -> dict[str, torch.Tensor]:
    """Compute reward for given sample_prots using the reward model.

    Args:
        reward_model: CompositeRewardModel instance or None.
        sample_prots: Dict with 'coors', 'residue_type', optionally 'chain_index'.
        target_hotspot_mask: Optional [batch_size, n_target_res] mask for target hotspot residues.
        ligand: Ligand for write_prot_ligand_to_pdb (optional).

    Returns:
        Dict[str, Tensor] with keys:
            - "total_reward": Tensor [batch_size]
            - component names from reward_dict["reward"] (e.g. "af2_pae", "af2_plddt"): Tensor [batch_size]
    """
    batch_size = sample_prots["coors"].shape[0]
    device = sample_prots["coors"].device

    if reward_model is None:
        logger.error(
            "No reward model available. Returning zero rewards — search selection "
            "will be effectively random. Set a reward model for meaningful search."
        )
        return {TOTAL_REWARD_KEY: torch.zeros(batch_size, device=device)}

    # --- cache lookup ---
    cache: RewardCache | None = getattr(reward_model, "_reward_cache", None)
    cached_results: dict[int, dict[str, float]] = {}
    uncached_indices: list[int] = []

    if cache is not None:
        for i in range(batch_size):
            key = _sequence_cache_key(sample_prots["residue_type"][i])
            entry = cache.get(key)
            if entry is not None:
                cached_results[i] = entry
            else:
                uncached_indices.append(i)
        if cached_results:
            logger.debug(
                f"RewardCache: {len(cached_results)}/{batch_size} hits "
                f"(hit_rate={cache.hit_rate:.2%})"
            )
    else:
        uncached_indices = list(range(batch_size))

    target_chain, binder_chain = None, None
    temp_dir = tempfile.mkdtemp()
    temp_pdb_paths: dict[int, str] = {}

    try:
        # Write PDB only for uncached samples
        for i in uncached_indices:
            coors = sample_prots["coors"][i]
            residue_type = sample_prots["residue_type"][i]
            chain_index = (
                sample_prots.get("chain_index", [None] * batch_size)[i] if "chain_index" in sample_prots else None
            )
            creation_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            device_str = str(device).replace(":", "_")
            temp_pdb_path = os.path.join(temp_dir, f"temp_sample_{i}_{creation_time}_{device_str}.pdb")
            temp_pdb_paths[i] = temp_pdb_path

            if ligand is not None:
                write_prot_ligand_to_pdb(
                    coors=coors,
                    residue_type=residue_type,
                    ligand=ligand,
                    pdb_path=temp_pdb_path,
                )
            else:
                chain_index_np = chain_index.detach().cpu().numpy() if chain_index is not None else None
                write_prot_to_pdb(
                    prot_pos=coors.float().detach().cpu().numpy(),
                    aatype=residue_type.detach().cpu().numpy(),
                    file_path=temp_pdb_path,
                    chain_index=chain_index_np,
                    overwrite=True,
                    no_indexing=True,
                )

            if target_chain is None:
                target_chain, binder_chain = get_chain_ids_from_pdb(temp_pdb_path)

        # Score uncached samples
        component_keys: set[str] = set()
        scored_results: dict[int, dict[str, float]] = {}

        for i in uncached_indices:
            temp_pdb_path = temp_pdb_paths[i]
            chain_index_i = (
                sample_prots.get("chain_index", [None] * batch_size)[i] if "chain_index" in sample_prots else None
            )
            target_hotspot_mask_i = (
                target_hotspot_mask[i % target_hotspot_mask.shape[0]] if target_hotspot_mask is not None else None
            )

            reward_kwargs: dict[str, Any] = {
                "target_chain": target_chain,
                "binder_chain": binder_chain,
            }
            if target_hotspot_mask_i is not None:
                reward_kwargs["target_hotspot_mask"] = target_hotspot_mask_i
            if chain_index_i is not None:
                reward_kwargs["chain_index"] = chain_index_i

            reward_dict = reward_model.score(
                pdb_path=temp_pdb_path,
                requires_grad=False,
                **reward_kwargs,
            )
            total_reward = reward_dict[TOTAL_REWARD_KEY].item()
            components = _extract_reward_components(reward_dict)
            components[TOTAL_REWARD_KEY] = total_reward
            component_keys.update(components.keys())
            logger.debug(f"Sample {i}: reward = {total_reward}")
            scored_results[i] = components

            # Store in cache
            if cache is not None:
                key = _sequence_cache_key(sample_prots["residue_type"][i])
                cache.put(key, components)

        # Merge cached + scored
        all_components: list[dict[str, float]] = []
        for i in range(batch_size):
            if i in cached_results:
                all_components.append(cached_results[i])
                component_keys.update(cached_results[i].keys())
            else:
                all_components.append(scored_results[i])

        total_rewards_list = [c[TOTAL_REWARD_KEY] for c in all_components]

        result = {
            TOTAL_REWARD_KEY: torch.tensor(total_rewards_list, device=device, dtype=torch.float32),
        }
        for key in sorted(component_keys):
            if key == TOTAL_REWARD_KEY:
                continue
            vals = [all_components[i].get(key, float("nan")) for i in range(batch_size)]
            result[key] = torch.tensor(vals, device=device, dtype=torch.float32)

        logger.info(
            f"Computed rewards for {batch_size} samples "
            f"({len(uncached_indices)} scored, {len(cached_results)} cached). "
            f"Mean reward: {result[TOTAL_REWARD_KEY].mean().item():.4f}"
        )
        return result
    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")
