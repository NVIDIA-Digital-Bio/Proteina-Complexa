"""
Training batch handlers for conditioning, self-conditioning, and auxiliary tasks.
Extracted from Proteina class for separation of concerns.
"""

import random
from collections.abc import Callable
from typing import Any

from proteinfoundation.utils.fold_utils import mask_cath_code_by_level


def _safe_get(cfg: Any, key: str, default: Any) -> Any:
    """Get config value with safe default. Works with OmegaConf and plain dicts."""
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def handle_cath_code_cond(
    batch: dict,
    bs: int,
    fold_cond: bool,
    mask_T_prob: float,
    mask_A_prob: float,
    mask_C_prob: float,
) -> dict:
    """
    Progressively mask CATH code levels for fold conditioning during training.

    Args:
        batch: Batch with cath_code attribute.
        bs: Batch size.
        fold_cond: Whether fold conditioning is enabled.
        mask_T_prob: Probability to mask T level.
        mask_A_prob: Probability to mask A level.
        mask_C_prob: Probability to mask C level.

    Returns:
        Modified batch.
    """
    if fold_cond:
        cath_code_list = batch.cath_code
        for i in range(bs):
            cath_code_list[i] = mask_cath_code_by_level(cath_code_list[i], level="H")
            if random.random() < mask_T_prob:
                cath_code_list[i] = mask_cath_code_by_level(cath_code_list[i], level="T")
                if random.random() < mask_A_prob:
                    cath_code_list[i] = mask_cath_code_by_level(cath_code_list[i], level="A")
                    if random.random() < mask_C_prob:
                        cath_code_list[i] = mask_cath_code_by_level(cath_code_list[i], level="C")
        batch.cath_code = cath_code_list
    else:
        if "cath_code" in batch:
            batch.pop("cath_code")
    return batch


def handle_self_cond(
    batch: dict,
    training_cfg: Any,
    call_nn_fn: Callable,
    fm: Any,
) -> dict:
    """
    Add self-conditioning: with 50% probability, run NN and add x_sc to batch.

    Args:
        batch: Training batch.
        training_cfg: Config with n_recycle and self_cond.
        call_nn_fn: Function(batch, n_recycle) -> nn_out.
        fm: Flow matcher with nn_out_to_clean_sample_prediction(batch, nn_out).

    Returns:
        Modified batch (possibly with x_sc key).
    """
    n_recycle = _safe_get(training_cfg, "n_recycle", 0)
    if random.random() > 0.5 and _safe_get(training_cfg, "self_cond", False):
        nn_out = call_nn_fn(batch, n_recycle=n_recycle)
        x_1_pred = fm.nn_out_to_clean_sample_prediction(batch=batch, nn_out=nn_out)
        batch["x_sc"] = {k: x_1_pred[k].detach() for k in x_1_pred}
    return batch


# Deprecated
def handle_target_cond(batch: dict, target_cond: bool) -> dict:
    """
    Apply target conditioning: fix target coordinates and optionally center binder.

    Args:
        batch: Batch with x_0, x_1, chain_mask, mask, optionally binder_center.
        target_cond: Whether target conditioning is enabled.

    Returns:
        Modified batch.
    """
    if target_cond:
        target_mask = batch["chain_mask"]
        binder_mask = ~target_mask.logical_and(batch["mask"])
        batch["x_0"]["bb_ca"][target_mask] = batch["x_1"]["bb_ca"][target_mask]
        if batch.get("binder_center") is not None:
            x_mean_binder = batch["binder_center"] * binder_mask[..., None]
            batch["x_0"]["bb_ca"] = batch["x_0"]["bb_ca"] - x_mean_binder
    return batch


def handle_target_dropout(batch: dict, target_dropout_rate: float) -> dict:
    """
    Randomly drop all target-related keys from the batch.

    With probability ``target_dropout_rate``, removes every key whose name
    contains "target", forcing the model to generate without target conditioning.

    Args:
        batch: Training batch.
        target_dropout_rate: Probability of dropping target keys (0.0 to 1.0).

    Returns:
        Modified batch (possibly with target keys removed).
    """
    if target_dropout_rate > 0 and random.random() < target_dropout_rate:
        target_keys = [k for k in batch if "target" in k]
        for k in target_keys:
            del batch[k]
    return batch


def handle_motif_dropout(batch: dict, motif_dropout_rate: float) -> dict:
    """
    Randomly drop all motif-related keys from the batch.

    With probability ``motif_dropout_rate``, removes every key whose name
    contains "motif", forcing the model to generate without motif conditioning.

    Args:
        batch: Training batch.
        motif_dropout_rate: Probability of dropping motif keys (0.0 to 1.0).

    Returns:
        Modified batch (possibly with motif keys removed).
    """
    if motif_dropout_rate > 0 and random.random() < motif_dropout_rate:
        motif_keys = [k for k in batch if "motif" in k]
        for k in motif_keys:
            del batch[k]
    return batch


def handle_batch_conditioning(
    batch: dict,
    bs: int,
    training_cfg: Any,
    call_nn_fn: Callable,
    fm: Any,
) -> tuple[dict, int]:
    """
    Apply all training-time conditioning handlers with safe config access.

    Handles CATH code conditioning, self-conditioning, target conditioning,
    target dropout, folding/inverse folding, and recycling. Uses _safe_get for
    all config values so missing keys default to disabled/zero and training
    does not fail.

    Args:
        batch: Training batch.
        bs: Batch size.
        training_cfg: Training config (OmegaConf or dict). Keys: fold_cond,
            mask_T_prob, mask_A_prob, mask_C_prob, target_cond, self_cond,
            n_recycle, p_folding_n_inv_folding_iters, target_dropout_rate,
            motif_dropout_rate.
        call_nn_fn: Function(batch, n_recycle) -> nn_out for self-conditioning.
        fm: Flow matcher with nn_out_to_clean_sample_prediction.

    Returns:
        (modified_batch, n_recycle).
    """
    fold_cond = _safe_get(training_cfg, "fold_cond", False)
    mask_T_prob = _safe_get(training_cfg, "mask_T_prob", 0.5)
    mask_A_prob = _safe_get(training_cfg, "mask_A_prob", 0.5)
    mask_C_prob = _safe_get(training_cfg, "mask_C_prob", 0.5)
    # target_cond = _safe_get(training_cfg, "target_cond", False)
    target_dropout_rate = _safe_get(training_cfg, "target_dropout_rate", 0.0)
    motif_dropout_rate = _safe_get(training_cfg, "motif_dropout_rate", 0.0)
    n_recycle = handle_recycling(training_cfg)

    batch = handle_cath_code_cond(batch, bs, fold_cond, mask_T_prob, mask_A_prob, mask_C_prob)
    batch = handle_self_cond(batch, training_cfg, call_nn_fn, fm)
    # DEPRECATED: target conditioning is now handled via conditional features
    # in the data pipeline, not during the training step.
    # batch = handle_target_cond(batch, target_cond)
    batch = handle_target_dropout(batch, target_dropout_rate)
    batch = handle_motif_dropout(batch, motif_dropout_rate)
    batch = handle_folding_n_inverse_folding(batch, training_cfg)

    return batch, n_recycle


def handle_recycling(training_cfg: Any) -> int:
    """
    Get number of recycling steps (random between 0 and n_recycle for training).

    Returns:
        Number of recycle steps (0 if disabled).
    """
    n_recycle = _safe_get(training_cfg, "n_recycle", 0)
    if n_recycle == 0:
        return 0
    return random.randint(0, n_recycle)


def handle_folding_n_inverse_folding(batch: dict, training_cfg: Any) -> dict:
    """
    With probability p, enable folding or inverse folding iteration for the batch.

    Adds use_ca_coors_nm_feature (folding) or use_residue_type_feature
    (inverse folding) to batch, each with 50% of the probability mass.

    Args:
        batch: Training batch.
        training_cfg: Config with p_folding_n_inv_folding_iters.

    Returns:
        Modified batch.
    """
    batch["use_ca_coors_nm_feature"] = False
    batch["use_residue_type_feature"] = False
    prob = _safe_get(training_cfg, "p_folding_n_inv_folding_iters", 0.0)
    if random.random() < prob:
        if random.random() < 0.5:
            batch["use_ca_coors_nm_feature"] = True
        else:
            batch["use_residue_type_feature"] = True
    return batch
