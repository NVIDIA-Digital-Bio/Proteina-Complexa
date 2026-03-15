"""Search utilities shared by all search algorithms.

Batching / chunking helpers (avoid GPU OOM while maximising parallelism):

    ``replicate_batch``               – expand samples for best-of-n (repeat_interleave)
    ``tile_batch`` / ``tile_tensor_dict`` – expand samples for branching (repeat / tile)
    ``chunk_batch``                   – split a large batch dict into sub-batches
    ``chunked_partial_simulation``    – run ``partial_simulation`` respecting ``max_batch_size``

Tag / provenance helpers (travel with every sample for traceable file naming):

    ``make_initial_search_tags``      – create tags at noise initialisation
    ``expand_tags_for_branches``      – expand tags after branching
    ``select_tags``                   – pick tags after reward selection

Intermediate saving:

    ``decode_and_save_intermediates`` – decode latents and write PDBs per step
    ``save_search_intermediates``     – PDB writer (called by above)

Combination:

    ``combine_lookahead_and_final``   – merge lookahead and final results
    ``generate_samples_to_completion``– look-ahead rollout (uses ``chunked_partial_simulation``)

Separate from search_factory to avoid circular imports.
"""

import math
import os
from typing import Any

import numpy as np
import torch
from loguru import logger

from proteinfoundation.search.base_search import SearchContext
from proteinfoundation.utils.pdb_utils import write_prot_to_pdb
from proteinfoundation.utils.sample_utils import prepend_target_to_samples, sample_formatting
from proteinfoundation.utils.tensor_utils import concat_dict_tensors


def expand_hotspot_mask(
    batch_mask: torch.Tensor | None,
    target_size: int,
    nsamples: int,
) -> torch.Tensor | None:
    """Expand a batch-level hotspot mask for grouped-layout output.

    Search finals use grouped layout (all beams for sample 0 first, then
    sample 1, etc.).  ``repeat_interleave`` replicates each mask entry the
    right number of times to match that layout.

    Returns ``None`` when *batch_mask* is ``None``.  Returns the mask
    unchanged when *target_size* already equals *nsamples*.
    """
    if batch_mask is None or target_size == 0:
        return None
    if target_size == nsamples:
        return batch_mask
    repeats = (target_size + nsamples - 1) // nsamples
    return batch_mask.repeat_interleave(repeats, dim=0)[:target_size]


def replicate_batch(batch: dict[str, Any], replicas: int) -> dict[str, Any]:
    """Replicate every sample in *batch* ``replicas`` times along dim-0.

    Uses ``repeat_interleave`` so that replicas of the same original sample
    are adjacent::

        [s0_r0, s0_r1, …, s1_r0, s1_r1, …]

    Non-tensor values (lists, scalars) are repeated accordingly.
    """
    if replicas <= 1:
        return batch

    expanded: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            expanded[key] = value.repeat_interleave(replicas, dim=0)
        elif isinstance(value, list):
            expanded[key] = [v for v in value for _ in range(replicas)]
        elif key == "nsamples":
            expanded[key] = value * replicas
        elif key == "nres":
            expanded[key] = value
        else:
            expanded[key] = value
    return expanded


def chunk_batch(batch: dict[str, Any], max_batch_size: int) -> list[dict[str, Any]]:
    """Split *batch* into sub-batches of at most *max_batch_size* along dim-0.

    Returns a list of batch dicts.  Scalar fields (e.g. ``nres``) are copied
    as-is; ``nsamples`` is updated per chunk.
    """
    total = None
    for v in batch.values():
        if isinstance(v, torch.Tensor) and v.dim() > 0:
            total = v.shape[0]
            break
    if total is None or total <= max_batch_size:
        return [batch]

    chunks: list[dict[str, Any]] = []
    for start in range(0, total, max_batch_size):
        end = min(start + max_batch_size, total)
        chunk: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, (torch.Tensor, list)):
                chunk[key] = value[start:end]
            elif key == "nsamples":
                chunk[key] = end - start
            else:
                chunk[key] = value
        chunks.append(chunk)
    return chunks


def tile_batch(batch: dict[str, Any], tiles: int) -> dict[str, Any]:
    """Tile every sample in *batch* ``tiles`` times along dim-0.

    Uses ``.repeat()`` (not ``repeat_interleave``), so copies of the full
    batch are concatenated end-to-end::

        [s0, s1, …, s0, s1, …]  (``tiles`` copies)

    This is the layout needed for branching in beam/FK search where each
    branch starts from the same state but evolves independently.
    """
    if tiles <= 1:
        return batch

    expanded: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            repeat_dims = [tiles] + [1] * (value.dim() - 1)
            expanded[key] = value.repeat(*repeat_dims)
        elif isinstance(value, list) or key == "nsamples":
            expanded[key] = value * tiles
        elif key == "nres":
            expanded[key] = value
        else:
            expanded[key] = value
    return expanded


def tile_tensor_dict(d: dict[str, torch.Tensor], tiles: int) -> dict[str, torch.Tensor]:
    """Tile each tensor in *d* ``tiles`` times along dim-0.

    Helper for expanding ``xt`` / ``x_1_pred`` dicts during branching.
    """
    if tiles <= 1:
        return d
    return {k: v.repeat(tiles, *([1] * (v.dim() - 1))) for k, v in d.items()}


def make_initial_search_tags(prefix: str, nsamples: int, beam_width: int = 1) -> list[str]:
    """Create initial metadata tags when samples are first allocated.

    Format: ``{prefix}_orig{sample}[_bm{beam}]``

    Tags follow the same ordering as ``init_mask.repeat_interleave(beam_width)``::

        [orig0_bm0, orig0_bm1, …, orig0_bmW, orig1_bm0, …]

    For algorithms without beams (beam_width=1) the tag is just ``{prefix}_orig{s}``.
    """
    tags: list[str] = []
    for s in range(nsamples):
        if beam_width == 1:
            tags.append(f"{prefix}_orig{s}")
        else:
            for b in range(beam_width):
                tags.append(f"{prefix}_orig{s}_bm{b}")
    return tags


def expand_tags_for_branches(
    tags: list[str],
    beam_width: int,
    n_branch: int,
    start_step: int,
    end_step: int,
) -> list[str]:
    """Expand tags to match the branched tensor layout.

    Appends ``-s{start_step}to{end_step}br{branch}`` to each tag so the
    actual denoising range is recorded in the name (not just a loop index).

    Example after 3 checkpoints (0→50→100→200):
        ``beam_orig0_bm2-s0to50br3-s50to100br0-s100to200br1``

    The resulting tensor layout (and therefore the tag list) is::

        for replica_idx in range(beam_width):
            for branch_idx in range(n_branch):
                ... nsamples entries ...
    """
    assert len(tags) % beam_width == 0, (
        f"expand_tags_for_branches: len(tags)={len(tags)} not divisible by beam_width={beam_width}"
    )
    expanded: list[str] = []
    for replica_idx in range(beam_width):
        replica_tags = tags[replica_idx::beam_width]
        for branch_idx in range(n_branch):
            expanded.extend([f"{t}-s{start_step}to{end_step}br{branch_idx}" for t in replica_tags])
    return expanded


def select_tags(tags: list[str], indices) -> list[str]:
    """Pick tags using the same indices used for tensor selection."""
    idx_list = indices.cpu().tolist() if hasattr(indices, "cpu") else list(indices)
    return [tags[i] for i in idx_list]


def filter_lookahead_samples(
    sample_prots: dict[str, Any],
    rewards_dict: dict[str, torch.Tensor],
    tags: list[str],
    reward_threshold: float | None = None,
) -> dict[str, Any] | None:
    """Filter look-ahead samples by reward threshold, attach rewards and tags.

    Shared by beam search, FK steering, and MCTS to avoid duplicating the
    filter-mask-tag dance in every algorithm.

    Args:
        sample_prots: Decoded sample dict (may contain 'rewards' -- it will be
            excluded and replaced by *rewards_dict*).
        rewards_dict: Reward tensors keyed by component name.
        tags: Metadata tag per sample (same length as batch dim).
        reward_threshold: Minimum total reward to keep.  ``None`` keeps all.

    Returns:
        Dict ready to append to ``all_sample_prots``, or ``None`` when the
        threshold filters out every sample.
    """
    from proteinfoundation.rewards.base_reward import TOTAL_REWARD_KEY

    total_reward = rewards_dict[TOTAL_REWARD_KEY]
    if reward_threshold is not None:
        reward_mask = total_reward > reward_threshold
    else:
        reward_mask = torch.ones_like(total_reward, dtype=torch.bool)

    if not reward_mask.any():
        return None

    skip = {"rewards", "metadata_tag"}
    filtered = {k: v[reward_mask] if v is not None else None for k, v in sample_prots.items() if k not in skip}
    filtered["rewards"] = {k: v[reward_mask] for k, v in rewards_dict.items()}
    mask_list = reward_mask.cpu().tolist()
    filtered["metadata_tag"] = [tags[j] for j, m in enumerate(mask_list) if m]
    kept_tags = filtered["metadata_tag"]
    logger.debug(f"Lookahead filter: kept {len(kept_tags)}/{total_reward.shape[0]} (threshold={reward_threshold})")
    return filtered


def decode_and_save_intermediates(
    proteina: Any,
    xt: dict[str, torch.Tensor],
    init_mask: torch.Tensor,
    nsamples: int,
    tags: list[str],
    batch: dict[str, Any],
    trajectory_dir: str,
    step_idx: int,
) -> None:
    """Decode current latent states and save as PDBs for trajectory debugging.

    Combines the decode (``sample_formatting`` + ``prepend_target``) and the
    PDB-write steps into a single call so beam search and FK steering don't
    duplicate the logic.
    """
    batch_size = next(iter(xt.values())).shape[0]
    assert batch_size % nsamples == 0, (
        f"decode_and_save_intermediates: batch_size ({batch_size}) not divisible by nsamples ({nsamples})"
    )
    # seg_mask = init_mask.repeat_interleave(batch_size // nsamples, dim=0)
    # Must use .repeat() to match tile_tensor_dict layout (not repeat_interleave)
    seg_mask = init_mask.repeat(batch_size // nsamples, 1)
    decoded = sample_formatting(
        x=xt,
        extra_info={"mask": seg_mask},
        ret_mode="coors37_n_aatype",
        data_modes=list(proteina.cfg_exp.product_flowmatcher),
        autoencoder=getattr(proteina, "autoencoder", None),
    )
    has_ligand = hasattr(proteina, "ligand")
    if batch.get("prepend_target", False) and not has_ligand:
        decoded = prepend_target_to_samples(decoded, batch, repeat_mode="tile")
    save_search_intermediates(
        trajectory_dir=trajectory_dir,
        step_idx=step_idx,
        decoded=decoded,
        tags=tags,
        batch=batch,
        has_ligand=has_ligand,
    )


def save_search_intermediates(
    trajectory_dir: str,
    step_idx: int,
    decoded: dict[str, Any],
    tags: list[str],
    batch: dict[str, Any],
    has_ligand: bool = False,
) -> None:
    """Save intermediate denoising states as PDBs, named by their metadata tag.

    Each sample is written to ``trajectory_dir/step_{step_idx}/{tag}.pdb``.
    This gives full visibility into the search trajectory without a separate
    manifest -- the filename *is* the provenance.

    Args:
        trajectory_dir: Root directory for trajectory output.
        step_idx: Search step index (used for the subdirectory name).
        decoded: Dict with at least 'coors' [N, n, 37, 3] and
            'residue_type' [N, n].  Optionally 'chain_index' [N, n].
        tags: Metadata tag strings, one per sample (length N).
        batch: Original batch (used to check prepend_target).
        has_ligand: Whether the model has a ligand (affects chain_index).
    """
    step_dir = os.path.join(trajectory_dir, f"step_{step_idx}")
    os.makedirs(step_dir, exist_ok=True)

    coors = decoded["coors"]
    residue_type = decoded["residue_type"]
    chain_index = decoded.get("chain_index")

    for i in range(coors.shape[0]):
        assert i < len(tags), (
            f"save_search_intermediates: sample index {i} out of range for "
            f"tags (len={len(tags)}), expected {coors.shape[0]} tags"
        )
        tag = tags[i]
        chain_idx = chain_index[i].cpu().numpy() if chain_index is not None else np.ones(coors.shape[1])
        write_prot_to_pdb(
            prot_pos=coors[i].float().detach().cpu().numpy(),
            aatype=residue_type[i].detach().cpu().numpy(),
            file_path=os.path.join(step_dir, f"{tag}.pdb"),
            chain_index=chain_idx,
            overwrite=True,
            no_indexing=True,
        )

    logger.debug(f"Saved {coors.shape[0]} intermediates to {step_dir}")


def combine_lookahead_and_final(
    lookahead: dict[str, Any] | None,
    final: dict[str, Any],
    final_rewards: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Combine look-ahead and final samples into a single sample_prots dict.

    Search algorithms return {"lookahead": ..., "final": ...}. Look-ahead samples
    are fully generated (t=1) but produced during search (roll-outs at checkpoints),
    with rewards computed during search. Final samples are the selected output.

    Args:
        lookahead: Sample dict with rewards (from search), or None.
        final: Final sample dict (no rewards; caller computes).
        final_rewards: Rewards for final samples as Dict[str, Tensor].

    Returns:
        Combined dict with coors, residue_type, mask, rewards (dict),
        sample_type, and metadata_tag.
    """
    tensor_keys = ["coors", "residue_type", "mask", "chain_index"]
    parts = []
    lookahead_rewards = None

    if lookahead is not None:
        n_lookahead = lookahead["coors"].shape[0]
        la_part = {k: v for k, v in lookahead.items() if k in tensor_keys}
        parts.append(la_part)
        if "rewards" in lookahead:
            lookahead_rewards = lookahead["rewards"]

    final_part = {k: v for k, v in final.items() if k in tensor_keys}
    parts.append(final_part)
    n_final = final["coors"].shape[0]

    combined = concat_dict_tensors(parts, dim=0)

    reward_parts = []
    if lookahead_rewards is not None:
        reward_parts.append(lookahead_rewards)
    if final_rewards is not None:
        reward_parts.append(final_rewards)
    if reward_parts:
        combined["rewards"] = concat_dict_tensors(reward_parts, dim=0)

    sample_types: list[str] = []
    if lookahead is not None:
        sample_types.extend(["lookahead"] * n_lookahead)
    sample_types.extend(["final"] * n_final)
    combined["sample_type"] = sample_types

    # metadata_tag is a List[str], not a tensor – concatenate separately
    meta_tags: list[str] = []
    if lookahead is not None and "metadata_tag" in lookahead:
        meta_tags.extend(lookahead["metadata_tag"])
    if "metadata_tag" in final:
        meta_tags.extend(final["metadata_tag"])
    if meta_tags:
        combined["metadata_tag"] = meta_tags

    return combined


_TENSOR_KEYS = ["coors", "residue_type", "mask", "chain_index"]


def clone_sample_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Deep-clone a sample dict so in-place refinement cannot overwrite originals.

    Recursively clones tensors and nested dicts/lists.  Scalar values and
    strings are copied by reference (immutable).
    """
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.clone()
        elif isinstance(v, dict):
            out[k] = clone_sample_dict(v)
        elif isinstance(v, list):
            out[k] = [x.clone() if isinstance(x, torch.Tensor) else x for x in v]
        else:
            out[k] = v
    return out


def append_samples(
    combined: dict[str, Any],
    samples: dict[str, Any],
    rewards: dict[str, torch.Tensor] | None = None,
    sample_type: str = "unrefined",
) -> None:
    """Append *samples* to an already-combined output dict in-place.

    Used to attach pre-refinement copies alongside refined samples so both
    versions are available in the final output with distinct ``sample_type``
    markers (e.g. ``final_unrefined``).
    """
    n = samples["coors"].shape[0]
    for k in _TENSOR_KEYS:
        if k in samples and k in combined:
            combined[k] = torch.cat([combined[k], samples[k]], dim=0)

    if rewards is not None:
        if "rewards" in combined:
            combined["rewards"] = concat_dict_tensors(
                [combined["rewards"], rewards],
                dim=0,
            )
        else:
            combined["rewards"] = rewards

    combined.setdefault("sample_type", []).extend([sample_type] * n)

    if "metadata_tag" in samples:
        tags = [f"{t}_{sample_type}" for t in samples["metadata_tag"]]
    else:
        tags = [f"{sample_type}_{i}" for i in range(n)]
    combined.setdefault("metadata_tag", []).extend(tags)


def chunked_partial_simulation(
    proteina: Any,
    batch: dict[str, Any],
    xt: dict[str, torch.Tensor],
    x_1_pred: dict[str, torch.Tensor] | None,
    mask: torch.Tensor,
    max_batch_size: int | None,
    **sim_kwargs,
):
    """Run ``partial_simulation`` respecting *max_batch_size*.

    All inputs (``batch``, ``xt``, ``x_1_pred``, ``mask``) must already be
    expanded to the same dim-0 size.  When the total exceeds
    *max_batch_size* the inputs are sliced into contiguous chunks, each
    processed separately, and the results concatenated.

    Returns:
        ``(result_xt, result_x_1_pred)`` – same as ``partial_simulation``.
    """
    total = next(iter(xt.values())).shape[0]

    # ── dimension sanity checks ─────────────────────────────────────
    assert mask.shape[0] == total, f"chunked_partial_sim: mask dim-0 ({mask.shape[0]}) != xt dim-0 ({total})"
    batch_dim = next(
        (v.shape[0] for v in batch.values() if isinstance(v, torch.Tensor) and v.dim() > 0),
        total,
    )
    assert batch_dim == total, f"chunked_partial_sim: batch tensor dim-0 ({batch_dim}) != xt dim-0 ({total})"
    if x_1_pred is not None:
        pred_dim = next(iter(x_1_pred.values())).shape[0]
        assert pred_dim == total, f"chunked_partial_sim: x_1_pred dim-0 ({pred_dim}) != xt dim-0 ({total})"

    if max_batch_size is None or total <= max_batch_size:
        return proteina.fm.partial_simulation(
            batch=batch,
            x=xt,
            x_1_pred=x_1_pred,
            mask=mask,
            **sim_kwargs,
        )

    logger.debug(f"[chunked_partial_sim] Splitting {total} candidates into chunks of {max_batch_size}")
    xt_parts: list[dict] = []
    pred_parts: list[dict] = []
    for start in range(0, total, max_batch_size):
        end = min(start + max_batch_size, total)
        c_xt = {k: v[start:end] for k, v in xt.items()}
        c_pred = {k: v[start:end] for k, v in x_1_pred.items()} if x_1_pred is not None else None
        c_mask = mask[start:end]
        c_batch: dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, (torch.Tensor, list)):
                c_batch[k] = v[start:end]
            elif k == "nsamples":
                c_batch[k] = end - start
            else:
                c_batch[k] = v

        r_xt, r_pred = proteina.fm.partial_simulation(
            batch=c_batch,
            x=c_xt,
            x_1_pred=c_pred,
            mask=c_mask,
            **sim_kwargs,
        )
        xt_parts.append(r_xt)
        if r_pred is not None:
            pred_parts.append(r_pred)

    return (
        concat_dict_tensors(xt_parts, dim=0),
        concat_dict_tensors(pred_parts, dim=0) if pred_parts else None,
    )


def generate_samples_to_completion(
    search_ctx: SearchContext,
    proteina: Any,
    xt: dict[str, torch.Tensor],
    current_step: int,
    mask: torch.Tensor,
    x_1_pred: dict[str, torch.Tensor] | None = None,
) -> dict:
    """Look-ahead: run flow from current step to completion, then decode.

    Continues the denoising trajectory from *current_step* to the final
    step, respecting ``max_batch_size`` to avoid OOM.  The batch
    conditioning and mask are tiled to match the expanded ``xt`` and then
    processed through ``chunked_partial_simulation``.

    Args:
        search_ctx: Shared search state built by
            ``BaseSearch.build_search_context``.  Provides the
            conditioning batch, schedule tensors, and simulation
            parameters that were previously read from
            ``proteina._current_*`` attributes.
        proteina: Proteina instance — only used for
            ``proteina.fm.partial_simulation`` (via
            ``chunked_partial_simulation``), ``proteina.cfg_exp``,
            and ``proteina.autoencoder``.
        xt: Current state dict (may be larger than ``mask.shape[0]``
            due to branching).
        current_step: Step to continue from.
        mask: Residue mask [nsamples, n_residues].
        x_1_pred: Optional[Dict[str, torch.Tensor]] = None,
    Returns:
        Final sample_prots dict with coors, residue_type, mask, etc.
    """
    batch_size = next(iter(xt.values())).shape[0]
    nsamples = mask.shape[0]
    final_step = search_ctx.nsteps

    n_tiles = math.ceil(batch_size / nsamples)
    expanded_batch = tile_batch(search_ctx.current_batch, n_tiles)
    expanded_mask = mask.repeat(n_tiles, 1)

    # Trim to exact batch_size when batch_size is not a perfect multiple
    if n_tiles * nsamples > batch_size:
        for k, v in expanded_batch.items():
            if isinstance(v, (torch.Tensor, list)):
                expanded_batch[k] = v[:batch_size]
        expanded_batch["nsamples"] = batch_size
        expanded_mask = expanded_mask[:batch_size]

    assert expanded_mask.shape[0] == batch_size, (
        f"generate_to_completion: expanded_mask dim-0 ({expanded_mask.shape[0]}) != batch_size ({batch_size})"
    )
    logger.debug(
        f"[generate_to_completion] {batch_size} samples, "
        f"nsamples={nsamples}, n_tiles={n_tiles}, "
        f"max_batch_size={search_ctx.max_batch_size}"
    )

    final_xt, _ = chunked_partial_simulation(
        proteina,
        batch=expanded_batch,
        xt=xt,
        x_1_pred=x_1_pred,  # Before we were skipping this so it would be set to None for the first step
        mask=expanded_mask,
        max_batch_size=search_ctx.max_batch_size,
        predict_for_sampling=search_ctx.predict_fn,
        start_step=current_step,
        end_step=final_step,
        ts=search_ctx.ts,
        gt=search_ctx.gt,
        self_cond=search_ctx.self_cond,
        simulation_step_params=search_ctx.simulation_step_params,
        device=search_ctx.device,
        guidance_w=search_ctx.guidance_w,
        ag_ratio=search_ctx.ag_ratio,
    )

    final_sample_prots = sample_formatting(
        x=final_xt,
        extra_info={"mask": expanded_mask},
        ret_mode="coors37_n_aatype",
        data_modes=list(proteina.cfg_exp.product_flowmatcher),
        autoencoder=getattr(proteina, "autoencoder", None),
    )

    if search_ctx.current_batch.get("prepend_target", False) and not hasattr(proteina, "ligand"):
        # Lookahead rollouts use tile layout (branches tile the batch
        # end-to-end), so repeat_mode="tile" is needed here.
        final_sample_prots = prepend_target_to_samples(final_sample_prots, search_ctx.current_batch, repeat_mode="tile")

    return final_sample_prots
