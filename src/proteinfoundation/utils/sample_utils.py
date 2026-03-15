"""
Sample formatting and target prepending utilities.
Extracted from Proteina class for separation of concerns.
"""

from typing import Any

import numpy as np
import torch
from torch import Tensor

from proteinfoundation.utils.coors_utils import nm_to_ang, trans_nm_to_atom37
from proteinfoundation.utils.pdb_utils import create_full_prot, to_pdb


def get_clean_sample(
    batch: dict[str, torch.Tensor],
    dm: str,
    autoencoder: Any | None = None,
) -> torch.Tensor:
    """
    Get clean sample for a given data mode.

    Args:
        batch: Batch to get clean sample from.
        dm: Data mode ('bb_ca' or 'local_latents').
        autoencoder: AutoEncoder instance required for dm=='local_latents'.

    Returns:
        Clean sample tensor for the given data mode.
    """
    if dm == "bb_ca":
        return batch["coords_nm"][:, :, 1, :]  # [b, n, 3]
    if dm == "local_latents":
        if autoencoder is None:
            raise ValueError("autoencoder required for local_latents data mode")
        encoded_batch = autoencoder.encode(batch)
        return encoded_batch["z_latent"]
    raise ValueError(f"Loading clean samples from data mode {dm} not supported.")


def add_clean_samples(
    batch: dict[str, torch.Tensor],
    product_flowmatcher: Any,
    autoencoder: Any | None = None,
) -> dict[str, torch.Tensor]:
    """
    Add clean samples for all data modes to the batch.

    Args:
        batch: Batch to add clean samples to.
        product_flowmatcher: Config dict with data mode keys.
        autoencoder: AutoEncoder instance (required for local_latents mode).

    Returns:
        Batch with 'x_1' key added.
    """
    batch["x_1"] = {dm: get_clean_sample(batch, dm, autoencoder) for dm in product_flowmatcher}
    return batch


def sample_formatting(
    x: dict[str, Tensor],
    extra_info: dict[str, Tensor],
    ret_mode: str,
    data_modes: list[str],
    autoencoder: Any | None = None,
):
    """
    Decoding function: convert flow matcher latent samples to structure format.

    Supports ret_modes: 'samples', 'atom37', 'pdb_string', 'coors37_n_aatype'.

    Args:
        x: Sample dict from flow matcher.
        extra_info: Dict with 'mask' key.
        ret_mode: Target format.
        data_modes: List of data modes (e.g. ['bb_ca'] or ['bb_ca', 'local_latents']).
        autoencoder: AutoEncoder instance (required for local_latents mode).

    Returns:
        Formatted sample in requested mode.
    """
    data_modes = sorted(data_modes)
    mask = extra_info["mask"]
    if data_modes == ["bb_ca"]:
        return format_sample_bb_ca(x=x, ret_mode=ret_mode, mask=mask)
    if data_modes == ["bb_ca", "local_latents"]:
        if autoencoder is None:
            raise ValueError("autoencoder required for local_latents data mode")
        return format_sample_local_latents(x=x, ret_mode=ret_mode, mask=mask, autoencoder=autoencoder)
    raise NotImplementedError(f"Format for data_modes {data_modes} not implemented")


def prepend_target_to_samples(
    sample_prots: dict[str, Tensor],
    batch: dict[str, Tensor],
    repeat_mode: str = "interleave",
) -> dict[str, Tensor]:
    """Prepend the target protein to each generated binder sample.

    In binder design the output PDB must contain both the target and the
    designed binder.  When a search algorithm produces more samples than
    the original batch size (e.g. beam_width=4 gives 4x more), the
    target tensors are expanded to match.  *How* they are expanded
    depends on the sample layout:

    **Two layouts exist in the codebase:**

    *Grouped layout* (``repeat_mode="interleave"``, **the default**)::

        [s0, s0, s0, s0, s1, s1, s1, s1, …]   ← .repeat_interleave()

      Used for **final outputs** of every search algorithm that
      produces more samples than the batch (beam search, FK steering,
      best-of-n).  All beams/replicas for the same original sample
      are contiguous.  This is the safe default for any new algorithm.

    *Tile layout* (``repeat_mode="tile"``, must be passed explicitly)::

        [s0, s1, s2, s0, s1, s2, …]     ← .repeat()

      Used only *during* search for branched candidates and lookahead
      rollouts (``generate_samples_to_completion``).  The full batch
      is repeated end-to-end so independent noise seeds diverge.

    **Quick reference:**

    ========================  ===========  ==============  ==============
    Caller                    Layout       repeat_mode     Expansion?
    ========================  ===========  ==============  ==============
    SinglePass finals         1:1          (default)       No
    MCTS finals               1:1          (default)       No
    BeamSearch finals         grouped      (default)       Yes
    FKSteering finals         grouped      (default)       Yes
    BestOfN finals            grouped      (default)       Yes
    Lookahead rollouts        tile         "tile" (explicit)  Yes
    Intermediate debug PDBs   tile         "tile" (explicit)  Yes
    ========================  ===========  ==============  ==============

    The default ``"interleave"`` is safe for all final outputs.  Only
    internal search helpers (lookahead rollouts, debug intermediates)
    need to pass ``repeat_mode="tile"`` explicitly.

    Args:
        sample_prots: Generated samples with ``coors``, ``residue_type``,
            ``mask``, and optionally ``chain_index``.
        batch: Original batch with ``x_target``, ``seq_target``,
            ``seq_target_mask``, ``target_chains``.
        repeat_mode: ``"interleave"`` (default, .repeat_interleave) or
            ``"tile"`` (.repeat).  See layout explanation above.

    Returns:
        *sample_prots* modified in-place with target prepended along
        the residue dimension.
    """
    sample_batch_size = sample_prots["coors"].shape[0]
    batch_size = batch["x_target"].shape[0]

    if sample_batch_size != batch_size:
        repeat_factor = sample_batch_size // batch_size
        if sample_batch_size % batch_size != 0:
            raise ValueError(f"Sample batch size {sample_batch_size} must be a multiple of batch size {batch_size}")
        _keys = ["x_target", "seq_target", "seq_target_mask", "target_chains"]
        if repeat_mode == "interleave":
            # FIX: beam search / FK steering output is grouped by sample
            # (all beams for sample 0, then all for sample 1, …) so targets
            # must be repeated the same way via repeat_interleave.
            repeated_target = {k: batch[k].repeat_interleave(repeat_factor, dim=0) for k in _keys}
        else:
            repeated_target = {k: batch[k].repeat(repeat_factor, *([1] * (batch[k].dim() - 1))) for k in _keys}
    else:
        repeated_target = batch

    sample_prots["coors"] = torch.cat([nm_to_ang(repeated_target["x_target"]), sample_prots["coors"]], dim=1)
    sample_prots["chain_index"] = torch.cat(
        [
            repeated_target["target_chains"],
            torch.ones_like(sample_prots["residue_type"])
            * (repeated_target["target_chains"].max(dim=-1, keepdim=True).values + 1),
        ],
        dim=1,
    )
    sample_prots["residue_type"] = torch.cat([repeated_target["seq_target"], sample_prots["residue_type"]], dim=1)
    sample_prots["mask"] = torch.cat([repeated_target["seq_target_mask"], sample_prots["mask"]], dim=1)
    sample_prots["coors"][~sample_prots["mask"]] = 0.0

    return sample_prots


def format_sample_bb_ca(
    x: dict[str, torch.Tensor],
    ret_mode: str,
    mask: torch.Tensor,
):
    """
    Format samples that contain only bb_ca (backbone CA) data mode.

    Args:
        x: Sample dict with key 'bb_ca'
        ret_mode: One of 'samples', 'atom37', 'coors37_n_aatype', 'pdb_string'
        mask: Residue mask [batch, n]

    Returns:
        Formatted sample in requested mode
    """
    if ret_mode == "samples":
        return x

    if ret_mode == "atom37":
        return trans_nm_to_atom37(x["bb_ca"].float())

    if ret_mode == "coors37_n_aatype":
        coors = trans_nm_to_atom37(x["bb_ca"].float()) * mask[..., None, None]
        residue_type = torch.zeros_like(coors)[..., 0, 0] * mask
        return {
            "coors": coors,
            "residue_type": residue_type.long(),
            "mask": mask,
        }

    if ret_mode == "pdb_string":
        pdb_strings = []
        coors = trans_nm_to_atom37(x["bb_ca"]).float().detach().cpu().numpy()
        residue_type = np.zeros_like(coors[:, :, 0, 0])
        atom37_mask = np.zeros_like(coors[:, :, :, 0])
        atom37_mask[:, :, 1] = 1.0
        mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
        atom37_mask = atom37_mask * mask_np[..., None]
        n = coors.shape[-3]

        for i in range(coors.shape[0]):
            prot = create_full_prot(
                atom37=coors[i, ...],
                atom37_mask=atom37_mask[i, ...],
                aatype=residue_type[i, ...],
            )
            pdb_string = to_pdb(prot=prot)
            pdb_strings.append({"pdb_string": pdb_string, "nres": n})
        return pdb_strings

    raise NotImplementedError(f"{ret_mode} format for data modes `[bb_ca]` not implemented")


def format_sample_local_latents(
    x: dict[str, torch.Tensor],
    ret_mode: str,
    mask: torch.Tensor,
    autoencoder: Any,
):
    """
    Format samples containing bb_ca and local_latents using autoencoder decoder.

    Args:
        x: Sample dict with keys 'bb_ca', 'local_latents'
        ret_mode: One of 'samples', 'coors37_n_aatype', 'pdb_string'
        mask: Residue mask [batch, n]
        autoencoder: AutoEncoder instance with decode() method

    Returns:
        Formatted sample in requested mode
    """
    output_decoder = autoencoder.decode(z_latent=x["local_latents"], ca_coors_nm=x["bb_ca"], mask=mask)

    if ret_mode == "samples":
        return x

    if ret_mode == "coors37_n_aatype":
        return {
            "coors": nm_to_ang(output_decoder["coors_nm"]),
            "residue_type": output_decoder["residue_type"],
            "mask": output_decoder["residue_mask"],
        }

    if ret_mode == "pdb_string":
        pdb_strings = []
        coors_atom_37 = nm_to_ang(output_decoder["coors_nm"]).float().detach().cpu().numpy()
        residue_type = output_decoder["residue_type"]
        atom_mask = output_decoder["atom_mask"]
        n = coors_atom_37.shape[-3]

        for i in range(atom_mask.shape[0]):
            prot = create_full_prot(
                atom37=coors_atom_37[i, ...],
                atom37_mask=atom_mask[i, ...],
                aatype=residue_type[i, ...],
            )
            pdb_string = to_pdb(prot=prot)
            pdb_strings.append({"pdb_string": pdb_string, "nres": n})
        return pdb_strings

    raise NotImplementedError(f"{ret_mode} format for data modes `[bb_ca, latent_locals]` not implemented")
