from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from colabdesign.af.alphafold.common import confidence, residue_constants
from colabdesign.af.loss import _get_con_loss, get_dgram_bins, get_ptm, mask_loss


def kabsch_align(P: jnp.ndarray, Q: jnp.ndarray, device: str = "cpu") -> jnp.ndarray:
    """Kabsch alignment using pure JAX operations.

    Args:
        P: Points to be aligned (N, 3).
        Q: Reference points (N, 3).

    Returns:
        Aligned P coordinates.
    """
    # Center the points
    P_mean = jnp.mean(P, axis=0)
    Q_mean = jnp.mean(Q, axis=0)
    P_centered = P - P_mean
    Q_centered = Q - Q_mean

    # Compute covariance matrix
    H = P_centered.T @ Q_centered

    # SVD decomposition
    U, S, Vh = jnp.linalg.svd(H, full_matrices=False)

    # Handle reflection case
    d = jnp.sign(jnp.linalg.det(Vh.T @ U.T))
    V = Vh.T
    U_new = U @ jnp.diag(jax.device_put(jnp.array([1, 1, d]), device))

    # Compute rotation matrix
    R = V @ U_new.T

    # Apply transformation
    P_aligned = (P - P_mean) @ R + Q_mean
    return P_aligned


def add_alignment_bb_ca_loss(model: Any, weight: float = 0.0, binder_only: bool = False) -> None:
    """Add backbone CA structure alignment loss to the model. RMSD between given structure and predicted CA positions.

    Args:
        model: The AF2 model to add the loss to.
        weight: Weight for the loss component.
        binder_only: If True, only compute loss on binder structure.
    """
    loss_name = "alignment_bb_ca" if not binder_only else "alignment_bb_ca_binder"

    def alignment_bb_ca_loss(params: dict[str, Any], aux: dict[str, Any]) -> dict[str, float]:
        """Loss function for backbone CA alignment.

        Args:
            params: Model params. Use params instead of inputs to support gradients w.r.t. structure.
            aux: Auxiliary model outputs.

        Returns:
            Dictionary containing the loss value.
        """
        assert params.get("struct") is not None, f"Structure is not provided for {loss_name}"

        # Get input and target positions
        input_pos = params["struct"][:, 1, :]  # Generated positions
        target_pos = aux["atom_positions"][:, 1, :]  # AF2 predicted CA positions
        if binder_only:
            target_pos = target_pos[model._target_len :]
            assert input_pos.shape[0] == target_pos.shape[0], (
                "Binder structure and generated structure must have the same length"
            )

        # Align input to target using Kabsch
        aligned_pos = kabsch_align(input_pos, target_pos, device=model._device)

        # Compute L2 loss on aligned positions
        diff = aligned_pos - target_pos
        loss = jnp.mean(jnp.sum(diff**2, axis=-1))

        return {loss_name: loss}

    model._callbacks["model"]["loss"].append(alignment_bb_ca_loss)
    model.opt["weights"][loss_name] = weight


def add_helix_binder_loss(model: Any, weight: float = 0.0) -> None:
    """Add helix binder loss to the model.

    Args:
        model: The AF2 model to add the loss to.
        weight: Weight for the loss component.
    """

    def binder_helicity(inputs: dict[str, Any], outputs: dict[str, Any]) -> dict[str, float]:
        if "offset" in inputs:
            offset = inputs["offset"]
        else:
            idx = inputs["residue_index"].flatten()
            offset = idx[:, None] - idx[None, :]

        # define distogram
        dgram = outputs["distogram"]["logits"]
        dgram_bins = get_dgram_bins(outputs)
        mask_2d = np.outer(
            np.append(np.zeros(model._target_len), np.ones(model._binder_len)),
            np.append(np.zeros(model._target_len), np.ones(model._binder_len)),
        )

        x = _get_con_loss(dgram, dgram_bins, cutoff=6.0, binary=True)
        if offset is None:
            if mask_2d is None:
                helix_loss = jnp.diagonal(x, 3).mean()
            else:
                helix_loss = jnp.diagonal(x * mask_2d, 3).sum() + (jnp.diagonal(mask_2d, 3).sum() + 1e-8)
        else:
            mask = offset == 3
            if mask_2d is not None:
                mask = jnp.where(mask_2d, mask, 0)
            helix_loss = jnp.where(mask, x, 0.0).sum() / (mask.sum() + 1e-8)

        return {"helix_binder": helix_loss}

    model._callbacks["model"]["loss"].append(binder_helicity)
    model.opt["weights"]["helix_binder"] = weight


def add_rg_loss(model: Any, weight: float = 0.1) -> None:
    """Add radius of gyration loss to the model.

    Args:
        model: The AF2 model to add the loss to.
        weight: Weight for the loss component.
    """

    def loss_fn(outputs: dict[str, Any]) -> dict[str, float]:
        xyz = outputs["structure_module"]
        ca = xyz["final_atom_positions"][:, residue_constants.atom_order["CA"]]
        ca = ca[-model._binder_len :]
        rg = jnp.sqrt(jnp.square(ca - ca.mean(0)).sum(-1).mean() + 1e-8)
        rg_th = 2.38 * ca.shape[0] ** 0.365

        rg = jax.nn.elu(rg - rg_th)
        return {"rg": rg}

    model._callbacks["model"]["loss"].append(loss_fn)
    model.opt["weights"]["rg"] = weight


def add_i_ptm_loss(model: Any, weight: float = 0.1) -> None:
    """Add interface pLDDT loss to the model.

    Args:
        model: The AF2 model to add the loss to.
        weight: Weight for the loss component.
    """

    def loss_iptm(inputs: dict[str, Any], outputs: dict[str, Any]) -> dict[str, float]:
        p = 1 - get_ptm(inputs, outputs, interface=True)
        i_ptm = mask_loss(p)
        return {"i_ptm": i_ptm}

    model._callbacks["model"]["loss"].append(loss_iptm)
    model.opt["weights"]["i_ptm"] = weight


def get_iptmenergy(inputs, outputs, tm_lambda):
    # default interface=True
    logits = outputs["predicted_aligned_error"]["logits"]
    asym_id = inputs["asym_id"]

    breaks = outputs["predicted_aligned_error"]["breaks"]
    bin_centers = confidence._calculate_bin_centers(breaks, use_jnp=True)

    residue_weights = jnp.ones(logits.shape[0])
    residue_weights.shape[0]
    clipped_num_res = jnp.maximum(residue_weights.sum(), 19)

    d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8
    weights = 1.0 / (1 + jnp.square(bin_centers) / jnp.square(d0))

    weighted_logits = logits + (tm_lambda * jnp.log(weights)[None, None, :])
    positional_energy = -jax.nn.logsumexp(weighted_logits, axis=-1)

    mask = (asym_id[:, None] != asym_id[None, :]).astype(jnp.float32)
    denom = jnp.maximum(mask.sum(), 1.0)
    energy = (positional_energy * mask).sum() / denom
    return energy


def add_i_ptm_energy_loss(model: Any, weight: float = 0.1, tm_lambda: float = 1) -> None:
    """Add interface pLDDT energy loss to the model.

    Args:
        model: The AF2 model to add the loss to.
        weight: Weight for the loss component.
        tm_lambda: Lambda for the TM score.
    """

    def loss_iptmenergy(inputs: dict[str, Any], outputs: dict[str, Any]) -> dict[str, float]:
        i_ptm_energy = get_iptmenergy(inputs, outputs, tm_lambda)  # want to minimize iptmenergy, so no 1 minus
        return {"i_ptm_energy": i_ptm_energy}

    model._callbacks["model"]["loss"].append(loss_iptmenergy)
    model.opt["weights"]["i_ptm_energy"] = weight


def add_termini_distance_loss(model: Any, weight: float = 0.1, threshold_distance: float = 7.0) -> None:
    """Add loss penalizing the distance between N and C termini.

    Args:
        model: The AF2 model to add the loss to.
        weight: Weight for the loss component.
        threshold_distance: Threshold distance for the loss.
    """

    def loss_fn(outputs: dict[str, Any]) -> dict[str, float]:
        xyz = outputs["structure_module"]
        ca = xyz["final_atom_positions"][:, residue_constants.atom_order["CA"]]
        ca = ca[-model._binder_len :]  # Considering only the last _binder_len residues

        # Extract N-terminus (first CA atom) and C-terminus (last CA atom)
        n_terminus = ca[0]
        c_terminus = ca[-1]

        # Compute the distance between N and C termini
        termini_distance = jnp.linalg.norm(n_terminus - c_terminus)

        # Compute the deviation from the threshold distance using ELU activation
        deviation = jax.nn.elu(termini_distance - threshold_distance)

        # Ensure the loss is never lower than 0
        termini_distance_loss = jax.nn.relu(deviation)
        return {"nc_termini": termini_distance_loss}

    # Append the loss function to the model callbacks
    model._callbacks["model"]["loss"].append(loss_fn)
    model.opt["weights"]["nc_termini"] = weight
