"""SequenceHallucination algorithm for protein generation.

This module provides sequence hallucination for improving protein structures
using ColabDesign's AlphaFold2 optimization pipeline.
"""

import math
import os
import shutil
import tempfile
import time
from typing import Any

import jax
import torch
from colabdesign import mk_afdesign_model
from loguru import logger

from proteinfoundation.rewards.alphafold2_reward_utils import add_helix_binder_loss, add_i_ptm_loss, add_rg_loss
from proteinfoundation.utils.pdb_utils import get_chain_ids_from_pdb, load_pdb, write_prot_to_pdb
from proteinfoundation.utils.tensor_utils import concat_dict_tensors

_DEFAULT_LOSS_WEIGHTS: dict[str, float] = {
    "pae": 0.4,
    "plddt": 0.1,
    "i_pae": 0.1,
    "con": 1.0,
    "i_con": 1.0,
    "dgram_cce": 0.0,
    "rg": 0.3,
    "i_ptm": 0.05,
    "helix_binder": -0.3,
}


class SequenceHallucination:
    """SequenceHallucination refinement algorithm.

    Optimises binder sequences using ColabDesign's AF2 design pipeline.
    Three optional stages run in order:
      Stage 2-3: Softmax + one-hot optimisation (``enable_soft_optimization``)
      Stage 4:   PSSM semigreedy optimisation (``enable_greedy_optimization``)

    If a sample fails during refinement (e.g. residue count mismatch from
    ColabDesign), the original unrefined structure is kept and processing
    continues for remaining samples.
    """

    def __init__(self, proteina_instance: Any, inf_cfg: Any) -> None:
        self.proteina = proteina_instance
        self.inf_cfg = inf_cfg

    # ------------------------------------------------------------------
    # Hotspot parsing  (NEW -- original passed hotspot=None)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_hotspots(
        target_hotspot_mask: torch.Tensor | None,
        chain_index: torch.Tensor,
        res_mask: torch.Tensor,
    ) -> list[str] | None:
        """Convert a boolean hotspot mask tensor into ColabDesign hotspot strings.

        ColabDesign expects hotspot as a list of target-chain residue indices
        (0-based integers).  The mask is True for residues that are hotspots.

        Returns None when no hotspots are available.
        """
        if target_hotspot_mask is None:
            return None

        # BUG FIX: guard against fully-padded samples where res_mask is all
        # False -- indexing chain_index[res_mask.bool()][0] would crash.
        if not res_mask.any():
            return None

        mask_bool = target_hotspot_mask.bool() & res_mask.bool()
        if not mask_bool.any():
            return None

        target_chain_id = chain_index[res_mask.bool()][0].item()
        hotspot_indices: list[str] = []
        target_pos = 0
        for j in range(res_mask.shape[0]):
            if not res_mask[j]:
                continue
            if chain_index[j].item() == target_chain_id:
                if mask_bool[j]:
                    hotspot_indices.append(str(target_pos))
                target_pos += 1

        return hotspot_indices if hotspot_indices else None

    # ------------------------------------------------------------------
    # Loss configuration
    # ------------------------------------------------------------------

    _BUILTIN_WEIGHT_KEYS = {"pae", "plddt", "i_pae", "con", "i_con", "dgram_cce"}

    @staticmethod
    def _set_builtin_weights(af_model: Any, loss_weights: dict[str, float]) -> None:
        """(Re-)set opt["weights"] for ColabDesign's built-in loss terms.

        Must be called after every ``prep_inputs`` because the internal
        ``restart()`` resets ``opt`` to its saved defaults.
        """
        af_model.opt["weights"].update(
            {k: loss_weights[k] for k in SequenceHallucination._BUILTIN_WEIGHT_KEYS if k in loss_weights}
        )

    @staticmethod
    def _register_loss_callbacks(af_model: Any, loss_weights: dict[str, float]) -> None:
        """Append custom loss callbacks to the AF2 model **once**.

        BUG FIX: Each ``add_*_loss`` call appends to
        ``af_model._callbacks["model"]["loss"]``.  ColabDesign never
        clears this list between ``prep_inputs`` calls.  The original
        code avoided this by creating a fresh af_model per sample; we
        now create the model once for performance, so callbacks must be
        registered exactly once -- calling this again would duplicate
        every callback and corrupt the loss.
        """
        add_rg_loss(af_model, loss_weights.get("rg", 0.0))
        add_i_ptm_loss(af_model, loss_weights.get("i_ptm", 0.0))
        add_helix_binder_loss(af_model, loss_weights.get("helix_binder", 0.0))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def refine(
        self,
        sample_prots: dict[str, torch.Tensor],
        target_hotspot_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Refine protein samples using sequence hallucination.

        Args:
            sample_prots: Dictionary containing at least ``coors``,
                ``residue_type``, ``chain_index``, and ``mask``.  All
                other keys are preserved in the output.
            target_hotspot_mask: Optional boolean tensor of shape
                ``[batch_size, n_residues]`` marking target hotspot
                residues.

        Returns:
            A copy of *sample_prots* with ``coors`` and ``residue_type``
            replaced by the refined versions for samples that succeeded.
            Samples that failed refinement retain their original values.
        """
        bs = sample_prots["coors"].shape[0]
        device_id = torch.cuda.current_device()
        jax_device = jax.devices("gpu")[device_id]

        ref_cfg = self.inf_cfg.refinement
        n_hard_iters = ref_cfg.get("n_hard_iters", 5)
        n_temp_iters = ref_cfg.get("n_temp_iters", 45)
        n_greedy_iters = ref_cfg.get("n_greedy_iters", 15)
        n_recycles = ref_cfg.get("n_recycles", 3)
        enable_soft = ref_cfg.get("enable_soft_optimization", True)
        enable_greedy = ref_cfg.get("enable_greedy_optimization", True)
        greedy_percentage = ref_cfg.get("greedy_percentage", 1)

        # IMPROVEMENT: loss weights are now configurable via
        # refinement.loss_weights in the YAML; defaults match the
        # original hardcoded values exactly.
        user_weights = ref_cfg.get("loss_weights", {})
        if hasattr(user_weights, "items"):
            user_weights = dict(user_weights)
        else:
            user_weights = {}
        loss_weights = {**_DEFAULT_LOSS_WEIGHTS, **user_weights}

        logger.info(f"Refinement: {bs} samples, soft={enable_soft}, greedy={enable_greedy}")

        # IMPROVEMENT: temp dir is now cleaned up in a finally block;
        # the original left it behind.
        temp_dir = tempfile.mkdtemp()
        try:
            target_chain, binder_chain = None, None

            # BUG FIX (performance): the original created a new
            # mk_afdesign_model per sample which is correct but very
            # slow (reloads weights each time).  We create the model
            # once and guard against callback accumulation via the
            # _register_loss_callbacks / _set_builtin_weights split.
            af_model = mk_afdesign_model(
                protocol="binder",
                debug=False,
                data_dir=os.environ.get("AF2_DIR"),
                use_multimer=True,
                num_recycles=n_recycles,
                use_initial_guess=False,
                use_initial_atom_pos=False,
                best_metric="loss",
                device=jax_device,
            )

            refined_sample_prots: list[dict[str, torch.Tensor]] = []
            callbacks_registered = False

            for i in range(bs):
                t0 = time.time()

                coors = sample_prots["coors"][i]
                residue_type = sample_prots["residue_type"][i]
                chain_index = sample_prots["chain_index"][i]
                res_mask = sample_prots["mask"][i].bool()
                n = int(res_mask.sum().item())

                # BUG FIX: wrap per-sample processing so a single
                # failure (e.g. ColabDesign residue mismatch) does not
                # abort the entire batch.
                try:
                    temp_pdb_path = os.path.join(temp_dir, f"temp_sample_{i}.pdb")
                    write_prot_to_pdb(
                        prot_pos=coors.detach().cpu().numpy(),
                        aatype=residue_type.detach().cpu().numpy(),
                        file_path=temp_pdb_path,
                        chain_index=(chain_index.detach().cpu().numpy() if chain_index is not None else None),
                        overwrite=True,
                        no_indexing=True,
                    )

                    if target_chain is None:
                        target_chain, binder_chain = get_chain_ids_from_pdb(temp_pdb_path)

                    # IMPROVEMENT: parse and pass hotspots (original
                    # hard-coded hotspot=None).
                    hotspot_mask_i = target_hotspot_mask[i] if target_hotspot_mask is not None else None
                    hotspot_list = self._parse_hotspots(hotspot_mask_i, chain_index, res_mask)

                    af_model.prep_inputs(
                        pdb_filename=temp_pdb_path,
                        target_chain=target_chain,
                        binder_chain=binder_chain,
                        mode="wildtype",
                        rm_target=False,
                        rm_target_seq=False,
                        rm_target_sc=False,
                        hotspot=hotspot_list,
                        use_binder_template=True,
                        rm_template_ic=True,
                    )

                    # BUG FIX (callback accumulation): callbacks persist
                    # across prep_inputs -- register them only on the
                    # first sample.  Built-in weights are reset by
                    # restart(), so re-apply every iteration.
                    if not callbacks_registered:
                        self._register_loss_callbacks(af_model, loss_weights)
                        callbacks_registered = True
                    self._set_builtin_weights(af_model, loss_weights)

                    # ---- Optimisation stages (unchanged from original) ----
                    if enable_soft:
                        logger.info(f"Sample {i + 1}/{bs} - Stage 2: Softmax optimisation")
                        af_model.design_soft(
                            n_temp_iters,
                            e_temp=1e-2,
                            models=[0],
                            num_models=1,
                            sample_models=False,
                            ramp_recycles=False,
                        )

                        logger.info(f"Sample {i + 1}/{bs} - Stage 3: One-hot optimisation")
                        af_model.design_hard(
                            n_hard_iters,
                            temp=1e-2,
                            models=[0],
                            num_models=1,
                            sample_models=False,
                            dropout=False,
                            ramp_recycles=False,
                        )

                    if enable_greedy:
                        logger.info(f"Sample {i + 1}/{bs} - Stage 4: PSSM semigreedy optimisation")
                        greedy_tries = math.ceil(n * (greedy_percentage / 100))
                        af_model.design_pssm_semigreedy(
                            soft_iters=0,
                            hard_iters=n_greedy_iters,
                            tries=greedy_tries,
                            models=[0],
                            num_models=1,
                            sample_models=False,
                            ramp_models=False,
                            save_best=True,
                        )

                    # ---- Load refined structure ----
                    if enable_soft or enable_greedy:
                        save_pdb_filename = temp_pdb_path.replace(".pdb", "_refolded.pdb")
                        af_model.save_pdb(save_pdb_filename)
                    else:
                        save_pdb_filename = temp_pdb_path

                    stage_4_sample = load_pdb(save_pdb_filename)
                    refined_coors = torch.as_tensor(stage_4_sample.atom_positions).to(coors.device).float()
                    refined_residue_type = torch.as_tensor(stage_4_sample.aatype).to(coors.device).long()

                    # BUG FIX: validate that ColabDesign returned the
                    # expected number of residues before writing into
                    # the padded tensor.
                    if refined_coors.shape[0] != n:
                        raise ValueError(f"Refined PDB has {refined_coors.shape[0]} residues but mask expects {n}")

                    # Pad refined data back into full-length tensors
                    refined_coors_full = coors.clone()
                    refined_residue_type_full = residue_type.clone()
                    refined_coors_full[res_mask] = refined_coors
                    refined_residue_type_full[res_mask] = refined_residue_type

                    refined_sample_prots.append(
                        {
                            "coors": refined_coors_full.unsqueeze(0),
                            "residue_type": refined_residue_type_full.unsqueeze(0),
                            # Keep original chain_index; ColabDesign uses
                            # only chain A/B internally.
                            "chain_index": chain_index.clone().unsqueeze(0),
                        }
                    )
                    elapsed = time.time() - t0
                    logger.info(f"Refined sample {i + 1}/{bs} in {elapsed:.1f}s")

                except Exception as exc:
                    elapsed = time.time() - t0
                    logger.warning(
                        f"Refinement failed for sample {i + 1}/{bs} after {elapsed:.1f}s, keeping original: {exc}"
                    )
                    refined_sample_prots.append(
                        {
                            "coors": coors.clone().unsqueeze(0),
                            "residue_type": residue_type.clone().unsqueeze(0),
                            "chain_index": chain_index.clone().unsqueeze(0),
                        }
                    )

            refined = concat_dict_tensors(refined_sample_prots, dim=0)

            # IMPROVEMENT: preserve all keys from the input that
            # refinement does not overwrite (e.g. mask, sample_type,
            # metadata_tag).  The original only returned the three keys
            # built inside the loop.
            result: dict[str, Any] = {}
            for key in sample_prots:
                result[key] = refined[key] if key in refined else sample_prots[key]
            return result

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
