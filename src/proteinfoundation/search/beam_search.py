"""Beam search algorithm for protein generation.

Branching strategy (was: ``beam_width × n_branch`` sequential
``partial_simulation`` calls → now: 1 batched call with ``max_batch_size``
chunking via ``chunked_partial_simulation``):

    1. Each beam replica's ``xt`` is tiled ``n_branch`` times via
       ``tile_tensor_dict`` → independent branch copies.
    2. All tiled replicas are concatenated into a single batch of
       ``nsamples × beam_width × n_branch`` candidates.
    3. ``chunked_partial_simulation`` splits into sub-batches of at most
       ``max_batch_size`` if the total exceeds GPU capacity.
    4. Results are concatenated and the top-k selection proceeds as before.

    Estimated speedup per search step: ``beam_width × n_branch`` fewer
    ``partial_simulation`` calls (e.g. 4×4 = 16× fewer calls).
    Actual speedup depends on how much the GPU can parallelise – the total
    work is the same, but wall-clock time drops significantly because GPU
    utilisation goes from ~1/(beam_width × n_branch) to near-100%.

    The step-to-step loop remains sequential because each step's candidates
    depend on the previous step's reward-based selection.

Look-ahead calls (``generate_samples_to_completion``) similarly respect
``max_batch_size`` via ``chunked_partial_simulation`` internally.

Output size (with ``keep_lookahead_samples=True``)::

    Total PDBs = N * W * (B * S + 1)

    N = nsamples            (batch size from dataloader)
    W = beam_width          (beams kept per sample after top-k)
    B = n_branch            (branches explored per beam per step)
    S = len(step_checkpoints) - 1   (number of search steps)

    Per step:  N * W * B candidates are generated and scored via look-ahead.
               All N * W * B are saved as lookahead PDBs.
               Top-k keeps the best N * W to continue to the next step.
    Final:     The surviving N * W beams after the last step.

    Example  (N=4, W=4, B=4, step_checkpoints=[0, 100, 200, 300, 400] → S=4):

        Lookahead PDBs   = N * W * B * S  = 4 * 4 * 4 * 4 = 256
        Final PDBs       = N * W          = 4 * 4          =  16
        Total             = 256 + 16                        = 272

    With ``keep_lookahead_samples=False`` only the N * W final PDBs are saved.
    With ``reward_threshold`` set, lookahead PDBs below the threshold are dropped.
"""

import os

import torch
from loguru import logger

from proteinfoundation.rewards.base_reward import TOTAL_REWARD_KEY
from proteinfoundation.rewards.reward_utils import compute_reward_from_samples
from proteinfoundation.search.base_search import BaseSearch
from proteinfoundation.search.search_utils import (
    chunked_partial_simulation,
    decode_and_save_intermediates,
    expand_tags_for_branches,
    filter_lookahead_samples,
    generate_samples_to_completion,
    make_initial_search_tags,
    select_tags,
    tile_batch,
    tile_tensor_dict,
)
from proteinfoundation.utils.sample_utils import prepend_target_to_samples, sample_formatting
from proteinfoundation.utils.tensor_utils import concat_dict_tensors


class BeamSearch(BaseSearch):
    """Beam search algorithm with top-k selection."""

    def search(self, batch: dict) -> dict:
        """Beam search prediction step.

        At each step checkpoint the current samples are branched (n_branch copies
        per beam), rolled out to completion for reward evaluation, and the top
        beam_width candidates are kept.  Metadata tags travel with every sample
        so the full lineage is recorded.

        Returns:
            Dict with 'lookahead' (optional) and 'final' sample dicts, each
            containing a 'metadata_tag' list of per-sample provenance strings.
        """
        # ── shared simulation context ────────────────────────────────────
        search_ctx = self.build_search_context(batch)

        # ── algorithm-specific config ────────────────────────────────────
        n_branch = self.inf_cfg.search.beam_search.n_branch
        beam_width = self.inf_cfg.search.beam_search.beam_width
        beam_cfg = self.inf_cfg.search.beam_search
        keep_lookahead_samples = beam_cfg.get("keep_lookahead_samples", False)
        reward_threshold = self.inf_cfg.search.get("reward_threshold", None)

        save_intermediates = beam_cfg.get("save_intermediate_states", False)
        trajectory_dir = beam_cfg.get("trajectory_dir", None)
        if save_intermediates and trajectory_dir is not None:
            trajectory_dir = os.path.abspath(trajectory_dir)
            os.makedirs(trajectory_dir, exist_ok=True)
        else:
            save_intermediates = False

        step_checkpoints = self.inf_cfg.search.get("step_checkpoints", None)
        if step_checkpoints is None:
            raise ValueError(
                "Beam search requires 'step_checkpoints' in search config (e.g. step_checkpoints: [0, 50, 100])"
            )
        if beam_width < 1:
            raise ValueError(f"beam_width must be >= 1, got {beam_width}")
        if n_branch < 1:
            raise ValueError(f"n_branch must be >= 1, got {n_branch}")
        if len(step_checkpoints) < 2:
            raise ValueError(
                f"step_checkpoints must have at least 2 entries, got {len(step_checkpoints)}: {step_checkpoints}"
            )
        if step_checkpoints[-1] != search_ctx.nsteps:
            raise ValueError(
                f"step_checkpoints[-1] ({step_checkpoints[-1]}) must equal "
                f"nsteps ({search_ctx.nsteps}); otherwise final samples are partially "
                f"denoised and rewards are computed on incomplete structures."
            )
        n_steps_total = len(step_checkpoints) - 1

        # ── initialise noise + tags ─────────────────────────────────────
        init_mask = batch["mask"]
        nsamples, n = init_mask.shape
        mask = init_mask.repeat_interleave(beam_width, dim=0)
        xt = self.proteina.fm.sample_noise(
            n,
            shape=(nsamples * beam_width,),
            device=search_ctx.device,
            mask=mask,
        )
        x_1_pred = None
        metadata_tags = make_initial_search_tags("beam", nsamples, beam_width)

        logger.info(
            f"[BeamSearch] Starting | nsamples={nsamples}, seq_len={n}, "
            f"beam_width={beam_width}, n_branch={n_branch}, "
            f"checkpoints={step_checkpoints}, "
            f"keep_lookahead={keep_lookahead_samples}, "
            f"save_intermediates={save_intermediates}"
        )

        # ── main search loop ────────────────────────────────────────────
        all_sample_prots = []
        for i in range(n_steps_total):
            start_step = step_checkpoints[i]
            end_step = step_checkpoints[i + 1]
            logger.info(f"\n[BeamSearch] Step {i + 1}/{n_steps_total}: denoising {start_step} -> {end_step}")

            # ── branch: batch all replicas × branches in one call ────────
            # Previously this was a nested ``for replica × for branch`` loop
            # making ``beam_width × n_branch`` sequential partial_simulation
            # calls.  Now all branches are assembled into a single batch and
            # processed via ``chunked_partial_simulation`` which respects
            # ``max_batch_size`` to avoid OOM.
            #
            # Build big_xt: one batch holding every (replica, branch, sample)
            # combination.  Layout is REPLICA-MAJOR:
            #
            #   for replica in range(beam_width):       ← outer
            #       for branch in range(n_branch):      ← middle (from .repeat)
            #           for sample in range(nsamples):   ← inner (contiguous)
            #
            # flat_index = replica * n_branch * nsamples
            #            + branch  * nsamples
            #            + sample
            #
            # Example: nsamples=3, beam_width=2, n_branch=4
            #
            #   xt (input, nsamples * beam_width = 6, GROUPED by sample):
            #     [s0r0, s0r1, s1r0, s1r1, s2r0, s2r1]
            #
            #   replica 0: v[0::2] → [s0r0, s1r0, s2r0]        (3 rows)
            #     tile x4  → [s0r0, s1r0, s2r0,                 branch 0
            #                  s0r0, s1r0, s2r0,                 branch 1
            #                  s0r0, s1r0, s2r0,                 branch 2
            #                  s0r0, s1r0, s2r0]                 branch 3
            #
            #   replica 1: v[1::2] → [s0r1, s1r1, s2r1]        (3 rows)
            #     tile x4  → [s0r1, s1r1, s2r1,                 branch 0
            #                  s0r1, s1r1, s2r1,                 branch 1
            #                  s0r1, s1r1, s2r1,                 branch 2
            #                  s0r1, s1r1, s2r1]                 branch 3
            #
            #   big_xt = concat → 24 rows (2 * 4 * 3)
            #     indices  0..2  → replica 0, branch 0, samples 0-2
            #     indices  3..5  → replica 0, branch 1, samples 0-2
            #     indices  6..8  → replica 0, branch 2, samples 0-2
            #     indices  9..11 → replica 0, branch 3, samples 0-2
            #     indices 12..14 → replica 1, branch 0, samples 0-2
            #     indices 15..17 → replica 1, branch 1, samples 0-2
            #     indices 18..20 → replica 1, branch 2, samples 0-2
            #     indices 21..23 → replica 1, branch 3, samples 0-2
            #
            branching_factor = beam_width * n_branch
            branch_parts = []
            pred_parts = []
            for replica_idx in range(beam_width):
                rep_xt = {k: v[replica_idx::beam_width] for k, v in xt.items()}
                branch_parts.append(tile_tensor_dict(rep_xt, n_branch))
                if x_1_pred is not None:
                    rep_pred = {k: v[replica_idx::beam_width] for k, v in x_1_pred.items()}
                    pred_parts.append(tile_tensor_dict(rep_pred, n_branch))

            big_xt = concat_dict_tensors(branch_parts, dim=0)
            big_pred = concat_dict_tensors(pred_parts, dim=0) if x_1_pred is not None else None

            expected_total = nsamples * branching_factor
            actual_total = next(iter(big_xt.values())).shape[0]
            assert actual_total == expected_total, (
                f"[BeamSearch] big_xt dim-0 ({actual_total}) != expected "
                f"nsamples({nsamples}) * bw({beam_width}) * nb({n_branch}) "
                f"= {expected_total}"
            )

            xt, x_1_pred = chunked_partial_simulation(
                self.proteina,
                batch=tile_batch(batch, branching_factor),
                xt=big_xt,
                x_1_pred=big_pred,
                # .repeat() tiles [m0,m1,m2] → [m0,m1,m2, m0,m1,m2, ...]
                # matching big_xt where samples are the innermost (contiguous) dim.
                mask=init_mask.repeat(branching_factor, 1),
                max_batch_size=search_ctx.max_batch_size,
                predict_for_sampling=search_ctx.predict_fn,
                start_step=start_step,
                end_step=end_step,
                ts=search_ctx.ts,
                gt=search_ctx.gt,
                self_cond=search_ctx.self_cond,
                simulation_step_params=search_ctx.simulation_step_params,
                device=search_ctx.device,
                guidance_w=search_ctx.guidance_w,
                ag_ratio=search_ctx.ag_ratio,
            )
            expanded_tags = expand_tags_for_branches(
                metadata_tags,
                beam_width,
                n_branch,
                start_step=start_step,
                end_step=end_step,
            )

            n_candidates = next(iter(xt.values())).shape[0]
            assert n_candidates == expected_total, (
                f"[BeamSearch] xt after branching ({n_candidates}) != expected ({expected_total})"
            )
            assert len(expanded_tags) == n_candidates, (
                f"[BeamSearch] tags ({len(expanded_tags)}) != xt ({n_candidates}) after branching"
            )
            logger.debug(
                f"[BeamSearch] Step {i + 1}: branched into {n_candidates} candidates "
                f"({beam_width} beams * {n_branch} branches * {nsamples} samples)"
            )

            # ── optional: save intermediate denoising states ────────────
            if save_intermediates:
                decode_and_save_intermediates(
                    proteina=self.proteina,
                    xt=xt,
                    init_mask=init_mask,
                    nsamples=nsamples,
                    tags=expanded_tags,
                    batch=batch,
                    trajectory_dir=trajectory_dir,
                    step_idx=i,
                )

            # ── look-ahead: roll out to completion and score ────────────
            # The loop range is ``range(n_steps_total)`` so every iteration
            # (including the last) needs a look-ahead + selection to pick the
            # best beam_width candidates from the n_branch * beam_width pool.
            lookahead_prots = generate_samples_to_completion(
                search_ctx,
                self.proteina,
                xt,
                end_step,
                init_mask,
                x_1_pred=x_1_pred,  # Before we were skipping this so it would be set to None for the first step
            )
            rewards_dict = compute_reward_from_samples(
                self.proteina.reward_model,
                lookahead_prots,
                search_ctx.current_batch.get("target_hotspot_mask"),
                getattr(self.proteina, "ligand", None),
            )
            total_rewards = rewards_dict[TOTAL_REWARD_KEY]
            logger.debug(
                f"[BeamSearch] Step {i + 1}: lookahead rewards | "
                f"mean={total_rewards.mean():.4f} "
                f"min={total_rewards.min():.4f} "
                f"max={total_rewards.max():.4f}"
            )

            # ── top-k selection ─────────────────────────────────────────
            # Reshape rewards to [nsamples, beam_width * n_branch] so we
            # pick the best beam_width candidates per original sample.
            assert total_rewards.shape[0] == n_candidates, (
                f"[BeamSearch] rewards ({total_rewards.shape[0]}) != candidates ({n_candidates})"
            )
            # FIX: flat layout from branching is replica-major:
            #   [r0_b0_s0..sN, r0_b1_s0..sN, …, rW_bB_s0..sN]
            # so dim-0 of the view must be beam_width (replica), not n_branch.
            rewards_reshaped = total_rewards.view(beam_width, n_branch, nsamples)
            # permute → [nsamples, beam_width, n_branch] → reshape flattens
            # the last two dims so each row's columns are replica-major:
            #   col_idx = replica * n_branch + branch
            rewards_for_selection = rewards_reshaped.permute(2, 0, 1).reshape(nsamples, beam_width * n_branch)

            top_k_indices = torch.topk(rewards_for_selection, k=beam_width, dim=1)[1]
            # top_k_indices are column indices into the replica-major row,
            # so we recover replica and branch via divmod on n_branch.
            replica_indices = top_k_indices // n_branch
            branch_indices = top_k_indices % n_branch

            # Map back to flat indices into big_xt / total_rewards using the
            # replica-major formula: replica * n_branch * N + branch * N + sample.
            sample_indices = torch.arange(nsamples, device=search_ctx.device).unsqueeze(1).expand(-1, beam_width)
            global_indices = replica_indices * n_branch * nsamples + branch_indices * nsamples + sample_indices
            # Flatten row-major → output is GROUPED by sample (all beam_width
            # winners for sample 0 first, then sample 1, …) which matches the
            # grouped layout that xt started with from repeat_interleave.
            select_indices = global_indices.flatten()

            selected_rewards = total_rewards[select_indices]
            logger.debug(
                f"[BeamSearch] Step {i + 1}: selected {select_indices.shape[0]} "
                f"from {n_candidates} | selected reward "
                f"mean={selected_rewards.mean():.4f} "
                f"min={selected_rewards.min():.4f} "
                f"max={selected_rewards.max():.4f}"
            )

            # ── optionally keep look-ahead samples ──────────────────────
            if keep_lookahead_samples:
                la = filter_lookahead_samples(
                    lookahead_prots,
                    rewards_dict,
                    expanded_tags,
                    reward_threshold,
                )
                if la is not None:
                    all_sample_prots.append(la)

            # ── narrow to beam_width winners ────────────────────────────
            xt = {k: v[select_indices] for k, v in xt.items()}
            metadata_tags = select_tags(expanded_tags, select_indices)
            if x_1_pred is not None:
                x_1_pred = {k: v[select_indices] for k, v in x_1_pred.items()}

            n_after = next(iter(xt.values())).shape[0]
            assert n_after == nsamples * beam_width, (
                f"[BeamSearch] xt after selection ({n_after}) != "
                f"nsamples({nsamples}) * beam_width({beam_width}) "
                f"= {nsamples * beam_width}"
            )
            assert len(metadata_tags) == n_after, (
                f"[BeamSearch] tags ({len(metadata_tags)}) != xt ({n_after}) after selection"
            )
            logger.debug(
                f"[BeamSearch] Step {i + 1}: after selection | xt={n_after} samples, tags={len(metadata_tags)}"
            )

        # ── format final output ─────────────────────────────────────────
        final_sample_prots = sample_formatting(
            x=xt,
            extra_info={"mask": mask},
            ret_mode="coors37_n_aatype",
            data_modes=list(self.proteina.cfg_exp.product_flowmatcher),
            autoencoder=getattr(self.proteina, "autoencoder", None),
        )
        if batch.get("prepend_target", False) and not hasattr(self.proteina, "ligand"):
            # Finals are grouped: [all beams for s0, all beams for s1, …].
            # Default "interleave" expands the target the same way.
            final_sample_prots = prepend_target_to_samples(final_sample_prots, batch)
        final_sample_prots["metadata_tag"] = metadata_tags

        expected = nsamples * beam_width
        actual = final_sample_prots["coors"].shape[0]
        assert actual == expected, f"Expected {expected} final samples, got {actual}"

        lookahead = None
        if keep_lookahead_samples and all_sample_prots:
            lookahead = concat_dict_tensors(all_sample_prots, dim=0)

        n_la = lookahead["coors"].shape[0] if lookahead is not None else 0
        logger.info(f"[BeamSearch] Complete | final={final_sample_prots['coors'].shape[0]}, lookahead={n_la}")
        return {"lookahead": lookahead, "final": final_sample_prots}
