"""FK-steering algorithm for protein generation.

Branching strategy is identical to beam search (see ``beam_search.py`` module
docstring for the full explanation):

    * Was: ``beam_width × n_branch`` sequential ``partial_simulation`` calls.
    * Now: 1 batched call via ``chunked_partial_simulation``, chunked to
      ``max_batch_size`` to stay within GPU memory.
    * Estimated speedup per step: ``beam_width × n_branch``.

The only difference from beam search is the selection mechanism: FK-steering
uses reward-weighted multinomial sampling (soft) instead of top-k (hard).
The step-to-step loop remains sequential for the same reason: each step
depends on the previous step's selection.
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


class FKSteering(BaseSearch):
    """FK-steering algorithm with importance sampling."""

    def search(self, batch: dict) -> dict:
        """FK-steering prediction step with importance sampling.

        Same branch-evaluate-select loop as beam search, but selection uses
        reward-weighted multinomial sampling (soft) instead of top-k (hard).

        Returns:
            Dict with 'lookahead' (optional) and 'final' sample dicts.
        """
        # ── shared simulation context ────────────────────────────────────
        search_ctx = self.build_search_context(batch)

        # ── algorithm-specific config ────────────────────────────────────
        fk_cfg = self.inf_cfg.search.fk_steering
        n_branch = fk_cfg.n_branch
        beam_width = fk_cfg.beam_width
        temperature = fk_cfg.temperature
        if temperature <= 0:
            raise ValueError(
                f"FK-steering temperature must be > 0 (got {temperature}). "
                "Zero causes division-by-zero in importance-weight softmax."
            )
        keep_lookahead_samples = fk_cfg.get("keep_lookahead_samples", False)
        reward_threshold = self.inf_cfg.search.get("reward_threshold", None)

        save_intermediates = fk_cfg.get("save_intermediate_states", False)
        trajectory_dir = fk_cfg.get("trajectory_dir", None)
        if save_intermediates and trajectory_dir is not None:
            trajectory_dir = os.path.abspath(trajectory_dir)
            os.makedirs(trajectory_dir, exist_ok=True)
        else:
            save_intermediates = False

        step_checkpoints = self.inf_cfg.search.get("step_checkpoints", None)
        if step_checkpoints is None:
            raise ValueError(
                "FK-steering requires 'step_checkpoints' in search config (e.g. step_checkpoints: [0, 50, 100])"
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
        logger.info(
            f"[FKSteering] Starting | nsamples={nsamples}, "
            f"beam_width={beam_width}, n_branch={n_branch}, "
            f"temperature={temperature}, checkpoints={step_checkpoints}"
        )
        mask = init_mask.repeat_interleave(beam_width, dim=0)
        xt = self.proteina.fm.sample_noise(
            n,
            shape=(nsamples * beam_width,),
            device=search_ctx.device,
            mask=mask,
        )
        x_1_pred = None
        metadata_tags = make_initial_search_tags("fk", nsamples, beam_width)

        # ── main search loop ────────────────────────────────────────────
        all_sample_prots = []
        for i in range(n_steps_total):
            start_step = step_checkpoints[i]
            end_step = step_checkpoints[i + 1]
            logger.info(f"[FKSteering] Step {i + 1}/{n_steps_total}: denoising {start_step} -> {end_step}")

            # ── branch: batch all replicas x branches in one call ────────
            # See beam_search.py for the full branching explanation.  Same
            # strategy: tile each replica's xt n_branch times, concat across
            # replicas, and run chunked_partial_simulation.
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
                f"[FKSteering] big_xt dim-0 ({actual_total}) != expected "
                f"nsamples({nsamples}) * bw({beam_width}) * nb({n_branch}) "
                f"= {expected_total}"
            )

            xt, x_1_pred = chunked_partial_simulation(
                self.proteina,
                batch=tile_batch(batch, branching_factor),
                xt=big_xt,
                x_1_pred=big_pred,
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
                f"[FKSteering] xt after branching ({n_candidates}) != expected ({expected_total})"
            )
            assert len(expanded_tags) == n_candidates, (
                f"[FKSteering] tags ({len(expanded_tags)}) != xt ({n_candidates}) after branching"
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

            # ── look-ahead + evaluate ───────────────────────────────────
            # Same note as beam search: the loop range is ``range(n_steps_total)``
            # so every iteration needs evaluation and selection.
            lookahead_prots = generate_samples_to_completion(
                search_ctx,
                self.proteina,
                xt,
                end_step,
                init_mask,
                x_1_pred=x_1_pred,  # preserve self-conditioning from prior step
            )
            rewards_dict = compute_reward_from_samples(
                self.proteina.reward_model,
                lookahead_prots,
                search_ctx.current_batch.get("target_hotspot_mask"),
                getattr(self.proteina, "ligand", None),
            )
            total_rewards = rewards_dict[TOTAL_REWARD_KEY]

            if keep_lookahead_samples:
                la = filter_lookahead_samples(
                    lookahead_prots,
                    rewards_dict,
                    expanded_tags,
                    reward_threshold,
                )
                if la is not None:
                    all_sample_prots.append(la)

            # ── importance sampling: P(sample) ∝ exp(reward / temperature) ──
            assert total_rewards.shape[0] == n_candidates, (
                f"[FKSteering] rewards ({total_rewards.shape[0]}) != candidates ({n_candidates})"
            )
            # FIX: same replica-major layout as beam search (see beam_search.py).
            rewards_reshaped = total_rewards.view(beam_width, n_branch, nsamples)
            rewards_for_selection = rewards_reshaped.permute(2, 0, 1).reshape(nsamples, beam_width * n_branch)
            logits = rewards_for_selection / temperature
            sampling_probs = torch.softmax(logits, dim=1)

            bad_rows = torch.isnan(sampling_probs).any(dim=1) | (sampling_probs.sum(dim=1) < 1e-8)
            if bad_rows.any():
                n_bad = bad_rows.sum().item()
                logger.warning(
                    f"[FKSteering] Step {i + 1}: {n_bad}/{nsamples} samples have "
                    f"NaN/zero probabilities after softmax — falling back to uniform"
                )
                uniform = torch.ones_like(sampling_probs[0]) / sampling_probs.shape[1]
                sampling_probs[bad_rows] = uniform

            sampled_indices = torch.multinomial(sampling_probs, num_samples=beam_width, replacement=True)
            replica_indices = sampled_indices // n_branch
            branch_indices = sampled_indices % n_branch

            sample_indices = torch.arange(nsamples, device=search_ctx.device).unsqueeze(1).expand(-1, beam_width)
            global_indices = replica_indices * n_branch * nsamples + branch_indices * nsamples + sample_indices
            select_indices = global_indices.flatten()

            xt = {k: v[select_indices] for k, v in xt.items()}
            metadata_tags = select_tags(expanded_tags, select_indices)
            if x_1_pred is not None:
                x_1_pred = {k: v[select_indices] for k, v in x_1_pred.items()}

            n_after = next(iter(xt.values())).shape[0]
            assert n_after == nsamples * beam_width, (
                f"[FKSteering] xt after selection ({n_after}) != "
                f"nsamples({nsamples}) * beam_width({beam_width}) "
                f"= {nsamples * beam_width}"
            )
            assert len(metadata_tags) == n_after, (
                f"[FKSteering] tags ({len(metadata_tags)}) != xt ({n_after}) after selection"
            )
            logger.debug(f"[FKSteering] Step {i + 1}: importance sampling done | xt={n_after} samples")

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

        return {"lookahead": lookahead, "final": final_sample_prots}
