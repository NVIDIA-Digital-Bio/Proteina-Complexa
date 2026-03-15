"""Monte Carlo Tree Search (MCTS) algorithm for protein generation.

This module provides MCTS for exploring the protein generation space with
exploration-exploitation balance.
"""

import torch
from loguru import logger

from proteinfoundation.rewards.base_reward import TOTAL_REWARD_KEY
from proteinfoundation.rewards.reward_utils import compute_reward_from_samples
from proteinfoundation.search.base_search import BaseSearch, SearchContext
from proteinfoundation.search.search_utils import chunked_partial_simulation, filter_lookahead_samples
from proteinfoundation.utils.mcts_utils import (
    MCTSState,
    backpropagate_reward,
    get_tree_statistics,
    print_tree_structure,
)
from proteinfoundation.utils.sample_utils import prepend_target_to_samples, sample_formatting
from proteinfoundation.utils.tensor_utils import concat_dict_tensors


class MCTSSearch(BaseSearch):
    """Monte Carlo Tree Search algorithm."""

    def search(self, batch: dict) -> dict:
        """
        Monte Carlo Tree Search prediction step with simplified batch processing.

        Algorithm:
        1. cur_states = root_states
        2. for each step checkpoint:
           3. for each simulation:
              4. cur_sim_states = cur_states
              5. for cur_search_step_checkpoint to end_step_checkpoint:
                 6. is_explore_new_state = (torch.rand() < eps)
                 7. if is_explore_new_state or all(cur_sim_states[sample_idx] has no child):
                    8. cur_sim_states = partial_simulation(cur_sim_states)
                 9. else:
                   10. for sample_idx in batch: cur_sim_states[sample_idx] takes the best ucb child
             11. Compute reward for cur_sim_states, save as exploration samples
             12. Backpropagate rewards
           13. cur_states = cur_states's child with the highest expected reward

        Args:
            batch: Input batch containing protein data. Must contain 'mask' tensor.

        Returns:
            Dictionary with sample_prots containing all samples
        """
        # ── shared simulation context ────────────────────────────────────
        search_ctx = self.build_search_context(batch)

        # ── algorithm-specific config ────────────────────────────────────
        mcts_config = self.inf_cfg.search.mcts
        n_simulations = mcts_config.n_simulations
        exploration_prob = mcts_config.get("exploration_prob", 0.1)
        exploration_constant = mcts_config.get("exploration_constant", 1.414)
        keep_lookahead_samples = mcts_config.get("keep_lookahead_samples", False)

        step_checkpoints = self.inf_cfg.search.get("step_checkpoints", None)
        if step_checkpoints is None:
            raise ValueError(
                "MCTS search requires 'step_checkpoints' in search config (e.g. step_checkpoints: [0, 50, 100])"
            )
        if step_checkpoints[-1] != search_ctx.nsteps:
            raise ValueError(
                f"Last step_checkpoint ({step_checkpoints[-1]}) must equal nsteps ({search_ctx.nsteps}). "
                "Otherwise MCTS will produce partially denoised samples."
            )

        init_mask = batch["mask"]
        nsamples, n = init_mask.shape

        logger.info(
            f"Starting MCTS with {nsamples} samples, {n_simulations} simulations per checkpoint, exploration_prob={exploration_prob}"
        )

        # Initialize root states and lineage tags for each sample
        cur_states = []
        metadata_tags = []
        for sample_idx in range(nsamples):
            xt_root = self.proteina.fm.sample_noise(
                n,
                shape=(1,),
                device=search_ctx.device,
                mask=init_mask[sample_idx : sample_idx + 1],
            )

            root_state = MCTSState(
                current_step=0,
                x_t=xt_root,
                x_1_pred=None,
                sample_idx=sample_idx,
                branch_idx=0,
            )
            cur_states.append(root_state)
            metadata_tags.append(f"mcts_orig{sample_idx}")

        logger.debug(f"[MCTS] Initialized {len(cur_states)} root states")

        all_sample_prots = []

        # Process each step checkpoint
        for checkpoint_idx in range(len(step_checkpoints) - 1):
            start_step = step_checkpoints[checkpoint_idx]
            end_step = step_checkpoints[checkpoint_idx + 1]
            logger.info(
                f"Processing checkpoint {checkpoint_idx + 1}/{len(step_checkpoints) - 1}: {start_step} -> {end_step}"
            )

            # Run MCTS simulations for this checkpoint
            for sim_idx in range(n_simulations):
                logger.debug(f"Simulation {sim_idx + 1}/{n_simulations}")

                # cur_sim_states = cur_states (copy for this simulation)
                cur_sim_states = cur_states.copy()

                # Process from current checkpoint to the last checkpoint
                for cur_checkpoint_idx in range(checkpoint_idx, len(step_checkpoints) - 1):
                    cur_search_step = step_checkpoints[cur_checkpoint_idx]
                    next_search_step = step_checkpoints[cur_checkpoint_idx + 1]

                    # Check if this is the last checkpoint (terminal state)
                    is_last_checkpoint = cur_checkpoint_idx == len(step_checkpoints) - 2

                    # is_explore_new_state = (torch.rand() < eps)
                    is_explore_new_state = torch.rand(1).item() < exploration_prob

                    # Check if all cur_sim_states have no children
                    all_have_no_children = all(len(state.children) == 0 for state in cur_sim_states)

                    should_explore = is_explore_new_state or all_have_no_children or is_last_checkpoint

                    if should_explore:
                        # Explore: cur_sim_states = partial_simulation(cur_sim_states)
                        # This creates new child states by running partial simulation
                        logger.debug(
                            f"[MCTS] Simulation {sim_idx + 1}/{n_simulations}, step {cur_search_step} -> {next_search_step}: Exploring: is_explore={is_explore_new_state}, all_no_children={all_have_no_children}"
                        )
                        cur_sim_states = self._mcts_partial_simulation_batch(
                            cur_sim_states,
                            batch,
                            init_mask,
                            cur_search_step,
                            next_search_step,
                            search_ctx,
                        )
                    else:
                        # Exploit: for sample_idx in batch: cur_sim_states[sample_idx] takes the best ucb child
                        # This selects the best child based on UCB scores
                        logger.debug(
                            f"[MCTS] Simulation {sim_idx + 1}/{n_simulations}, step {cur_search_step} -> {next_search_step}: Exploiting: selecting best UCB children"
                        )
                        for sample_idx, state in enumerate(cur_sim_states):
                            if state.children:
                                best_child = max(
                                    state.children,
                                    key=lambda c: c.ucb_score(exploration_constant),
                                )
                                cur_sim_states[sample_idx] = best_child

                # Compute reward for cur_sim_states (all samples are at final state)
                simulation_samples = self._mcts_compute_rewards_batch(
                    cur_sim_states,
                    batch,
                    init_mask,
                    search_ctx,
                )

                # Build metadata tags for this simulation's lookahead samples
                # Uses the current lineage tag + this simulation's checkpoint/sim/branch
                sim_tags = [
                    f"{tag}-s{start_step}to{end_step}sim{sim_idx}br{state.branch_idx}"
                    for tag, state in zip(metadata_tags, cur_sim_states, strict=False)
                ]

                if keep_lookahead_samples:
                    reward_threshold = self.inf_cfg.search.get("reward_threshold", None)
                    la = filter_lookahead_samples(
                        simulation_samples,
                        simulation_samples["rewards"],
                        sim_tags,
                        reward_threshold,
                    )
                    if la is not None:
                        all_sample_prots.append(la)

                # Backpropagate rewards (simulation_samples is always not None)
                rewards = simulation_samples["rewards"][TOTAL_REWARD_KEY]
                if sim_idx == 0:  # Log reward info for first simulation
                    logger.debug(
                        f"[MCTS] Rewards: mean={rewards.mean().item():.4f}, std={rewards.std().item() if rewards.numel() > 1 else 0.0:.4f}, min={rewards.min().item():.4f}, max={rewards.max().item():.4f}"
                    )
                for i, (state, reward) in enumerate(zip(cur_sim_states, rewards, strict=False)):
                    backpropagate_reward(state, reward.item())

            # Log tree statistics and print tree structure
            for sample_idx, state in enumerate(cur_states):
                stats = get_tree_statistics(state)
                logger.debug(
                    f"[MCTS] Sample {sample_idx} tree stats: nodes={stats['total_nodes']}, depth={stats['max_depth']}, visits={stats['total_visits']}, avg_reward={stats['root_average_reward']:.4f}"
                )

                # Print tree structure for visualization with UCB scores and estimated rewards
                logger.info(
                    f"[MCTS] Tree structure for sample {sample_idx} (exploration_constant={exploration_constant}):"
                )
                print_tree_structure(state, max_depth=3, exploration_constant=exploration_constant)

            # cur_states = cur_states's child with the highest expected reward
            # Append the chosen branch to each sample's lineage tag
            best_branch_indices = self._move_roots_to_best_children(cur_states)
            for i, br_idx in enumerate(best_branch_indices):
                if br_idx is not None:
                    metadata_tags[i] = f"{metadata_tags[i]}-s{start_step}to{end_step}br{br_idx}"
            logger.debug(f"[MCTS] Moved roots to best children for checkpoint {checkpoint_idx + 1}")

        # Collect final samples (rewards computed in proteina.predict_step)
        final_sample_prots, _ = self._collect_final_mcts_samples(cur_states, batch, init_mask)
        final_sample_prots["metadata_tag"] = metadata_tags

        lookahead = None
        if all_sample_prots:
            lookahead = concat_dict_tensors(all_sample_prots, dim=0)

        logger.info(
            f"[MCTS] Completed MCTS. Final samples: {final_sample_prots['coors'].shape[0]}, "
            f"Look-ahead: {lookahead['coors'].shape[0] if lookahead else 0}"
        )

        return {"lookahead": lookahead, "final": final_sample_prots}

    def _mcts_partial_simulation_batch(
        self,
        states: list[MCTSState],
        batch: dict,
        init_mask: torch.Tensor,
        start_step: int,
        end_step: int,
        search_ctx: SearchContext,
    ) -> list[MCTSState]:
        """Perform batch partial simulation for all states and create new child states.

        Args:
            states: List of states to simulate from (one per sample).
            batch: Original conditioning batch.
            init_mask: Residue mask [nsamples, n_residues].
            start_step: Starting simulation step.
            end_step: Ending simulation step.
            search_ctx: Shared search context with simulation parameters.

        Returns:
            List of new child states created from the simulation.
        """
        # Concatenate all x_t and x_1_pred for batch processing
        batch_xt = {}
        batch_x_1_pred = {}

        for key in states[0].x_t.keys():
            batch_xt[key] = torch.cat([state.x_t[key] for state in states], dim=0)

        if states[0].x_1_pred is not None:
            for key in states[0].x_1_pred.keys():
                batch_x_1_pred[key] = torch.cat([state.x_1_pred[key] for state in states], dim=0)
        else:
            batch_x_1_pred = None

        # Use the shared chunking helper to cap simulation sub-batch size and
        # avoid OOM when nsamples exceeds search.max_batch_size.
        logger.debug(f"[MCTS] Running partial simulation from step {start_step} to {end_step} for {len(states)} states")
        new_xt, new_x_1_pred = chunked_partial_simulation(
            proteina=self.proteina,
            batch=batch,
            xt=batch_xt,
            x_1_pred=batch_x_1_pred,
            mask=init_mask,
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

        # De-batch the results into new states
        new_states = []
        for i, state in enumerate(states):
            # Extract this sample's data from the batch
            sample_xt = {key: value[i : i + 1] for key, value in new_xt.items()}
            if new_x_1_pred is not None:
                sample_x_1_pred = {key: value[i : i + 1] for key, value in new_x_1_pred.items()}
            else:
                sample_x_1_pred = None

            # Create new child state
            new_state = MCTSState(
                current_step=end_step,
                x_t=sample_xt,
                x_1_pred=sample_x_1_pred,
                parent=state,
                sample_idx=state.sample_idx,
                branch_idx=len(state.children),  # Next available branch index
            )

            # Add to parent's children
            state.children.append(new_state)
            new_states.append(new_state)

        return new_states

    def _mcts_compute_rewards_batch(
        self,
        states: list[MCTSState],
        batch: dict,
        init_mask: torch.Tensor,
        search_ctx: SearchContext,
    ) -> dict:
        """Compute rewards for all states (all samples are already at final state).

        Args:
            states: List of states to compute rewards for (all at final state).
            batch: Original conditioning batch.
            init_mask: Residue mask [nsamples, n_residues].
            search_ctx: Shared search context (used for hotspot mask access).

        Returns:
            Dictionary containing the samples and rewards.
        """
        # Collect all x_t for batch processing
        batch_xt = {}
        for key in states[0].x_t.keys():
            batch_xt[key] = torch.cat([state.x_t[key] for state in states], dim=0)

        # NOTE: states are at step_checkpoints[-1] after the inner simulation
        # loop.  Unlike beam search (which uses generate_samples_to_completion),
        # this decodes directly without rolling out to nsteps.  This is correct
        # only when step_checkpoints[-1] == nsteps; otherwise the samples are
        # partially denoised and rewards will be computed on incomplete structures.
        assert len(states) == init_mask.shape[0], (
            f"[MCTS] states ({len(states)}) != nsamples ({init_mask.shape[0]}); "
            f"mask expansion needed if MCTS supports branching"
        )
        sample_prots = sample_formatting(
            x=batch_xt,
            extra_info={"mask": init_mask},
            ret_mode="coors37_n_aatype",
            data_modes=list(self.proteina.cfg_exp.product_flowmatcher),
            autoencoder=getattr(self.proteina, "autoencoder", None),
        )

        # MCTS produces 1 sample per batch entry — no expansion needed.
        if batch.get("prepend_target", False) and not hasattr(self.proteina, "ligand"):
            sample_prots = prepend_target_to_samples(sample_prots, batch)

        logger.debug(f"[MCTS] Computing rewards for {len(states)} states")
        rewards_dict = compute_reward_from_samples(
            self.proteina.reward_model,
            sample_prots,
            search_ctx.current_batch.get("target_hotspot_mask"),
            getattr(self.proteina, "ligand", None),
        )
        sample_prots["rewards"] = rewards_dict
        rewards_total = rewards_dict[TOTAL_REWARD_KEY]
        logger.debug(
            f"[MCTS] Computed rewards: mean={rewards_total.mean().item():.4f}, std={rewards_total.std().item() if rewards_total.numel() > 1 else 0.0:.4f}"
        )

        return sample_prots

    def _move_roots_to_best_children(self, root_states: list[MCTSState]) -> list:
        """Move each root state to its best child (highest average reward).

        Args:
            root_states: List of root states to move (modified in place).

        Returns:
            List of selected branch indices (one per sample), or None if no
            children existed for that sample.
        """
        selected_branches = []
        for i, root_state in enumerate(root_states):
            if root_state.children:
                best_child = max(root_state.children, key=lambda c: c.average_reward)

                root_state.current_step = best_child.current_step
                root_state.x_t = best_child.x_t
                root_state.x_1_pred = best_child.x_1_pred
                root_state.visits = best_child.visits
                root_state.cumulative_reward = best_child.cumulative_reward
                root_state.branch_idx = best_child.branch_idx

                root_state.children = best_child.children
                root_state.is_fully_expanded = best_child.is_fully_expanded

                for child in root_state.children:
                    child.parent = root_state

                selected_branches.append(best_child.branch_idx)
                logger.debug(
                    f"Sample {i}: Moved root to best child (branch {best_child.branch_idx}) with avg_reward={best_child.average_reward:.4f}"
                )
            else:
                selected_branches.append(None)
        return selected_branches

    def _collect_final_mcts_samples(self, root_states: list[MCTSState], batch: dict, init_mask: torch.Tensor) -> tuple:
        """
        Collect final samples from MCTS trees (root states are already at best paths).

        Args:
            root_states: List of root states for each sample (already at best paths)
            batch: Original batch
            init_mask: Initial mask

        Returns:
            Tuple of (sample_prots, final_xt) - both proteins and latents
        """
        # Collect all x_t for batch processing
        batch_xt = {}
        for key in root_states[0].x_t.keys():
            batch_xt[key] = torch.cat([state.x_t[key] for state in root_states], dim=0)

        # NOTE: root_states are at step_checkpoints[-1].  If that is less than
        # nsteps the returned samples will be partially denoised.  Consider
        # using generate_samples_to_completion for a full rollout.
        logger.debug(f"[MCTS] Collecting final samples from {len(root_states)} root states")

        # Format to protein
        sample_prots = sample_formatting(
            x=batch_xt,
            extra_info={"mask": init_mask},
            ret_mode="coors37_n_aatype",
            data_modes=list(self.proteina.cfg_exp.product_flowmatcher),
            autoencoder=getattr(self.proteina, "autoencoder", None),
        )

        # MCTS produces 1 sample per batch entry — no expansion needed.
        if batch.get("prepend_target", False) and not hasattr(self.proteina, "ligand"):
            sample_prots = prepend_target_to_samples(sample_prots, batch)

        return sample_prots, batch_xt
