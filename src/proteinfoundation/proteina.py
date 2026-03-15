import copy
import os
import random
from functools import partial
from typing import Literal

import lightning as L
import torch
from jaxtyping import Float
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from loguru import logger
from omegaconf import OmegaConf

from proteinfoundation.flow_matching.product_space_flow_matcher import ProductSpaceFlowMatcher
from proteinfoundation.nn.genie2 import Genie2Denoiser
from proteinfoundation.utils.file_utils import create_dir as _create_dir
from proteinfoundation.utils.sample_utils import add_clean_samples, sample_formatting
from proteinfoundation.utils.training_handlers import handle_batch_conditioning
from proteinfoundation.utils.validation_utils import (
    clean_validation_files,
    get_pdb_novelty_metric,
    get_structural_metrics,
)

# Architecture selection: v1 is the default.
# Set USE_V2_COMPLEXA_ARCH=True in .env to use v2 (ligand / AME models).
_USE_V2 = os.getenv("USE_V2_COMPLEXA_ARCH", "False") == "True"
if _USE_V2:
    from proteinfoundation.nn.local_latents_transformer_v2 import (
        LocalLatentsTransformer,
    )
else:
    from proteinfoundation.nn.local_latents_transformer import (
        LocalLatentsTransformer,
    )

from proteinfoundation.nn.protein_transformer import ProteinTransformerAF3
from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder
from proteinfoundation.rewards.reward_utils import compute_reward_from_samples, initialize_reward_model
from proteinfoundation.search.search_factory import instantiate_refinement, instantiate_search
from proteinfoundation.search.search_utils import (
    append_samples,
    clone_sample_dict,
    combine_lookahead_and_final,
    expand_hotspot_mask,
)
from proteinfoundation.utils.fold_utils import extract_cath_code_from_batch
from proteinfoundation.utils.pdb_utils import write_prot_to_pdb

create_dir = rank_zero_only(_create_dir)


class Proteina(L.LightningModule):
    def __init__(self, cfg_exp, store_dir=None, autoencoder_ckpt_path=None):
        super().__init__()
        if not os.environ.get("USE_V2_COMPLEXA_ARCH"):
            logger.info("USE_V2_COMPLEXA_ARCH not set, using v1 architecture (default)")
        self.save_hyperparameters()
        self.cfg_exp = cfg_exp
        self.inf_cfg = None  # Only for inference runs
        self.validation_output_lens = {}
        self.validation_output_data = []
        self.store_dir = store_dir if store_dir is not None else "./tmp"
        self.val_path_tmp = os.path.join(self.store_dir, "val_samples")
        # create_dir(self.val_path_tmp)

        if "local_latents" in cfg_exp.product_flowmatcher:
            if autoencoder_ckpt_path is not None:
                # Allow adding new keys
                logger.info(f"Manually setting autoencoder_ckpt_path to {autoencoder_ckpt_path}")
                OmegaConf.set_struct(cfg_exp, False)
                # Update the configuration with the new key-value pair
                cfg_exp.autoencoder_ckpt_path = autoencoder_ckpt_path
                # Re-enable struct mode if needed
                OmegaConf.set_struct(cfg_exp, True)

            self.autoencoder, self.latent_dim = self.load_autoencoder(cfg_exp, freeze_params=True)
            # Add right latent dimensionality in the config file, needed to instantiate the flow matcher below
            if self.autoencoder is not None:
                cfg_exp.product_flowmatcher.local_latents.dim = self.latent_dim

        self.fm = ProductSpaceFlowMatcher(cfg_exp)

        # Neural network
        if cfg_exp.nn.name == "ca_af3":
            self.nn = ProteinTransformerAF3(**cfg_exp.nn)
        # elif cfg_exp.nn.name == "ca_af3_int":
        #     self.nn = ProteinTransformerAF3Int(**cfg_exp.nn)
        elif cfg_exp.nn.name == "local_latents_transformer":
            self.nn = LocalLatentsTransformer(**cfg_exp.nn, latent_dim=self.latent_dim)
        # elif cfg_exp.nn.name == "local_latents_transformer_int":
        #     self.nn = LocalLatentsTransformerInt(
        #         **cfg_exp.nn, latent_dim=self.latent_dim
        #     )
        elif cfg_exp.nn.name == "ca_genie2":
            self.nn = Genie2Denoiser(**cfg_exp.nn)
        else:
            raise OSError(f"Wrong nn selected for CAFlow {cfg_exp.nn.name}")

        # Scaling laws stuff
        self.nflops = 0
        self.nsamples_processed = 0
        self.nparams = sum(p.numel() for p in self.nn.parameters() if p.requires_grad)

        # For autoguidance, overridden in `self.configure_inference`
        self.nn_ag = None

    def load_autoencoder(self, cfg_exp, freeze_params=True):
        """Loads autoencoder, if required."""
        if "autoencoder_ckpt_path" in cfg_exp:  # for new runs trained with refactored codebase
            ae_ckp_path = cfg_exp.autoencoder_ckpt_path
        elif (
            "autoencoder_ckpt_path" in cfg_exp.product_flowmatcher.local_latents
        ):  # for old runs trained with old codebase
            ae_ckp_path = cfg_exp.product_flowmatcher.local_latents.autoencoder_ckpt_path
        else:
            raise ValueError("No autoencoder checkpoint path provided")

        if ae_ckp_path is None:
            return None, None

        # Load and freeze parameters
        autoencoder = AutoEncoder.load_from_checkpoint(ae_ckp_path)
        if freeze_params:
            for param in autoencoder.parameters():
                param.requires_grad = False
        return autoencoder, autoencoder.latent_dim

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.cfg_exp.opt.lr)
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        """Adds additional variables to checkpoint."""
        checkpoint["nflops"] = self.nflops
        checkpoint["nsamples_processed"] = self.nsamples_processed

    def on_load_checkpoint(self, checkpoint):
        """Loads additional variables from checkpoint."""
        try:
            self.nflops = checkpoint["nflops"]
            self.nsamples_processed = checkpoint["nsamples_processed"]
        except (KeyError, AttributeError):
            logger.info("Failed to load nflops and nsamples_processed from checkpoint")
            self.nflops = 0
            self.nsamples_processed = 0

    def call_nn(
        self,
        batch: dict[str, torch.Tensor],
        n_recycle: int = 0,
    ) -> dict[str, torch.Tensor]:
        """
        Calls NN with recycling. Should this be here or in the NN? Possibly better here,
        in case we want to recycle using decoder for some approach, etc, and this is akin
        to self conditioning, also here.
        Also, if we want to recycle clean sample predictions... Then we'd need this here,
        as the nn does not know about relations between v, x1, ...
        """
        # First call
        nn_out = self.nn(batch)

        # Recycle n_recycle times detaching gradients and updating input
        for _ in range(n_recycle):
            x_1_pred = self.fm.nn_out_to_clean_sample_prediction(batch=batch, nn_out=nn_out)
            batch["x_recycle"] = {dm: x_1_pred[dm].detach() for dm in x_1_pred}
            nn_out = self.nn(batch)

        # Final prediction
        return nn_out

    def predict_for_sampling(
        self,
        batch: dict,
        mode: Literal["full", "ucond"],
        n_recycle: int = 0,
    ) -> tuple[dict[str, torch.Tensor] | float | None]:
        """
        This function predicts clean samples for multiple models:
        x_pred, the 'original' model, if mode == full
        x_pred_ucond, the unconditional model, , if mode == ucond

        TODO: Need to update to include autoguidance again

        These predictions will later be used to sample with guidance and autoguidance.

        Args:
            batch: Dict
            mode: str

        Returns:
            x_pred (tensor) for the requested mode
        """
        if mode == "full":
            nn_out = self.call_nn(batch, n_recycle=n_recycle)
        elif mode == "ucond":
            assert "cath_code" in batch, "Only support CFG when cath_code is provided"
            uncond_batch = batch.copy()
            uncond_batch.pop("cath_code")
            nn_out = self.call_nn(uncond_batch, n_recycle=n_recycle)
        else:
            raise OSError(f"Wrong {mode} passed to `predict_for_sampling`")

        return nn_out

    def skip_forward_pass(self, batch: dict, batch_idx: int):
        """
        Skips the forward pass and returns 0.
        """
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    def training_step(self, batch: dict, batch_idx: int):
        """
        Computes training loss for batch of samples.

        Args:
            batch: Data batch.

        Returns:
            Training loss averaged over batch dimension.
        """
        val_step = batch_idx == -1  # validation step is indicated with batch_idx -1
        log_prefix = "validation_loss" if val_step else "train"

        batch = add_clean_samples(
            batch,
            self.cfg_exp.product_flowmatcher,
            getattr(self, "autoencoder", None),
        )

        # Corrupt the batch
        batch = self.fm.corrupt_batch(batch)  # adds x_1, t, x_0, x_t, mask
        bs, n = batch["mask"].shape

        # Handle conditioning variables (safe config getters; missing keys default to disabled)
        batch, n_recycle = handle_batch_conditioning(
            batch,
            bs,
            self.cfg_exp.training,
            self.call_nn,
            self.fm,
        )

        nn_out = self.call_nn(batch, n_recycle=n_recycle)
        losses = self.fm.compute_loss(
            batch=batch,
            nn_out=nn_out,
        )  # Dict[str, Tensor w.batch shape [*]]

        self.log_losses(bs=bs, losses=losses, log_prefix=log_prefix, batch=batch)
        train_loss = sum([torch.mean(losses[k]) for k in losses if "_justlog" not in k])

        self.log(
            f"{log_prefix}/loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=bs,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        if not val_step:  # Don't log these for val step
            self.log_train_loss_n_prog_bar(bs, train_loss)
            self.update_n_log_flops(bs, n)
            self.update_n_log_nsamples_processed(bs)
            self.log_nparams()

        return train_loss

    def log_losses(
        self,
        bs: int,
        losses: dict[str, Float[torch.Tensor, "b"]],
        log_prefix: str,
        batch: dict,
    ):
        for k in losses:
            log_name = k[: -len("_justlog")] if k.endswith("_justlog") else k

            self.log(
                f"{log_prefix}/loss_{log_name}",
                torch.mean(losses[k]),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=bs,
                sync_dist=True,
                add_dataloader_idx=False,
            )

            if self.cfg_exp.training.get("p_folding_n_inv_folding_iters", 0.0) > 0.0:
                # Log also for folding and inverse folding iters
                # divides by p_aux to account for the fact that for most steps loss will be just zero
                # (since need to sync across devices, etc, need to handle logging carefully)
                p_aux = self.cfg_exp.training["p_folding_n_inv_folding_iters"] / 2
                loss = torch.mean(losses[k])  # [b]

                f_inv_fold = batch["use_ca_coors_nm_feature"] * 1.0 / p_aux
                self.log(
                    f"{log_prefix}_invfold_ca_iter/loss_{log_name}",
                    loss * f_inv_fold,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=bs,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

                f_fold = batch["use_residue_type_feature"] * 1.0 / p_aux
                self.log(
                    f"{log_prefix}_fold_iter/loss_{log_name}",
                    loss * f_fold,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=bs,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

    def log_train_loss_n_prog_bar(self, b: int, train_loss: torch.Tensor):
        self.log(
            "train_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=b,
            sync_dist=True,
            add_dataloader_idx=False,
        )

    def log_nparams(self):
        self.log(
            "scaling/nparams",
            self.nparams * 1.0,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )  # constant line but ok, easy to compare # params

    def update_n_log_nsamples_processed(self, b: int):
        self.nsamples_processed = self.nsamples_processed + b * self.trainer.world_size
        self.log(
            "scaling/nsamples_processed",
            self.nsamples_processed * 1.0,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )

    def update_n_log_flops(self, b: int, n: int):
        """
        Updates and logs flops, if available
        """
        try:
            nflops_step = self.nn.nflops_computer(b, n)  # nn should implement this function if we want to see nflops
        except Exception:
            nflops_step = None

        if nflops_step is not None:
            self.nflops = (
                self.nflops + nflops_step * self.trainer.world_size
            )  # Times number of processes so it logs sum across devices
            self.log(
                "scaling/nflops",
                self.nflops * 1.0,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=1,
                sync_dist=True,
            )

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int):
        """Validation step dispatching to length-based generation or data-based loss.

        Args:
            batch: batch from dataset
            batch_idx: batch index (unused)
            dataloader_idx: 0 for length dataloader (generation), 1 for data dataloader (loss)
        """
        if dataloader_idx == 0:
            self.validation_step_lens(batch, batch_idx)
        elif dataloader_idx == 1:
            self.validation_step_data(batch, batch_idx)
        else:
            raise OSError(f"Validation dataloader with index {dataloader_idx} not recognized")

    def validation_step_data(self, batch: dict, batch_idx: int):
        """Evaluates the training loss on validation data."""
        with torch.no_grad():
            loss = self.training_step(batch, batch_idx=-1)
            self.validation_output_data.append(loss.item())

    def validation_step_lens(self, batch: dict, batch_idx: int):
        """
        Generates samples and saves samples in the self.validatoin_output list. Each sample is stored as the atom37
        coordinates, of shape [n, 37, 3]. The variable self.validation_output is thus a list of tensors of shape
        [n, 37, 3].

        Args:
            batch: data batch, contains no data, but the info of the samples to generate (nsamples, nres, dt).

        Returns:
            Nothing, just stores the samples in the list.
        """
        sampling_args = copy.deepcopy(self.cfg_exp.generation.args)
        cath_code = (
            extract_cath_code_from_batch(batch) if sampling_args.fold_cond else None
        )  # When using unconditional model, don't use cath_code
        del sampling_args.fold_cond
        del sampling_args.ag_ckpt_path
        assert sampling_args.ag_ratio == 0.0, "Should turn off autoguidance for validation"

        with torch.no_grad():
            for val_mode in self.cfg_exp.generation.model:
                # fn_predict_for_sampling = partial(
                #     self.predict_for_sampling, n_recycle=0
                # )
                fn_predict_for_sampling = self.predict_for_sampling
                gen_samples = self.fm.full_simulation(
                    predict_for_sampling=fn_predict_for_sampling,
                    batch=batch,
                    nsteps=400,
                    nsamples=batch["nsamples"],
                    n=batch["nres"],
                    self_cond=False,
                    sampling_model_args=self.cfg_exp.generation.model[val_mode],
                    device=self.device,
                )
                # Dict with the data_modes as keys, and values with batch shape b

                # Format the generated samples back to proteins
                sample_prots = sample_formatting(
                    x=gen_samples,
                    extra_info={"mask": batch["mask"]},
                    ret_mode="coors37_n_aatype",
                    data_modes=list(self.cfg_exp.product_flowmatcher),
                    autoencoder=getattr(self, "autoencoder", None),
                )
                # Dict with keys `coors` (a37), `residue_type`, and `mask`,
                # shapes [b, n, 37, 3], [b, n], [b, n]

                generation_list = []
                for i in range(sample_prots["coors"].shape[0]):
                    generation_list.append(
                        (sample_prots["coors"][i], sample_prots["residue_type"][i])
                    )  # Tuple (coors [n, 37, 3], aatype [n])

                if val_mode not in self.validation_output_lens:
                    self.validation_output_lens[val_mode] = []
                self.validation_output_lens[val_mode] += generation_list

    def on_validation_epoch_end(self):
        """Process validation results at epoch end."""
        self.on_validation_epoch_end_data()
        # TODO: Re-enable length-based generation validation once refactored.
        # Disabled because it is expensive (generates full PDB samples + runs
        # structural metrics every val epoch) and currently under rework for
        # the new group-based collation.  Training loss validation still runs.
        # self.on_validation_epoch_end_lens()

    def on_validation_epoch_end_data(self):
        self.validation_output_data = []

    def on_validation_epoch_end_lens(self):
        """
        Generates PDB files from produced samples and computes metrics.
        It does this for all sampling modes considered.
        """
        # Save structures to pdb files
        for val_mode in self.validation_output_lens:
            paths = []
            len_tracker = {}
            for i, (coors_atom37, residue_type) in enumerate(self.validation_output_lens[val_mode]):
                n = coors_atom37.shape[-3]
                len_tracker[n] = len_tracker.get(n, 0)
                len_tracker[n] += 1
                name = (
                    f"epoch_{self.current_epoch}_bid_n_{n}_num_{len_tracker[n]}_rank_{self.global_rank}_{val_mode}.pdb"
                )
                full_path = os.path.join(self.val_path_tmp, name)
                if not os.path.exists(self.val_path_tmp):
                    create_dir(self.val_path_tmp)
                try:
                    write_prot_to_pdb(
                        prot_pos=coors_atom37.float().detach().cpu().numpy(),
                        aatype=residue_type.detach().cpu().numpy(),
                        file_path=full_path,
                        overwrite=True,
                        no_indexing=True,
                    )
                    paths.append(full_path)
                except Exception as e:
                    logger.error(
                        f"[Global rank: {self.global_rank}]: Failed to write protein to PDB on validation: {e}"
                    )

            # Subset of paths for non new metrics
            paths_subset = paths.copy()
            random.shuffle(paths_subset)
            paths_subset = paths_subset[:40]

            # Compute metrics
            try:
                structural_results = get_structural_metrics(paths_subset, val_mode)
                for log_key, value in structural_results.items():
                    self.log(
                        log_key,
                        value,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        logger=True,
                        batch_size=1,
                        sync_dist=True,
                    )
            except Exception as e:
                logger.warning(f"[Global rank: {self.global_rank}]: Failed to get structural metrics: {e}")
            if self.cfg_exp.generation.metric.compute_novelty_pdb and self.global_step > 5000:
                try:
                    novelty_results = get_pdb_novelty_metric(
                        paths_subset,
                        val_mode,
                        self.cfg_exp.hardware.ncpus_per_task_train_,
                    )
                    for log_key, value in novelty_results.items():
                        self.log(
                            log_key,
                            value,
                            on_step=False,
                            on_epoch=True,
                            prog_bar=False,
                            logger=True,
                            batch_size=1,
                            sync_dist=True,
                        )
                except Exception as e:
                    logger.warning(f"[Global rank: {self.global_rank}]: Failed to get pdb novelty metric: {e}")

            # Clean up
            clean_validation_files(paths)
        self.validation_output_lens = {}

    def configure_inference(self, inf_cfg, nn_ag):
        """Sets inference config with all sampling parameters required by the method (dt, etc)
        and autoguidance network (or None if not provided)."""
        self.inf_cfg = inf_cfg
        self.nn_ag = nn_ag

    def generate(self, batch: dict) -> dict:
        """
        Runs a single generation pass with the current configuration.

        Args:
            batch: Data batch containing generation parameters. Must contain 'mask' tensor.

        Returns:
            gen_samples: Dictionary with generated samples for each data mode
        """
        self_cond = self.inf_cfg.args.self_cond
        nsteps = self.inf_cfg.args.nsteps
        guidance_w = self.inf_cfg.args.get("guidance_w", 1.0)
        ag_ratio = self.inf_cfg.args.get("ag_ratio", 0.0)

        fn_predict_for_sampling = partial(self.predict_for_sampling, n_recycle=self.inf_cfg.get("n_recycle", 0))

        # Derive nsamples and n from mask shape
        mask = batch["mask"]
        nsamples, n = mask.shape

        gen_samples = self.fm.full_simulation(
            batch=batch,
            predict_for_sampling=fn_predict_for_sampling,
            nsteps=nsteps,
            nsamples=nsamples,
            n=n,
            self_cond=self_cond,
            sampling_model_args=self.inf_cfg.model,
            device=self.device,
            guidance_w=guidance_w,
            ag_ratio=ag_ratio,
        )

        return gen_samples

    # ------------------------------------------------------------------
    # predict_step helpers
    # ------------------------------------------------------------------

    def _get_search_instance(self):
        """Return cached search instance, re-creating only if algorithm changed."""
        search_algorithm = getattr(self.inf_cfg, "search", {}).get("algorithm", "single-pass")
        if (
            not hasattr(self, "_search_instance")
            or self._search_instance is None
            or getattr(self, "_search_algorithm", None) != search_algorithm
        ):
            self._search_instance = instantiate_search(self, self.inf_cfg, search_algorithm)
            self._search_algorithm = search_algorithm
        return self._search_instance

    def _refinement_enabled(self) -> bool:
        ref_cfg = getattr(self.inf_cfg, "refinement", {})
        return bool(ref_cfg.get("algorithm", None)) if ref_cfg else False

    def _apply_refinement(self, final_prots, lookahead_prots):
        """Refine samples and optionally save pre-refinement copies.

        Only call when ``_refinement_enabled()`` is True.

        Returns (final_prots, lookahead_prots, unrefined_final, unrefined_lookahead).

        NOTE: SequenceHallucination.refine() accepts target_hotspot_mask but
        the original hardcoded hotspot=None.  When hotspot-guided refinement
        is needed, pass expand_hotspot_mask(...) here.
        """
        ref_cfg = self.inf_cfg.refinement
        algorithm = ref_cfg.get("algorithm")

        if not hasattr(self, "_refinement_instance") or self._refinement_instance is None:
            self._refinement_instance = instantiate_refinement(self, self.inf_cfg, algorithm)

        refine_targets = ref_cfg.get("refine_targets", "final")
        save_pre = ref_cfg.get("save_pre_refinement", "none")

        unrefined_final = clone_sample_dict(final_prots) if save_pre in ("final", "all") else None
        unrefined_lookahead = None
        if save_pre == "all" and refine_targets == "all" and lookahead_prots is not None:
            unrefined_lookahead = clone_sample_dict(lookahead_prots)

        final_prots = self._refinement_instance.refine(final_prots)
        if refine_targets == "all" and lookahead_prots is not None:
            lookahead_prots = self._refinement_instance.refine(lookahead_prots)

        return final_prots, lookahead_prots, unrefined_final, unrefined_lookahead

    def _append_unrefined_samples(
        self,
        sample_prots,
        unrefined_final,
        unrefined_lookahead,
        expanded_hotspot_mask,
        ligand,
    ):
        """Score and append pre-refinement copies to the output dict."""
        if unrefined_final is not None:
            unrefined_final_rewards = None
            if self.reward_model is not None:
                unrefined_final_rewards = compute_reward_from_samples(
                    self.reward_model,
                    unrefined_final,
                    expanded_hotspot_mask,
                    ligand,
                )
            append_samples(
                sample_prots,
                unrefined_final,
                unrefined_final_rewards,
                "final_unrefined",
            )
        if unrefined_lookahead is not None:
            append_samples(
                sample_prots,
                unrefined_lookahead,
                unrefined_lookahead.get("rewards"),
                "lookahead_unrefined",
            )

    # ------------------------------------------------------------------
    # predict_step
    # ------------------------------------------------------------------

    @torch.inference_mode(mode=True)
    def predict_step(self, batch: dict, batch_idx: int) -> dict:
        """Run search → (optional) refinement → reward scoring → combine.

        Args:
            batch: Must contain ``mask`` tensor [batch_size, n_residues].

        Returns:
            Dict with ``coors``, ``residue_type``, ``mask``, ``rewards``,
            ``sample_type``, and optionally ``chain_index`` / ``metadata_tag``.
        """
        if "mask" not in batch:
            raise ValueError("Batch must contain 'mask' tensor")

        if not hasattr(self, "reward_model") or self.reward_model is None:
            self.reward_model = initialize_reward_model(self.inf_cfg)

        # ---- Search ----
        search_result = self._get_search_instance().search(batch)
        final_prots = search_result["final"]
        lookahead_prots = search_result.get("lookahead")

        # ---- Refinement (only when configured) ----
        unrefined_final = None
        unrefined_lookahead = None
        if self._refinement_enabled():
            final_prots, lookahead_prots, unrefined_final, unrefined_lookahead = self._apply_refinement(
                final_prots, lookahead_prots
            )

        # ---- Score finals ----
        # Lookaheads are scored during search (tile layout, modulo correct).
        # Finals are scored here.  All search algorithms output finals in
        # grouped layout (all beams/replicas for sample 0 first, then 1, …).
        #
        # BUG FIX: the original passed raw [nsamples] hotspot_mask and used
        # i % nsamples which is wrong for grouped layout.
        # expand_hotspot_mask uses repeat_interleave to match.
        nsamples = batch["mask"].shape[0]
        hotspot_mask = batch.get("target_hotspot_mask")
        ligand = getattr(self, "ligand", None)
        expanded_hotspot_mask = expand_hotspot_mask(
            hotspot_mask,
            final_prots["coors"].shape[0],
            nsamples,
        )

        final_rewards = None
        if self.reward_model is not None:
            final_rewards = compute_reward_from_samples(
                self.reward_model,
                final_prots,
                expanded_hotspot_mask,
                ligand,
            )

        # ---- Combine output ----
        sample_prots = combine_lookahead_and_final(
            lookahead=lookahead_prots,
            final=final_prots,
            final_rewards=final_rewards,
        )

        if unrefined_final is not None or unrefined_lookahead is not None:
            self._append_unrefined_samples(
                sample_prots,
                unrefined_final,
                unrefined_lookahead,
                expanded_hotspot_mask,
                ligand,
            )

        return sample_prots
