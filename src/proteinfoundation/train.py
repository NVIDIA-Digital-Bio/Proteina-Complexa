# Apply atomworks patches early - before any imports that use atomworks/biotite
import json
import os
import sys
from pathlib import Path

import hydra
import lightning as L
import loralib as lora
import torch
import wandb
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.utilities import rank_zero_only
from loguru import logger
from omegaconf import OmegaConf

import proteinfoundation.patches.atomworks_patches  # noqa: F401
from proteinfoundation.proteina import Proteina
from proteinfoundation.utils.ema_callback import EMA, EmaModelCheckpoint
from proteinfoundation.utils.fetch_last_ckpt import fetch_last_ckpt
from proteinfoundation.utils.fold_utils import transform_global_percentage_to_mask_dropout
from proteinfoundation.utils.lora_utils import replace_lora_layers
from proteinfoundation.utils.seed_callback import SeedCallback
from proteinfoundation.utils.training_analysis_utils import (
    GradAndWeightAnalysisCallback,
    LogEpochTimeCallback,
    LogSetpTimeCallback,
    SkipNanGradCallback,
)


@rank_zero_only
def log_info(msg: str):
    logger.info(msg)


@rank_zero_only
def create_dir(ckpt_path_store: str, parents: bool = True, exist_ok: bool = True):
    Path(ckpt_path_store).mkdir(parents=parents, exist_ok=exist_ok)


def check_cluster() -> bool:
    """Verifies whether this is running on the cluster."""
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    is_cluster_run = slurm_job_id is not None
    if is_cluster_run:
        logger.add(
            sys.stdout,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
        )  # Send to stdout
    log_info(f"Is cluster run: {is_cluster_run}")
    log_info(f"SLURM job id: {slurm_job_id}")
    return is_cluster_run


# TODO: This is no longer used
def handle_cath_conditioning(cfg_exp) -> None:
    """Sets up dropout ratio for CATH conditioning based on global percentage."""
    if cfg_exp.training.get("fold_label_sample_ratio") is not None:
        log_info("Setting fold label dropout rate based on fold_label_sample_ratio")
        (
            cfg_exp.training.mask_T_prob,
            cfg_exp.training.mask_A_prob,
            cfg_exp.training.mask_C_prob,
        ) = transform_global_percentage_to_mask_dropout(cfg_exp.training.fold_label_sample_ratio)
        log_info(
            "Set mask_T_prob: %.3f, mask_A_prob: %.3f, mask_C_prob: %.3f"
            % (
                cfg_exp.training.mask_T_prob,
                cfg_exp.training.mask_A_prob,
                cfg_exp.training.mask_C_prob,
            )
        )
    return cfg_exp


def get_run_dirs(cfg_exp) -> tuple[str, str, str]:
    """Get root directory for run and directory to store checkpoints."""
    run_name = cfg_exp.run_name
    log_info(f"Job name: {run_name}")
    root_run = os.path.join(".", "store", run_name)  # Everything stored in ./store/<run_id>
    log_info(f"Root run: {root_run}")

    ckpt_path_store = os.path.join(root_run, "checkpoints")  # Checkpoints in ./store/run_id/checkpoints/<ckpt-file>
    log_info(f"Checkpoints directory: {ckpt_path_store}")
    return run_name, root_run, ckpt_path_store


def initialize_callbacks(cfg_exp) -> list:
    """Initializes general training callbacks."""
    callbacks = [SeedCallback()]  # , UnusedParametersCallback()]  # Different devices will be assigend different seeds

    # Gradient and weight stats thoughout training, possibly skip updates with nan in grad
    if cfg_exp.opt.grad_and_weight_analysis:
        callbacks.append(GradAndWeightAnalysisCallback())
    if cfg_exp.opt.skip_nan_grad:
        callbacks.append(SkipNanGradCallback())

    callbacks.append(LogEpochTimeCallback())
    callbacks.append(LogSetpTimeCallback())

    log_info(f"Using EMA with decay {cfg_exp.ema.decay}")
    callbacks.append(EMA(**cfg_exp.ema))
    return callbacks


def get_training_precision(cfg_exp, is_cluster_run: bool) -> str:
    """Gets and sets correct training precision."""
    precision = "32"
    if not cfg_exp.force_precision_f32:
        log_info("Using mixed precision")
        torch.set_float32_matmul_precision("medium")
        if is_cluster_run:
            precision = "bf16-mixed"
        else:
            precision = "16"
    else:
        torch.set_float32_matmul_precision("high")
    return precision


def load_data_module(cfg_exp, is_cluster_run: bool) -> tuple:
    """Loads data config and creates corresponding datamodule.

    Supports two patterns:
    1. Unified datamodule (Lightning): config has 'datamodule' key
    2. Atomworks config: config has 'train' key with atomworks dataset definitions
    """
    num_cpus = cfg_exp.hardware.ncpus_per_task_train_
    log_info(f"Number of CPUs per task used (will be used for number dataloader number of workers): {num_cpus}")
    cfg_data = cfg_exp.dataset

    # Check for unified/Lightning datamodule pattern
    if hasattr(cfg_data, "datamodule"):
        # Overwrite number of workers
        if hasattr(cfg_data.datamodule, "num_workers"):
            cfg_data.datamodule.num_workers = num_cpus
        log_info(f"Data config {cfg_data}")

        # Instantiate the datamodule
        datamodule = hydra.utils.instantiate(cfg_data.datamodule)

        # Add validation loader if supported
        if hasattr(datamodule, "add_validation_dataloader") and hasattr(cfg_exp, "generation"):
            cfg_exp_val_data = cfg_exp.generation.dataloader
            n_replicas = cfg_exp.hardware.ngpus_per_node_ * cfg_exp.hardware.nnodes_
            datamodule.add_validation_dataloader(cfg_exp_val_data, n_replicas=n_replicas)

        return cfg_data, datamodule

    # Check for atomworks train config pattern
    elif hasattr(cfg_data, "train"):
        from proteinfoundation.datasets.atomworks_utils import (
            recursively_instantiate_datasets_and_samplers,
            simple_dataloader,
        )

        log_info(f"Data config {cfg_data}")

        # Instantiate datasets and samplers
        dataset_and_sampler = recursively_instantiate_datasets_and_samplers(cfg_data.train)

        # Create dataloader
        train_loader = simple_dataloader(
            dataset=dataset_and_sampler["dataset"],
            loader_cfg=cfg_exp.dataloader["train"],
        )

        # Return a simple namespace-like object that has train_dataloader
        class AtomworksDataModule:
            def __init__(self, train_loader):
                self._train_loader = train_loader

            def train_dataloader(self):
                return self._train_loader

            def val_dataloader(self):
                return None

        datamodule = AtomworksDataModule(train_loader)
        return cfg_data, datamodule

    else:
        raise ValueError(
            "Dataset config must have either 'datamodule' key (for unified/Lightning pattern) "
            "or 'train' key (for atomworks pattern). "
            f"Found keys: {list(cfg_data.keys())}"
        )


def _splice_pretrained_weights(
    model_state_dict: dict,
    ckpt_state_dict: dict,
    # Skip concat_pair_factory entirely - architecture changes too much between versions
    skip_prefixes: tuple[str, ...] = ("nn.concat_pair_factory",),
    # Prefixes whose linear_out* weights may change input dim (e.g., adding conditioning features).
    # Any key matching "<prefix>.linear_out*.weight" will be resized on shape mismatch.
    # Covers v1 (single linear_out) and v2 (linear_out, linear_out_ligand, linear_out_target).
    resizable_prefixes: tuple[str, ...] = (
        "nn.init_repr_factory",
        "nn.cond_factory",
        "nn.pair_repr_builder.init_repr_factory",
        "nn.concat_factory",
    ),
) -> dict:
    """Splice pre-trained weights into a model with potentially different dimensions.

    For resizable keys with shape mismatches: truncates if ckpt is larger,
    zero-pads if model is larger. Skips keys matching skip_prefixes entirely.
    Keys not in resizable_keys with shape mismatches will be silently dropped
    by load_state_dict(strict=False).

    In summary: You only need to resize weights whose input dimension changes due
    to the feature concatenation (the linear_out projections). We skip the concat
    factories themselves as they are either brand new (handled by strict=False)
    or architecturally different (handled by skip_prefixes).

    Args:
        model_state_dict: Current model state dict.
        ckpt_state_dict: Pre-trained checkpoint state dict.
        skip_prefixes: Key prefixes to skip entirely (all weights under prefix).
        resizable_keys: Keys where shape mismatch is handled by truncation/zero-padding.

    Returns:
        Spliced state dict ready for model.load_state_dict(strict=False).
    """

    def _is_resizable(key: str) -> bool:
        """Check if key is a linear_out* weight under a resizable prefix."""
        return any(
            key.startswith(prefix) and "linear_out" in key and key.endswith(".weight") for prefix in resizable_prefixes
        )

    state_dict = {}
    for k, v in ckpt_state_dict.items():
        if any(k.startswith(prefix) for prefix in skip_prefixes):
            continue
        if _is_resizable(k) and k in model_state_dict and model_state_dict[k].shape != v.shape:
            dim_now = model_state_dict[k].shape[-1]
            dim_pre = v.shape[-1]
            if dim_pre > dim_now:
                state_dict[k] = v[..., :dim_now]  # example if we want to remove the chain idx seq feat
            else:  # if the pre-trained ckpt is smaller, we zero-pad the model weights. Cannot be == due to if shape check above
                state_dict[k] = torch.zeros_like(model_state_dict[k])
                state_dict[k][..., :dim_pre] = v
        else:
            state_dict[k] = v
    return state_dict


def get_model_n_ckpt_resume(cfg_exp, ckpt_path_store: str) -> tuple[Proteina, str | None]:
    """Loads model and checkpoint for training.

    Handles pre-trained checkpoint loading (with weight splicing for shape mismatches),
    LoRA layer replacement, and training resumption from last checkpoint.
    """
    model = Proteina(cfg_exp)

    # get last ckpt if needs to resume training from there
    last_ckpt_name = fetch_last_ckpt(ckpt_path_store)
    last_ckpt_path = os.path.join(ckpt_path_store, last_ckpt_name) if last_ckpt_name is not None else None
    log_info(f"Last checkpoint: {last_ckpt_path}")

    # If LoRA is turned on, replace Linear with LoRA layers
    if cfg_exp.get("lora") and cfg_exp.lora.get("r"):
        replace_lora_layers(
            model,
            cfg_exp.lora.r,
            cfg_exp.lora.lora_alpha,
            cfg_exp.lora.lora_dropout,
            cfg_exp.lora.get("exclude_keys", ()),
        )
        lora.mark_only_lora_as_trainable(model, bias=cfg_exp.lora.train_bias)

    # If this is the first run for fine-tuning, load pre-trained checkpoint and don't load optimizer states
    pretrain_ckpt_path = cfg_exp.get("pretrain_ckpt_path", None)
    if last_ckpt_path is None and pretrain_ckpt_path is not None:
        if not os.path.exists(pretrain_ckpt_path):
            raise FileNotFoundError(f"Pre-trained checkpoint not found: {pretrain_ckpt_path}")
        log_info(f"Loading from pre-trained checkpoint path {pretrain_ckpt_path}")
        ckpt = torch.load(pretrain_ckpt_path, map_location="cpu", weights_only=False)
        state_dict = _splice_pretrained_weights(model.state_dict(), ckpt["state_dict"])
        model.load_state_dict(state_dict, strict=False)

    # If not resuming from `last` ckpt training set seed
    if last_ckpt_path is None:
        log_info(f"Seeding everything to seed {cfg_exp.seed}")
        L.seed_everything(cfg_exp.seed)

    # # OPTIMIZATION: Remove decoder from autoencoder during training (only encoder needed)
    # if model.autoencoder is not None:
    #     log_info(
    #         "Removing autoencoder decoder during training to save memory (encoder only needed)"
    #     )
    #     del model.autoencoder.decoder
    #     model.autoencoder.decoder = None
    #     # Force garbage collection to free memory immediately
    #     import gc

    #     gc.collect()
    #     torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return model, last_ckpt_path


def setup_ckpt(cfg_exp, ckpt_path_store: str) -> list:
    """Creates checkpointing callbacks and directory to store checkpoints."""
    args_ckpt_last = {
        "dirpath": ckpt_path_store,
        "save_weights_only": False,
        "filename": "ignore",
        "every_n_train_steps": cfg_exp.log.last_ckpt_every_n_steps,
        "save_last": True,
    }
    args_ckpt = {
        "dirpath": ckpt_path_store,
        "save_last": False,
        "save_weights_only": False,
        "filename": "chk_{epoch:08d}_{step:012d}",
        "every_n_train_steps": cfg_exp.log.checkpoint_every_n_steps,
        "monitor": "train_loss",
        "save_top_k": 10000,
        "mode": "min",
    }
    checkpoint_callback = EmaModelCheckpoint(**args_ckpt)
    checkpoint_callback_last = EmaModelCheckpoint(**args_ckpt_last)

    create_dir(ckpt_path_store, parents=True, exist_ok=True)
    return [checkpoint_callback, checkpoint_callback_last]


@rank_zero_only
def store_n_log_configs(cfg_exp, cfg_data, run_name: str, ckpt_path_store: str, wandb_logger) -> None:
    """Stores config files locally and logs them to wandb run."""

    def store_n_log_config(cfg, cfg_path, wandb_logger):
        with open(cfg_path, "w") as f:
            cfg_aux = OmegaConf.to_container(cfg, resolve=True)
            json.dump(cfg_aux, f, indent=4, sort_keys=True)

        if wandb_logger is not None:
            artifact = wandb.Artifact(f"config_files_{run_name}", type="config")
            artifact.add_file(cfg_path)
            wandb_logger.experiment.log_artifact(artifact)

    cfg_exp_file = os.path.join(ckpt_path_store, f"exp_config_{run_name}.json")
    cfg_data_file = os.path.join(ckpt_path_store, f"data_config_{run_name}.json")

    store_n_log_config(cfg_exp, cfg_exp_file, wandb_logger)
    store_n_log_config(cfg_data, cfg_data_file, wandb_logger)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="training_local_latents",
)
def main(cfg_exp) -> None:
    load_dotenv()
    log_info(f"Name of config being used: {HydraConfig.get().job.config_name}")

    is_cluster_run = check_cluster()
    nolog = cfg_exp.get("nolog", False)  # To use do `python proteinfoundation/train.py +nolog=true`
    single = cfg_exp.get("single", False)
    show_prog_bar = cfg_exp.get("show_prog_bar", False)
    if not is_cluster_run or single:
        # Rewrite number of GPUs and nodes for local runs or if single flag is used
        cfg_exp.hardware.ngpus_per_node_ = 1
        cfg_exp.hardware.nnodes_ = 1
        cfg_exp.run_name = cfg_exp.run_name + "_local"
    log_info(f"Exp config {cfg_exp}")

    run_name, root_run, ckpt_path_store = get_run_dirs(cfg_exp)
    callbacks = initialize_callbacks(cfg_exp)

    # logger
    wandb_logger = None
    if cfg_exp.log.log_wandb and not nolog:
        wandb_project = cfg_exp.log.wandb_project
        wandb_id = run_name
        wandb_entity = cfg_exp.log.get("wandb_entity", None)
        logger.info(f"Using WandB logger with project={wandb_project}, run name={wandb_id}, entity={wandb_entity}")
        wandb_logger = WandbLogger(
            project=wandb_project,
            id=run_name,
            entity=wandb_entity,
        )

    Trainer = L.Trainer

    cfg_data, datamodule = load_data_module(cfg_exp, is_cluster_run)

    # checkpoints
    if cfg_exp.log.checkpoint:  # and not nolog:
        ckpt_callbacks = setup_ckpt(cfg_exp, ckpt_path_store)
        callbacks += ckpt_callbacks
        store_n_log_configs(cfg_exp, cfg_data, run_name, ckpt_path_store, wandb_logger)

    # Train
    plugins = [SLURMEnvironment(auto_requeue=True)] if is_cluster_run else []
    # show_prog_bar = args.show_prog_bar or not is_cluster_run
    show_prog_bar = show_prog_bar or not is_cluster_run
    trainer = Trainer(
        max_epochs=cfg_exp.opt.max_epochs,
        accelerator=cfg_exp.hardware.accelerator,
        devices=cfg_exp.hardware.ngpus_per_node_,  # This is number of gpus per node, not total
        num_nodes=cfg_exp.hardware.nnodes_,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=cfg_exp.log.log_every_n_steps,
        default_root_dir=root_run,
        check_val_every_n_epoch=None,  # Leave like this
        val_check_interval=cfg_exp.opt.val_check_interval,
        strategy=cfg_exp.opt.dist_strategy,
        enable_progress_bar=show_prog_bar,
        plugins=plugins,
        limit_val_batches=100,
        accumulate_grad_batches=cfg_exp.opt.accumulate_grad_batches,
        num_sanity_val_steps=0,
        precision=get_training_precision(cfg_exp, is_cluster_run),
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.0,
        limit_train_batches=cfg_exp.opt.get("limit_train_batches", None),
    )
    # Create model, warm-up or last ckpt
    model, resume_ckpt_path = get_model_n_ckpt_resume(cfg_exp, ckpt_path_store)
    trainer.fit(model, datamodule, ckpt_path=resume_ckpt_path)


if __name__ == "__main__":
    main()
