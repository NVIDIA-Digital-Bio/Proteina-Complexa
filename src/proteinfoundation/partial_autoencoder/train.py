import json
import os
import pickle
import sys
from pathlib import Path

import hydra
import lightning as L
import torch
import wandb
from dotenv import load_dotenv
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.utilities import rank_zero_only
from loguru import logger
from omegaconf import OmegaConf

from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder
from proteinfoundation.utils.ema_callback import EMA, EmaModelCheckpoint
from proteinfoundation.utils.fetch_last_ckpt import fetch_last_ckpt
from proteinfoundation.utils.seed_callback import SeedCallback
from proteinfoundation.utils.training_analysis_utils import (
    GradAndWeightAnalysisCallback,
    LogEpochTimeCallback,
    LogSetpTimeCallback,
    SkipLargeGradients,
    SkipNanGradCallback,
)


@rank_zero_only
def log_info(msg):
    logger.info(msg)


@rank_zero_only
def create_dir(ckpt_path_store, parents=True, exist_ok=True):
    Path(ckpt_path_store).mkdir(parents=parents, exist_ok=exist_ok)


def check_cluster():
    """
    Verifies whether this is running on the cluster.
    """
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


def get_run_dirs(cfg_exp):
    """
    Get root directory for run and directory to store checkpoints.
    """
    run_name = cfg_exp.run_name
    log_info(f"Job name: {run_name}")
    root_run = os.path.join(".", "store", run_name)  # Everything stored in ./store/<run_id>
    log_info(f"Root run: {root_run}")

    ckpt_path_store = os.path.join(root_run, "checkpoints")  # Checkpoints in ./store/run_id/checkpoints/<ckpt-file>
    log_info(f"Checkpoints directory: {ckpt_path_store}")
    return run_name, root_run, ckpt_path_store


def initialize_callbacks(cfg_exp):
    """
    Initializes general training callbacks.
    """
    callbacks = [SeedCallback()]  # Different devices will be assigend different seeds

    # Gradient and weight stats thoughout training, possibly skip updates with nan in grad
    if cfg_exp.opt.grad_and_weight_analysis:
        callbacks.append(GradAndWeightAnalysisCallback())
    if cfg_exp.opt.skip_nan_grad:
        callbacks.append(SkipNanGradCallback())
    if cfg_exp.opt.skip_large_grad_updates.use:
        callbacks.append(
            SkipLargeGradients(
                moving_avg_size=cfg_exp.opt.skip_large_grad_updates.moving_avg_size,
                factor_threshold=cfg_exp.opt.skip_large_grad_updates.factor_threshold,
                min_opt_steps=cfg_exp.opt.skip_large_grad_updates.min_opt_steps,
            )
        )

    callbacks.append(LogEpochTimeCallback())
    callbacks.append(LogSetpTimeCallback())

    log_info(f"Using EMA with decay {cfg_exp.ema.decay}")
    callbacks.append(EMA(**cfg_exp.ema))
    return callbacks


def get_training_precision(cfg_exp, is_cluster_run):
    """
    Gets and sets correct training precision.
    """
    precision = "32"
    if not cfg_exp.force_precision_f32:
        log_info("Using mixed precision")
        torch.set_float32_matmul_precision("medium")
        if is_cluster_run:
            precision = "bf16-mixed"
        else:
            precision = "16"
    return precision


def load_data_module(cfg_exp, is_cluster_run):
    """
    Loads data config file and creates corresponding datamodule.
    """
    num_cpus = cfg_exp.hardware.ncpus_per_task_train_
    log_info(f"Number of CPUs per task used (will be used for number dataloader number of workers): {num_cpus}")
    cfg_data = cfg_exp.dataset

    cfg_data.datamodule.num_workers = num_cpus  # Overwrite number of cpus
    if cfg_data.get("exclude_id_pkl_path") is not None:
        with open(cfg_data.exclude_id_pkl_path, "rb") as fin:
            exclude_ids = pickle.load(fin)
        if cfg_data.datamodule.dataselector.exclude_ids is not None:
            cfg_data.datamodule.dataselector.exclude_ids += exclude_ids
        else:
            cfg_data.datamodule.dataselector.exclude_ids = exclude_ids
    if not is_cluster_run:
        cfg_data["datamodule"]["batch_size"] = 4
        log_info("Local run, settign batch size to 4")
    log_info(f"Data config {cfg_data}")

    datamodule = hydra.utils.instantiate(cfg_data.datamodule)
    return cfg_data, datamodule


def get_model_n_ckpt_resume(cfg_exp, ckpt_path_store):
    """
    Loads the model and the checkpoint to start training from. This could be just a set
    of parameters (`pretrain_ckpt_path`) or resuming training (`last`). It also handles
    LoRA layers if requested.
    """
    model = AutoEncoder(cfg_exp)

    # get last ckpt if needs to resume training from there
    last_ckpt_name = fetch_last_ckpt(ckpt_path_store)
    last_ckpt_path = os.path.join(ckpt_path_store, last_ckpt_name) if last_ckpt_name is not None else None
    log_info(f"Last checkpoint: {last_ckpt_path}")

    # If this is the first run for fine-tuning, load pre-trained checkpoint and don't load optimizer states
    pretrain_ckpt_path = cfg_exp.get("pretrain_ckpt_path", None)
    if last_ckpt_path is None and pretrain_ckpt_path is not None:
        log_info(f"Loading from pre-trained checkpoint path {pretrain_ckpt_path}")
        ckpt = torch.load(pretrain_ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=False)

    # If not resuming from `last` ckpt training set seed
    if last_ckpt_path is None:
        log_info(f"Seeding everything to seed {cfg_exp.seed}")
        L.seed_everything(cfg_exp.seed)

    return model, last_ckpt_path


def setup_ckpt(cfg_exp, ckpt_path_store):
    """
    Created checkpointing callbacks and creates directory to store checkpoints.
    """
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
def store_n_log_configs(cfg_exp, cfg_data, run_name, ckpt_path_store, wandb_logger):
    """
    Stores config files locally and logs them to wandb run.
    """

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
    config_name="training_ae",
)
def main(cfg_exp) -> None:
    load_dotenv()

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
    cfg_data, datamodule = load_data_module(cfg_exp, is_cluster_run)

    # Create model, warm-up or last ckpt
    model, resume_ckpt_path = get_model_n_ckpt_resume(cfg_exp, ckpt_path_store)

    # logger
    wandb_logger = None
    if cfg_exp.log.log_wandb and not nolog:
        wandb_logger = WandbLogger(
            project=cfg_exp.log.wandb_project, id=run_name, entity=cfg_exp.log.get("wandb_entity", None)
        )

    # checkpoints
    if cfg_exp.log.checkpoint and not nolog:
        ckpt_callbacks = setup_ckpt(cfg_exp, ckpt_path_store)
        callbacks += ckpt_callbacks
        store_n_log_configs(cfg_exp, cfg_data, run_name, ckpt_path_store, wandb_logger)

    # Train
    plugins = [SLURMEnvironment(auto_requeue=True)] if is_cluster_run else []
    # show_prog_bar = args.show_prog_bar or not is_cluster_run
    show_prog_bar = show_prog_bar or not is_cluster_run
    trainer = L.Trainer(
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
        accumulate_grad_batches=cfg_exp.opt.accumulate_grad_batches,
        num_sanity_val_steps=0,
        precision=get_training_precision(cfg_exp, is_cluster_run),
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.0,
        # limit_train_batches=25,
    )
    trainer.fit(model, datamodule, ckpt_path=resume_ckpt_path)
    # If resume_ckpt_path is None then it creates a new optimizer


if __name__ == "__main__":
    main()
