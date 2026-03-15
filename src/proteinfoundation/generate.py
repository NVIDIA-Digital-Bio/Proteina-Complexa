# Apply atomworks patches early - before any imports that use atomworks/biotite
import proteinfoundation.patches.atomworks_patches  # noqa: F401

# Suppress warnings before any other imports
from proteinfoundation.cli.startup import quiet_startup

quiet_startup()

import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import biotite
import hydra
import lightning as L
import loralib as lora
import numpy as np
import pandas as pd
import torch
from atomworks.ml.encoding_definitions import AF2_ATOM37_ENCODING
from atomworks.ml.transforms.encoding import atom_array_from_encoding
from dotenv import load_dotenv
from loguru import logger
from omegaconf import open_dict

from proteinfoundation.proteina import Proteina
from proteinfoundation.rewards.base_reward import TOTAL_REWARD_KEY
from proteinfoundation.utils.config_utils import filter_config_for_logging
from proteinfoundation.utils.lora_utils import replace_lora_layers
from proteinfoundation.utils.pdb_utils import write_prot_to_pdb


def setup(
    cfg: dict,
    create_root: bool = True,
    config_name: str = ".",
    job_id: int = 0,
    task_name: str = None,
    run_name: str = None,
) -> str:
    """
    Checks if metrics being computed are compatible, sets the right seed, and creates the root directory
    where the run will store things.

    Returns:
        Path of the root directory (string)
    """
    logger.info(" ".join(sys.argv))

    assert torch.cuda.is_available(), "CUDA not available"  # Needed for ESMfold and designability
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
    )  # Send to stdout

    # Set root path for this inference run
    if task_name is not None:
        # now = datetime.now()
        # timestamp = f"Y{now.year}_M{now.month:02d}_D{now.day:02d}_H{now.hour:02d}_M{now.minute:02d}_S{now.second:02d}"
        root_path = f"./inference/{config_name}_{task_name}"  # _{timestamp}"
        if run_name is not None:
            root_path = f"{root_path}_{run_name}"
    else:
        root_path = f"./inference/{config_name}"
    if create_root:
        os.makedirs(root_path, exist_ok=True)
    else:
        if not os.path.exists(root_path):
            raise ValueError("Results path %s does not exist" % root_path)

    # Set seed
    cfg.seed = cfg.seed + job_id  # Different seeds for different splits ids
    logger.info(f"Seeding everything to seed {cfg.seed}")
    L.seed_everything(cfg.seed)

    return root_path


def validate_checkpoint_paths(cfg: dict) -> None:
    """Validate that checkpoint paths exist and are readable before expensive model loading."""
    ckpt_dir = cfg.get("ckpt_path")
    ckpt_name = cfg.get("ckpt_name")
    ae_ckpt_path = cfg.get("autoencoder_ckpt_path")

    errors = []

    if ckpt_dir is None:
        errors.append("ckpt_path is not set in config")
    elif not os.path.isdir(ckpt_dir):
        errors.append(f"ckpt_path directory does not exist: {ckpt_dir}")
    elif not os.access(ckpt_dir, os.R_OK):
        errors.append(f"ckpt_path directory is not readable: {ckpt_dir}")

    if ckpt_name is None:
        errors.append("ckpt_name is not set in config")
    elif ckpt_dir and os.path.isdir(ckpt_dir):
        ckpt_file = os.path.join(ckpt_dir, ckpt_name)
        if not os.path.isfile(ckpt_file):
            errors.append(f"Checkpoint file does not exist: {ckpt_file}")
        elif not os.access(ckpt_file, os.R_OK):
            errors.append(f"Checkpoint file is not readable: {ckpt_file}")

    if ae_ckpt_path is not None:
        if not os.path.isfile(ae_ckpt_path):
            errors.append(f"autoencoder_ckpt_path does not exist: {ae_ckpt_path}")
        elif not os.access(ae_ckpt_path, os.R_OK):
            errors.append(f"autoencoder_ckpt_path is not readable: {ae_ckpt_path}")

    if errors:
        for err in errors:
            logger.error(err)
        raise FileNotFoundError("Checkpoint validation failed:\n  " + "\n  ".join(errors))

    ckpt_file = os.path.join(ckpt_dir, ckpt_name)
    logger.info(f"Checkpoint validated: {ckpt_file}")
    if ae_ckpt_path:
        logger.info(f"Autoencoder checkpoint validated: {ae_ckpt_path}")


def check_cfg_validity(cfg_data: dict, cfg_sample_args: dict) -> None:
    """
    Checks if guidance arguments (CFG and AG) are valid.
    """
    # Logging CFG
    if cfg_sample_args.guidance_w != 1.0:
        logger.info(
            f"Guidance is turned on with guidance weight {cfg_sample_args.guidance_w} and autoguidance ratio {cfg_sample_args.ag_ratio}."
        )
        assert cfg_sample_args.ag_ratio >= 0.0 and cfg_sample_args.ag_ratio <= 1.0, (
            f"Autoguidance ratio should be between 0 and 1, but now is {cfg_sample_args.ag_ratio}."
        )
        assert (cfg_sample_args.ag_ratio == 1.0) or cfg_sample_args.fold_cond, (
            "Classifier-free guidance should only be turned on for conditional generation."
        )
        assert (cfg_sample_args.ag_ratio == 0.0) or (cfg_sample_args.ag_ckpt_path is not None), (
            "Autoguidance checkpoint path should be provided"
        )
    else:
        logger.info("Guidance is turned off.")


def load_ag_ckpt(cfg: dict) -> None | torch.nn.Module:
    """
    Loads the neural network for the "bad" checkpoint in autoguidance, if requested.

    Returns:
        A nn module, if autogudance enabled.
    """
    nn_ag = None
    if cfg.ag_ratio > 0 and cfg.guidance_w != 1.0:
        logger.info(
            f"Using autoguidance with guidance weight {cfg.guidance_w} and autoguidance ratio {cfg.ag_ratio} based on the checkpoint {cfg.ag_ckpt_path}"
        )
        ckpt_ag_file = cfg.ag_ckpt_path
        assert os.path.exists(ckpt_ag_file), f"Not a valid checkpoint {ckpt_ag_file}"
        model_ag = Proteina.load_from_checkpoint(ckpt_ag_file)

        # OPTIMIZATION: Remove encoder from autoguidance model autoencoder during generation (only decoder needed)
        if model_ag.autoencoder is not None:
            logger.info("Removing autoencoder encoder from autoguidance model during generation to save memory")
            del model_ag.autoencoder.encoder
            model_ag.autoencoder.encoder = None

        nn_ag = model_ag.nn
    return nn_ag


def load_ckpt_n_configure_inference(cfg: dict) -> Proteina:
    """
    Loads the model, potentially the autoguidance checkpoint as well, if requested.

    Returns:
        Model (Proteina)
    """
    # Load model from checkpoint
    ckpt_path = cfg.ckpt_path
    ckpt_file = os.path.join(ckpt_path, cfg.ckpt_name)
    logger.info(f"Using checkpoint {ckpt_file}")
    assert os.path.exists(ckpt_file), f"Not a valid checkpoint {ckpt_file}"

    model = Proteina.load_from_checkpoint(
        ckpt_file,
        strict=False,
        autoencoder_ckpt_path=cfg.get("autoencoder_ckpt_path"),
    )

    # HARDCODED FIX: Load full autoencoder from a separate checkpoint if decoder is missing
    if hasattr(model, "autoencoder") and model.autoencoder is not None and model.autoencoder.decoder is None:
        logger.info("Decoder missing from main checkpoint, loading full autoencoder from hardcoded path")

        full_autoencoder_path = cfg.get("autoencoder_ckpt_path")
        if full_autoencoder_path is None:
            raise ValueError("autoencoder_ckpt_path not found in config")
        if os.path.exists(full_autoencoder_path):
            logger.info(f"Loading full autoencoder from: {full_autoencoder_path}")
            full_model = Proteina.load_from_checkpoint(full_autoencoder_path, strict=False)

            # Replace the autoencoder with the full one
            if full_model.autoencoder is not None and full_model.autoencoder.decoder is not None:
                model.autoencoder = full_model.autoencoder
                logger.info("Successfully loaded full autoencoder with decoder")

                # Clean up
                del full_model
                import gc

                gc.collect()
            else:
                logger.warning("Full autoencoder checkpoint also doesn't have decoder")
        else:
            logger.warning(f"Full autoencoder checkpoint not found at: {full_autoencoder_path}")

    # If using lora, create lora layers and reload the state_dict
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    if any(["lora" in key for key in ckpt["state_dict"].keys()]) and not (cfg.get("lora") and cfg["lora"].get("r")):
        raise ValueError("LoRA layers found in checkpoint but not in config")
    if cfg.get("lora") and cfg["lora"].get("r"):
        logger.info("Re-create LoRA layers and reload the weights now")
        replace_lora_layers(
            model,
            cfg["lora"]["r"],
            cfg["lora"]["lora_alpha"],
            cfg["lora"]["lora_dropout"],
        )
        lora.mark_only_lora_as_trainable(model, bias=cfg["lora"]["train_bias"])
        ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=False)  # Set strict=False here too

    # Set inference variables and potentially load autoguidance
    nn_ag = load_ag_ckpt(cfg.generation.args)

    model.configure_inference(cfg.generation, nn_ag=nn_ag)

    return model


def split_by_job(cfg: dict, job_id: int, njobs: int) -> dict:
    """
    Since generation may be split across multiple jobs, this function determines how many samples are produced per job.
    Then, it sets the right value in the config dict, and returns the updated config.

    Returns:
        Config updated with the correct number of samples to generate.
    """
    nsamples = cfg.dataloader.dataset.nrepeat_per_sample
    if nsamples == 1 and njobs > 1:
        if hasattr(cfg.dataloader.dataset.nres, "nsamples"):
            n = cfg.dataloader.dataset.nres.nsamples
            nsamples_per_split = (n - 1) // njobs + 1
            if nsamples_per_split * job_id >= n:
                logger.info(f"Job id {job_id} get 0 samples. Finishing job...")
                sys.exit(0)
            else:
                cfg.dataloader.dataset.nres.nsamples = min(nsamples_per_split, n - nsamples_per_split * job_id)
        # motif use nsamples from conditional feature
        if cfg.dataloader.dataset.conditional_features is not None:
            for conditional_feature in cfg.dataloader.dataset.conditional_features:
                if hasattr(conditional_feature, "nsamples"):
                    n = conditional_feature.nsamples
                    nsamples_per_split = (n - 1) // njobs + 1
                    if nsamples_per_split * job_id >= n:
                        logger.info(f"Job id {job_id} get 0 samples. Finishing job...")
                        sys.exit(0)
                    else:
                        conditional_feature.nsamples = min(nsamples_per_split, n - nsamples_per_split * job_id)
    else:
        nsamples_per_split = (nsamples - 1) // njobs + 1
        if nsamples_per_split * job_id >= nsamples:
            logger.info(f"Job id {job_id} get 0 samples. Finishing job...")
            sys.exit(0)
        else:
            cfg.dataloader.dataset.nrepeat_per_sample = min(nsamples_per_split, nsamples - nsamples_per_split * job_id)
    return cfg


def save_predictions(
    root_path: str,
    predictions: list[dict],
    job_id: int = 0,
    cath_codes: list[list[list[str]]] = None,
    suffix: str = "",
) -> tuple[list[str], pd.DataFrame]:
    """
    Saves generated samples.

    Args:
        root_path: root directory where samples will be stored (within subdirectories)/
        predictions: List of dicts (one per batch). Each dict contains:
            - 'coors': [batch_size, n, 37, 3]
            - 'residue_type': [batch_size, n]
            - 'mask': [batch_size, n]
            - 'chain_index': [batch_size, n] (optional)
            - 'rewards': [batch_size] (optional)
            - 'reward_components': list of dict per sample (optional)
            - 'sample_type': list of "lookahead"|"final" (optional)
        job_id: job number, used to store files.
        cath_codes: conditional sampling...
        suffix: suffix to add to directory names

    Returns:
        Tuple of (pdb_paths, reward_df)
    """
    pdb_paths = []
    rewards = []
    samples_per_length = defaultdict(int)
    for batch_idx, batch_pred in enumerate(predictions):
        batch_size = batch_pred["coors"].shape[0]
        for i in range(batch_size):
            coors_atom37 = batch_pred["coors"][i]  # [n, 37, 3]
            residue_type = batch_pred["residue_type"][i]  # [n]
            chain_index = batch_pred.get("chain_index", None)
            if chain_index is not None:
                chain_index = chain_index[i]  # [n]
            rewards_dict = batch_pred.get("rewards", None)
            n = np.sum(np.sum(np.abs(coors_atom37[:, 1, :].detach().cpu().numpy()), axis=-1) > 1e-7)
            if "binder" in suffix and chain_index is None:
                chain_index = torch.ones(coors_atom37.shape[0])
            # # Create directory where everything related to this sample will be stored
            # if cath_codes[j] is not None:
            #     suffix = "_fold_" + "+".join(cath_codes[j][i])
            # else:
            # suffix = ""
            meta = ""
            if "metadata_tag" in batch_pred and i < len(batch_pred["metadata_tag"]):
                tag = batch_pred["metadata_tag"][i]
                if tag:
                    meta = f"_{tag}"
            dir_name = f"job_{job_id}_n_{n}_id_{samples_per_length[n]}{meta}"
            dir_name_suffix = f"job_{job_id}_n_{n}_id_{samples_per_length[n]}{meta}{suffix}"
            samples_per_length[n] += 1
            sample_root_path = os.path.join(root_path, dir_name)
            os.makedirs(sample_root_path, exist_ok=True)

            dir_name = dir_name_suffix
            fname = dir_name + ".pdb"
            pdb_path = os.path.join(sample_root_path, fname)
            if chain_index is not None:
                chain_index = chain_index.detach().cpu().numpy()
            write_prot_to_pdb(
                prot_pos=coors_atom37.float().detach().cpu().numpy(),
                aatype=residue_type.detach().cpu().numpy(),
                file_path=pdb_path,
                chain_index=chain_index,
                overwrite=True,
                no_indexing=True,
            )
            pdb_paths.append(pdb_path)

            sample_type = None
            if "sample_type" in batch_pred and i < len(batch_pred["sample_type"]):
                sample_type = batch_pred["sample_type"][i]

            row_data = {
                "pdb_path": os.path.abspath(pdb_path),
                "pdb_index": i,
                "aatype": ",".join(residue_type.detach().cpu().numpy().astype(str)),
            }
            # Add total_reward and reward components from unified rewards dict
            if rewards_dict is not None:
                row_data["total_reward"] = rewards_dict[TOTAL_REWARD_KEY][i].float().detach().cpu().numpy()
                for key, tensor in rewards_dict.items():
                    if key != TOTAL_REWARD_KEY:
                        row_data[key] = tensor[i].float().detach().cpu().numpy()
            else:
                row_data["total_reward"] = np.nan
            if sample_type is not None:
                row_data["sample_type"] = sample_type
            if "metadata_tag" in batch_pred and i < len(batch_pred["metadata_tag"]):
                row_data["metadata_tag"] = batch_pred["metadata_tag"][i]
            rewards.append(row_data)
    return pdb_paths, pd.DataFrame(rewards)


def save_protein_ligand_predictions(
    root_path: str,
    predictions: list[dict],
    complex_arrays: list,
    job_id: int = 0,
    cath_codes: list[list[list[str]]] = None,
    suffix: str = "",
) -> tuple[list[str], pd.DataFrame]:
    """
    Saves generated samples with ligands.

    Args:
        root_path: root directory where samples will be stored (within subdirectories)/
        predictions: List of dicts (one per batch). Each dict contains:
            - 'coors': [batch_size, n, 37, 3]
            - 'residue_type': [batch_size, n]
            - 'mask': [batch_size, n]
            - 'chain_index': [batch_size, n] (optional)
            - 'rewards': [batch_size] (optional)
        complex_arrays: List of complex arrays (one per batch)
        job_id: job number, used to store files.
        cath_codes: conditional sampling...
        suffix: suffix to add to directory names

    Returns:
        Tuple of (pdb_paths, reward_df)
    """
    pdb_paths = []
    rewards = []
    samples_per_length = defaultdict(int)
    for batch_idx, batch_pred in enumerate(predictions):
        batch_size = batch_pred["coors"].shape[0]
        for i in range(batch_size):
            coors_atom37 = batch_pred["coors"][i]  # [n, 37, 3]
            residue_type = batch_pred["residue_type"][i]  # [n]
            chain_index = batch_pred.get("chain_index", None)
            if chain_index is not None:
                chain_index = chain_index[i]  # [n]
            rewards_dict = batch_pred.get("rewards", None)

            n = np.sum(np.sum(np.abs(coors_atom37[:, 1, :].detach().cpu().numpy()), axis=-1) > 1e-7)

            # # Create directory where everything related to this sample will be stored
            # if cath_codes[j] is not None:
            #     suffix = "_fold_" + "+".join(cath_codes[j][i])
            # else:
            # suffix = "_complex"
            meta = ""
            if "metadata_tag" in batch_pred and i < len(batch_pred["metadata_tag"]):
                tag = batch_pred["metadata_tag"][i]
                if tag:
                    meta = f"_{tag}"
            dir_name = f"job_{job_id}_n_{n}_id_{samples_per_length[n]}{meta}{suffix}"
            samples_per_length[n] += 1
            sample_root_path = os.path.join(root_path, dir_name)
            os.makedirs(sample_root_path, exist_ok=True)

            fname = dir_name + ".pdb"
            pdb_path = os.path.join(sample_root_path, fname)
            if chain_index is not None:
                chain_index = chain_index.detach().cpu().numpy()
            biotite.structure.io.save_structure(pdb_path, complex_arrays[batch_idx][i])
            pdb_paths.append(pdb_path)

            sample_type = None
            if "sample_type" in batch_pred and i < len(batch_pred["sample_type"]):
                sample_type = batch_pred["sample_type"][i]

            row_data = {
                "pdb_path": os.path.abspath(pdb_path),
                "pdb_index": i,
                "aatype": ",".join(residue_type.detach().cpu().numpy().astype(str)),
            }
            if rewards_dict is not None:
                row_data["total_reward"] = rewards_dict[TOTAL_REWARD_KEY][i].float().detach().cpu().numpy()
                for key, tensor in rewards_dict.items():
                    if key != TOTAL_REWARD_KEY:
                        row_data[key] = tensor[i].float().detach().cpu().numpy()
            else:
                row_data["total_reward"] = np.nan
            if sample_type is not None:
                row_data["sample_type"] = sample_type
            if "metadata_tag" in batch_pred and i < len(batch_pred["metadata_tag"]):
                row_data["metadata_tag"] = batch_pred["metadata_tag"][i]
            rewards.append(row_data)

    return pdb_paths, pd.DataFrame(rewards)


def save_motif_predictions(
    root_path: str,
    predictions: list[dict],
    job_id: int = 0,
    motif_pdb_name: str = None,
) -> list[str]:
    """
    Saves generated motif samples.

    Args:
        root_path: root directory where samples will be stored
        predictions: List of dicts (one per batch). Each dict contains:
            - 'coors': [batch_size, n, 37, 3]
            - 'residue_type': [batch_size, n]
            - 'mask': [batch_size, n]
        job_id: job number, used to store files.
        motif_pdb_name: name of motif PDB file

    Returns:
        List of PDB file paths
    """
    pdb_paths = []
    sample_idx = 0
    for batch_pred in predictions:
        batch_size = batch_pred["coors"].shape[0]
        for i in range(batch_size):
            coors_atom37 = batch_pred["coors"][i]  # [n, 37, 3]
            residue_type = batch_pred["residue_type"][i]  # [n]
            dir_name = f"job_{job_id}_id_{sample_idx}_motif_{motif_pdb_name}"
            sample_idx += 1
            sample_root_path = os.path.join(root_path, dir_name)
            os.makedirs(sample_root_path, exist_ok=True)
            fname = dir_name + ".pdb"
            pdb_path = os.path.join(sample_root_path, fname)
            write_prot_to_pdb(
                prot_pos=coors_atom37.float().detach().cpu().numpy(),
                aatype=residue_type.detach().cpu().numpy(),
                file_path=pdb_path,
                overwrite=True,
                no_indexing=True,
            )
            pdb_paths.append(pdb_path)

    return pdb_paths


def save_rewards_to_csv(df: pd.DataFrame, root_path: str, config_name: str, job_id: int) -> str:
    """
    Save reward DataFrame to CSV file.

    Args:
        df: DataFrame containing reward results
        root_path: Root directory for saving results
        config_name: Name of the configuration
        job_id: Job ID

    Returns:
        Path to the saved CSV file
    """
    csv_filename = f"rewards_{config_name}_{job_id}.csv"
    csv_path = os.path.join(root_path, csv_filename)

    df.to_csv(csv_path, index=False)
    logger.info(f"Rewards saved to: {csv_path}")

    return csv_path


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="inference_base",
)
def main(cfg):
    load_dotenv()
    validate_checkpoint_paths(cfg)
    # the base config name in case the config file is generated from sweep script.
    config_name = cfg.get("base_config_name", hydra.core.hydra_config.HydraConfig.get().job.config_name)
    job_id = cfg.get("job_id", 0)
    root_path = cfg.get("root_path", None)
    save_timing = cfg.get("save_timing", True)
    run_name = cfg.get("run_name", None)
    task_name = cfg.generation.dataloader.dataset.get("task_name", None)
    conditional_features_types = []
    conditional_features_cfg = cfg.generation.dataloader.dataset.get("conditional_features", None)
    if conditional_features_cfg is not None:
        for conditional_feature_cfg in conditional_features_cfg:
            conditional_features_types.append(conditional_feature_cfg._target_.split(".")[-1])

    ligand_cond = "LigandFeatures" in conditional_features_types
    target_cond = "TargetFeatures" in conditional_features_types
    motif_cond = "MotifFeatures" in conditional_features_types
    fold_cond = "CathCodes" in conditional_features_types
    cath_codes = (
        cfg.generation.dataloader.dataset.conditional_features[conditional_features_types.index("CathCodes")].cath_codes
        if fold_cond
        else None
    )

    if root_path is None:
        root_path = setup(
            cfg,
            create_root=True,
            config_name=config_name,
            job_id=job_id,
            task_name=task_name,
            run_name=run_name,
        )
    else:
        os.makedirs(root_path, exist_ok=True)

    # Record start time
    start_time = time.time()
    logger.info(f"Starting generation job at {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")

    njobs = cfg.get("gen_njobs", 1)

    # Exit if results from analysis already exist (assumes samples already there)
    # File to store analysis (next step, this is generate) results
    csv_filename = f"results_{config_name}_{job_id}.csv"
    csv_path = os.path.join(root_path, "..", csv_filename)
    # Exit if results from analysis already exist
    if os.path.exists(csv_path):
        logger.info(f"Results already exist at {csv_path}. Exiting generate.py.")
        sys.exit(0)

    cfg_gen = cfg.generation
    check_cfg_validity(cfg_gen.dataloader.dataset, cfg_gen.args)

    # Load model
    model = load_ckpt_n_configure_inference(cfg)
    torch.set_float32_matmul_precision("high")

    # Create generation dataset
    cfg_gen = split_by_job(cfg_gen, job_id, njobs)

    # Motif-specific dataset creation
    if motif_cond:
        motif_csv_path = os.path.join(
            root_path,
            f"{task_name or 'motif'}_{job_id}_motif_info.csv",
        )
        """
        Motif Configuration Examples:
        
        The motif dataset supports two modes for specifying which atoms to include:
        
        1. **Atom-level specification** (precise control):
           motif_dict_cfg:
             my_motif:
               motif_pdb_path: "path/to/motif.pdb"
               motif_atom_spec: "A64: [O, CG]; A65: [N, CA]; A66: [CB, CD]"
               # atom_selection_mode is ignored when motif_atom_spec is provided
        
        2. **Residue/range-based specification** (automatic atom selection):
           motif_dict_cfg:
             my_motif:
               motif_pdb_path: "path/to/motif.pdb" 
               contig_string: "A1-7/A28-79"
               atom_selection_mode: "tip_atoms"  # NEW: Choose atom selection mode
               
           Available atom_selection_mode options:
           - "ca_only": Only CA atoms (default, fastest)
           - "all": All available atoms (most complete motif)
           - "backbone": Backbone atoms only (N, CA, C, O)
           - "sidechain": Sidechain atoms only
           - "tip_atoms": Tip atoms of sidechains (e.g., OH for Ser, NH2 for Arg)
           - "random": Random subset of available atoms
           
        If atom_selection_mode is not specified, defaults to "ca_only" for backward compatibility.
        """
        if conditional_features_cfg is not None:
            with open_dict(conditional_features_cfg):
                for conditional_feature_cfg in conditional_features_cfg:
                    if hasattr(conditional_feature_cfg, "motif_csv_path"):
                        conditional_feature_cfg.motif_csv_path = motif_csv_path
                        break
    logger.info(f"cfg_gen: {filter_config_for_logging(cfg_gen)}")
    dataloader = hydra.utils.instantiate(cfg_gen.dataloader)
    dataset = dataloader.dataset

    if ligand_cond:  #! this is where se set self.ligand for the model
        ligand = dataset.conditional_features[conditional_features_types.index("LigandFeatures")].ligand
        model.ligand = ligand  # shouldn't be set here, but pass in dataset

    # Sample model
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        inference_mode=False,
    )  # set it to False, as we need refinement in predict step
    predictions = trainer.predict(model, dataloader)
    # predictions is now a list of dicts (one per batch), each dict contains:
    # - 'coors': [batch_size, n, 37, 3]
    # - 'residue_type': [batch_size, n]
    # - 'mask': [batch_size, n]
    # - 'chain_index': [batch_size, n] (optional)
    # - 'rewards': [batch_size] (optional)

    if ligand_cond:
        complex_arrays = []
        for batch_idx, batch_pred in enumerate(predictions):
            complex_sublist = []
            batch_size = batch_pred["coors"].shape[0]
            for i in range(batch_size):
                coors_atom37 = batch_pred["coors"][i]  # [n, 37, 3]
                residue_type = batch_pred["residue_type"][i]  # [n]
                #! TODO figure out why OXT's at origin are added here
                atom37_mask = torch.sum(torch.abs(coors_atom37), axis=-1) > 1e-7
                prot = atom_array_from_encoding(
                    encoded_coord=coors_atom37,
                    encoded_mask=atom37_mask,
                    encoded_seq=residue_type.int(),
                    encoding=AF2_ATOM37_ENCODING,
                ).copy()
                #! the above line adds OXT occupancy 0 for all residues
                prot = prot[prot.occupancy > 0]  #! this is the fix it should drop N atoms where N is number of residues
                res_names = np.array([name for name in prot.res_name], dtype="U5")
                prot.del_annotation("res_name")
                prot.set_annotation("res_name", res_names)
                lig_coords = (
                    dataset[0]["x_target"].clone().numpy() * 10.0
                )  # convert to angstroms # assume only one type of ligand in the dataset
                lig = ligand.copy()
                lig.coord = lig_coords
                chain_b = np.array(["B" for chain in prot.chain_id], dtype=lig.chain_id.dtype)
                prot.del_annotation("chain_id")
                prot.set_annotation("chain_id", chain_b)
                complex_array = biotite.structure.concatenate([prot, lig])
                complex_sublist.append(complex_array)
            complex_arrays.append(complex_sublist)

        pdb_paths, reward_df = save_predictions(
            root_path,
            predictions,
            job_id=job_id,
            cath_codes=cath_codes,
            suffix="_binder",
        )
        complex_paths, _complex_reward_df = save_protein_ligand_predictions(
            root_path,
            predictions,
            complex_arrays,
            job_id=job_id,
            cath_codes=cath_codes,
        )
        if len(reward_df) > 0:
            csv_path = save_rewards_to_csv(
                df=reward_df,
                root_path=root_path,
                config_name=config_name,
                job_id=job_id,
            )
    elif motif_cond:
        pdb_paths = save_motif_predictions(
            root_path,
            predictions,
            job_id=job_id,
            motif_pdb_name=task_name,
        )
        import shutil

        motif_csv = f"./{task_name or ''}_motif_info.csv"
        if os.path.exists(motif_csv):
            shutil.copy(motif_csv, root_path)
    else:
        pdb_paths, reward_df = save_predictions(
            root_path,
            predictions,
            job_id=job_id,
            cath_codes=cath_codes,
        )

        if len(reward_df) > 0:
            csv_path = save_rewards_to_csv(
                df=reward_df,
                root_path=root_path,
                config_name=config_name,
                job_id=job_id,
            )

    # Record end time
    end_time = time.time()
    logger.info(f"Generation job finished at {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total generation time: {end_time - start_time:.2f} seconds")

    # Save timing information to CSV
    if save_timing:
        # timing_csv_path = os.path.join(root_path, "..", f"timing_{config_name}_{job_id}.csv")
        # timing_csv_path = os.path.join(root_path, f"timing_{config_name}_{job_id}.csv") #! this puts it in the inf_XX
        timing_csv_path = os.path.join(root_path, f"timing_{job_id}.csv")  #! this puts it in the inf_XX
        with open(timing_csv_path, "w") as f:
            f.write("job_id,total_time,nsamples\n")
            f.write(f"{job_id},{end_time - start_time:.2f},{len(pdb_paths)}\n")
        logger.info(f"Timing information saved to: {timing_csv_path}")


if __name__ == "__main__":
    main()
