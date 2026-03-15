"""
Shared utilities for evaluation module.
"""

import os
import shutil
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from loguru import logger
from omegaconf import DictConfig, OmegaConf

# =============================================================================
# Progress Utilities
# =============================================================================


def maybe_tqdm(iterable: Iterable, desc: str, show_progress: bool = False) -> Iterable:
    """
    Wrap iterable with tqdm if show_progress is True.

    This provides optional progress bars without requiring tqdm as a hard dependency
    in calling code. Import is lazy to avoid startup overhead when not used.

    Args:
        iterable: Any iterable to wrap
        desc: Description for the progress bar
        show_progress: Whether to show the progress bar (default: False)

    Returns:
        The original iterable, or a tqdm-wrapped version if show_progress=True
    """
    if show_progress:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc)
    return iterable


# =============================================================================
# Environment Variable Validation
# =============================================================================

# Required environment variables for different metric types
REQUIRED_ENV_VARS = {
    "bioinformatics": {
        "SC_EXEC": "Path to shape complementarity (sc) binary",
    },
    "tmol": {
        "TMOL_PATH": "Path to TMOL package installation",
    },
    "folding_rf3": {
        "RF3_CKPT_PATH": "Path to RF3 checkpoint file",
        "RF3_EXEC_PATH": "Path to RF3 executable",
    },
    "novelty": {
        "FOLDSEEK_EXEC": "Path to foldseek executable",
    },
}


def validate_env_vars(
    metric_types: list[str] | None = None,
    raise_on_missing: bool = False,
) -> dict[str, list[tuple[str, str]]]:
    """
    Validate that required environment variables are set for requested metrics.

    Args:
        metric_types: List of metric types to validate. If None, validates all.
                     Valid types: "bioinformatics", "tmol",
                                  "folding_rf3", "folding_boltz2", "novelty"
        raise_on_missing: If True, raise ValueError on missing required vars

    Returns:
        Dictionary mapping metric types to list of (var_name, description) tuples
        for any missing required environment variables.
    """
    if metric_types is None:
        metric_types = list(REQUIRED_ENV_VARS.keys())

    missing = {}

    for metric_type in metric_types:
        if metric_type not in REQUIRED_ENV_VARS:
            logger.warning(f"Unknown metric type for env validation: {metric_type}")
            continue

        required = REQUIRED_ENV_VARS[metric_type]
        metric_missing = []

        for var_name, description in required.items():
            value = os.environ.get(var_name)
            if not value:
                metric_missing.append((var_name, description))
                logger.warning(f"Environment variable {var_name} not set ({description})")

        if metric_missing:
            missing[metric_type] = metric_missing

    if raise_on_missing and missing:
        msg_parts = []
        for metric_type, vars_list in missing.items():
            var_names = ", ".join(v[0] for v in vars_list)
            msg_parts.append(f"{metric_type}: {var_names}")
        raise ValueError(f"Missing required environment variables: {'; '.join(msg_parts)}")

    return missing


def check_binary_exists(env_var: str, binary_name: str) -> bool:
    """
    Check if a binary specified by an environment variable exists and is executable.

    Args:
        env_var: Name of the environment variable containing the path
        binary_name: Human-readable name of the binary for error messages

    Returns:
        True if the binary exists and is executable, False otherwise
    """
    path = os.environ.get(env_var)
    if not path:
        logger.warning(f"{binary_name} path not set (env var: {env_var})")
        return False

    if not os.path.exists(path):
        logger.warning(f"{binary_name} binary not found at: {path}")
        return False

    if not os.access(path, os.X_OK):
        logger.warning(f"{binary_name} binary is not executable: {path}")
        return False

    return True


# =============================================================================
# Configuration Parsing
# =============================================================================


def parse_cfg_for_table(cfg: DictConfig) -> tuple[list[str], dict]:
    """
    Flatten config and uses it to initialize results dataframes columns.

    Returns:
        2-tuple, with the columns (list of strings) and the flattened dictionary.
    """

    # Prefixes within the broad "generation_" ignore that should be kept.
    # These are the model sampling parameters users may sweep over.
    keep_generation_prefixes = [
        "generation_args_",  # diffusion args: nsteps, guidance_w, self_cond, ...
        "generation_model_",  # model sampling: schedules, simulation_step_params, ...
        "generation_n_recycle",  # number of recycles
    ]

    def _is_kept_generation_col(col: str) -> bool:
        """Check if a generation_ column should be kept (model sampling params)."""
        return any(col.startswith(prefix) for prefix in keep_generation_prefixes)

    def keep_col(col: str) -> bool:
        flag = True
        ignore_cols = [
            "dataset",
            "metric",
            "dataset_target_dict",
            "generation_target_dict",
            "generation_",
            "sample_storage_path",
            "output_dir",
            "baseline_model",
            "aggregation_limit",
            "job_id",
            "eval_njobs",
            "input_mode",
            "evaluation_mode",
            "protein_type",
            "benchmarks",
        ]
        for s in ignore_cols:
            if s in col:
                flag = False
                break
        # Override: keep model sampling params even though they match "generation_"
        if not flag and _is_kept_generation_col(col):
            flag = True
        return flag

    flat_cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    flat_dict = pd.json_normalize(flat_cfg, sep="_").to_dict(orient="records")[0]
    flat_dict = {k: str(v) for k, v in flat_dict.items()}
    columns = list(flat_dict.keys())
    # Remove columns containing ignored patterns
    columns = [col for col in columns if keep_col(col)]

    # Add back specific columns if present
    special_cols = [
        "dataset_target_task_name",
        "generation_target_task_name",
        "generation_dataloader_dataset_task_name",
        "generation_dataloader_dataset_nrepeat_per_sample",
    ]
    for col in special_cols:
        if col in flat_dict and col not in columns:
            columns.append(col)

    # Rebuild flat_dict keyed and ordered by columns so that
    # list(flat_dict.values()) is positionally consistent with columns.
    # (Special cols are appended at the end of columns but may appear
    # earlier in the original dict — without reordering, positional
    # row construction via list(flat_dict.values()) would be misaligned.)
    flat_dict = {k: flat_dict[k] for k in columns}
    return columns, flat_dict


# =============================================================================
# File Discovery and Job Splitting
# =============================================================================


def get_pdb_files_from_dir(pdb_dir: str, ignore_postfix: str | list[str] | None = None) -> list[str]:
    """
    Get all PDB files from a directory.

    Args:
        pdb_dir: Directory containing PDB files
        ignore_postfix: Postfix(es) to ignore when finding PDB files

    Returns:
        List of paths to PDB files
    """
    pdb_dir = Path(pdb_dir)
    if ignore_postfix is not None:
        if isinstance(ignore_postfix, str):
            ignore_postfix = [ignore_postfix]
        pdb_files = [
            f for f in pdb_dir.rglob("*.pdb") if not any(f.name.endswith(postfix) for postfix in ignore_postfix)
        ]
    else:
        pdb_files = [f for f in pdb_dir.rglob("*.pdb")]

    pdb_files = sorted([str(f) for f in pdb_files])
    logger.info(f"Found {len(pdb_files)} PDB files in {pdb_dir}")
    return pdb_files


def split_pdb_files_by_job(pdb_files: list[str], job_id: int, njobs: int) -> list[str]:
    """
    Split PDB files across multiple jobs.

    Args:
        pdb_files: List of all PDB file paths
        job_id: Current job ID (0-indexed)
        njobs: Total number of jobs

    Returns:
        List of PDB file paths assigned to this job
    """
    if njobs == 1:
        return pdb_files

    # Distribute files evenly across jobs
    files_per_job = len(pdb_files) // njobs
    remaining_files = len(pdb_files) % njobs

    # Calculate start and end indices for this job
    start_idx = job_id * files_per_job + min(job_id, remaining_files)
    end_idx = start_idx + files_per_job + (1 if job_id < remaining_files else 0)

    assigned_files = pdb_files[start_idx:end_idx]
    logger.info(f"Job {job_id}/{njobs}: Processing {len(assigned_files)} files (indices {start_idx}-{end_idx - 1})")

    return assigned_files


def split_by_job_generated(
    root_path: str,
    job_id: int,
    return_root: bool = False,
) -> list[str]:
    """
    Split evaluation jobs by job id for model-generated structures.
    Selects files starting with `job_{job_id}_`, as each eval job will start
    after the corresponding generation job finishes.

    Args:
        root_path: Root path where generated samples are stored
        job_id: Job id for this evaluation job
        return_root: Whether to return root directories or full file paths (default: False)

    Returns:
        List of paths to where PDBs are stored (each PDB is at a different path).
    """
    sample_root_paths = []
    for root in os.listdir(root_path):
        if os.path.isdir(os.path.join(root_path, root)) and root.startswith(f"job_{job_id}_"):
            if return_root:
                sample_root_paths.append(os.path.join(root_path, root))
            else:
                sample_root_paths.append(os.path.join(root_path, root, f"{root}.pdb"))

    logger.info(f"Job id {job_id}: Found {len(sample_root_paths)} samples starting with `job_{job_id}_`")
    return sample_root_paths


def prepare_sample_paths(
    sample_paths: list[str],
    output_dir: str,
    input_mode: str,
) -> list[str]:
    """
    Prepares sample paths for evaluation by copying from inference to eval output directory.

    This ensures all evaluation artifacts are stored in the output_dir, making results
    self-contained and enabling proper aggregation across jobs.

    For pdb_dir mode: sample_paths are PDB file paths, creates directories with copied files.
    For generated mode: sample_paths are directories, copies entire directory structure.

    Args:
        sample_paths: List of PDB file paths (pdb_dir mode) or directory paths (generated mode)
        output_dir: Output directory for evaluation results
        input_mode: Either "pdb_dir" or "generated"

    Returns:
        List of directory paths in output_dir containing copied PDB files
    """
    formatted_paths = []

    if input_mode == "pdb_dir":
        # sample_paths are PDB file paths
        for pdb_path in sample_paths:
            pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]
            tmp_dir = os.path.join(output_dir, f"tmp_{pdb_name}")
            os.makedirs(tmp_dir, exist_ok=True)
            formatted_path = os.path.join(tmp_dir, f"tmp_{pdb_name}.pdb")
            shutil.copy2(pdb_path, formatted_path)
            formatted_paths.append(tmp_dir)
    else:
        # For generated mode, sample_paths are directories - copy to output_dir
        for dir_path in sample_paths:
            sample_name = os.path.basename(dir_path)
            dest_dir = os.path.join(output_dir, sample_name)

            # Copy if destination doesn't exist or source is newer
            if not os.path.exists(dest_dir):
                shutil.copytree(dir_path, dest_dir)
            else:
                # Destination exists - copy any new/updated files
                for item in os.listdir(dir_path):
                    src_item = os.path.join(dir_path, item)
                    dst_item = os.path.join(dest_dir, item)
                    if os.path.isfile(src_item):
                        # Only copy if source is newer or dest doesn't exist
                        if not os.path.exists(dst_item) or os.path.getmtime(src_item) > os.path.getmtime(dst_item):
                            shutil.copy2(src_item, dst_item)

            formatted_paths.append(dest_dir)

    logger.info(f"Prepared {len(formatted_paths)} sample paths in {output_dir}")
    return formatted_paths


# =============================================================================
# Timing Utilities
# =============================================================================


def read_and_update_timing_csv(timing_csv_path: str, job_id: int, evaluation_time: float, nsamples: int) -> None:
    """
    Read existing timing CSV from generation, update with evaluation time, and save back.
    """
    try:
        if os.path.exists(timing_csv_path):
            timing_df = pd.read_csv(timing_csv_path)
            job_mask = timing_df["job_id"] == job_id
            if job_mask.any():
                if "total_time" in timing_df.columns and "generation_time" not in timing_df.columns:
                    timing_df = timing_df.rename(columns={"total_time": "generation_time"})
                    logger.info("Renamed 'total_time' column to 'generation_time' (old format detected)")

                if "generation_time" in timing_df.columns:
                    generation_time = timing_df.loc[job_mask, "generation_time"].iloc[0]
                else:
                    generation_time = 0.0
                    timing_df.loc[job_mask, "generation_time"] = generation_time

                timing_df.loc[job_mask, "evaluation_time"] = evaluation_time
                timing_df.loc[job_mask, "total_time"] = generation_time + evaluation_time
                timing_df.loc[job_mask, "nsamples"] = nsamples

                timing_df.to_csv(timing_csv_path, index=False)
                logger.info(
                    f"Updated timing information: generation_time={generation_time:.2f}s, "
                    f"evaluation_time={evaluation_time:.2f}s, total_time={generation_time + evaluation_time:.2f}s"
                )
            else:
                logger.warning(f"No existing timing data found for job_id {job_id}")
        else:
            logger.info(f"Creating new timing CSV at {timing_csv_path}")
            timing_df = pd.DataFrame(
                {
                    "job_id": [job_id],
                    "generation_time": [0.0],
                    "evaluation_time": [evaluation_time],
                    "total_time": [evaluation_time],
                    "nsamples": [nsamples],
                }
            )
            timing_df.to_csv(timing_csv_path, index=False)
            logger.info(f"Created timing CSV with evaluation_time={evaluation_time:.2f}s")
    except Exception as e:
        logger.error(f"Failed to update timing CSV: {e}")
