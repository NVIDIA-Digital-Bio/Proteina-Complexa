#!/usr/bin/env python3
"""
Complexa CLI - Binder Design Pipeline

Run individual pipeline steps or the full design pipeline locally on a single GPU.
All output goes to ./inference and ./evaluation_results automatically.
Logs are saved to ./logs/ with timestamps.

Commands:
    complexa init       Initialize environment (.env + env.sh for a runtime)
    complexa demo       Show usage examples and explanations
    complexa validate   Validate configuration and check prerequisites
    complexa design     Run the full binder design pipeline (generate → filter → evaluate → analyze)
    complexa analysis   Run evaluate → analyze pipeline (for evaluating PDB files)
    complexa generate   Generate binder structures using the partially latent flow matching model
    complexa filter     Filter and rank generated samples by reward scores
    complexa evaluate   Evaluate samples with refolding metrics
    complexa analyze    Aggregate and analyze results
    complexa status     Check pipeline status
    complexa download   Download model weights (interactive wizard)
    complexa target     Manage target configurations (list, add, show)

Examples:
    # Run the full design pipeline
    complexa design configs/search_binder_local.yaml

    # Run individual steps
    complexa generate configs/search_binder_local.yaml
    complexa filter configs/search_binder_local.yaml
    complexa evaluate configs/search_binder_local.yaml
    complexa analyze configs/search_binder_local.yaml

    # With config overrides (Hydra-style) - applied to ALL steps in design
    complexa design configs/search_binder_local.yaml ++run_name=exp1 ++generation.task_name=02_PDL1
    complexa generate configs/search_binder_local.yaml ++generation.task_name=02_PDL1 ++seed=42

    # Check status
    complexa status

    # Download model weights
    complexa download

    # Show full output (no log file)
    complexa design configs/search_binder_local.yaml --verbose
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

# Lazy imports - these are loaded only when needed to speed up CLI startup
# The actual imports happen in the functions that use them


def _import_progress_bar():
    """Lazy import for progress bar utilities."""
    from proteinfoundation.cli.progress_bar import clear_pipeline_step, set_pipeline_step

    return clear_pipeline_step, set_pipeline_step


def _import_target_manager():
    """Lazy import for target manager utilities."""
    from proteinfoundation.cli.target_manager import add_target_cli, add_target_interactive, list_targets, show_target

    return add_target_cli, add_target_interactive, list_targets, show_target


def _import_validate():
    """Lazy import for validation utilities."""
    from proteinfoundation.cli.validate import run_validation

    return run_validation


def _quiet_startup():
    """Lazy quiet startup - only called when running pipeline steps."""
    try:
        from proteinfoundation.cli.startup import quiet_startup

        quiet_startup()
    except ImportError:
        pass


# =============================================================================
# Constants
# =============================================================================

STEP_MODULES = {
    "generate": "proteinfoundation.generate",
    "filter": "proteinfoundation.filter",
    "evaluate": "proteinfoundation.evaluate",
    "analyze": "proteinfoundation.analyze",
}

STEP_DESCRIPTIONS = {
    "generate": "Generate binder structures using the partially latent flow matching model",
    "filter": "Filter and rank generated samples by reward scores",
    "evaluate": "Evaluate samples with refolding and interface metrics",
    "analyze": "Aggregate and analyze evaluation results",
}

STEP_REQUIRES_GPU = {
    "generate": True,
    "filter": False,
    "evaluate": True,
    "analyze": False,
}

DESIGN_PIPELINE_STEPS = ["generate", "filter", "evaluate", "analyze"]
ANALYSIS_PIPELINE_STEPS = ["evaluate", "analyze"]

# Default output directories (created automatically by the pipeline steps)
DEFAULT_INFERENCE_DIR = Path("./inference")
DEFAULT_EVAL_DIR = Path("./evaluation_results")
LOG_DIR = Path("./logs")


# =============================================================================
# Logging Setup
# =============================================================================


def _get_timestamp() -> str:
    """Get a formatted timestamp for log files."""
    now = datetime.now()
    return f"Y{now.year}_M{now.month:02d}_D{now.day:02d}_H{now.hour:02d}_M{now.minute:02d}_S{now.second:02d}"


def create_log_file(
    step_name: str,
    run_name: str | None = None,
    target_name: str | None = None,
) -> Path:
    """Create a timestamped log file for a step.

    Parameters
    ----------
    step_name : str
        Name of the step (e.g., "generate", "filter")
    run_name : str, optional
        Optional run name to include in the log filename
    target_name : str, optional
        Optional target name to include in the log filename

    Returns
    -------
    Path
        Path to the log file
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = _get_timestamp()

    # Build filename parts: step_target_run_timestamp.log
    parts = [step_name]
    if target_name:
        parts.append(target_name)
    if run_name:
        parts.append(run_name)
    parts.append(timestamp)

    log_file = LOG_DIR / f"{'_'.join(parts)}.log"
    return log_file


def create_pipeline_log_file(
    run_name: str | None = None,
    target_name: str | None = None,
    pipeline_name: str = "design",
) -> Path:
    """Create a timestamped log file for a pipeline.

    Parameters
    ----------
    run_name : str, optional
        Optional run name to include in the log filename
    target_name : str, optional
        Optional target name to include in the log filename
    pipeline_name : str
        Name of the pipeline (default: "design")

    Returns
    -------
    Path
        Path to the log file
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = _get_timestamp()

    # Build filename parts: {pipeline}_pipeline_target_run_timestamp.log
    parts = [f"{pipeline_name}_pipeline"]
    if target_name:
        parts.append(target_name)
    if run_name:
        parts.append(run_name)
    parts.append(timestamp)

    log_file = LOG_DIR / f"{'_'.join(parts)}.log"
    return log_file


# =============================================================================
# Pipeline Configuration
# =============================================================================


@dataclass
class StepConfig:
    """Configuration for running a pipeline step."""

    config_file: Path
    config_name: str

    # Job control
    job_id: int = 0

    # Logging
    log_file: Path | None = None
    verbose: bool = False  # If True, show all output; if False, log to file

    # Optional run name (postfix for output dirs and logs)
    run_name: str | None = None

    # Target name for generation (extracted from config or overrides)
    target_name: str | None = None

    # Extra config overrides to pass to Hydra (e.g., ["++key=value", ...])
    config_overrides: list[str] | None = None
    pipeline_log_dir: Path | None = None

    # Environment variables to inject into subprocess (from pipeline YAML env_vars key)
    env_vars: dict[str, str] | None = None

    @classmethod
    def from_yaml(cls, config_path: Path, overrides: list[str] | None = None) -> StepConfig:
        """Load config from a YAML file.

        Parameters
        ----------
        config_path : Path
            Path to the config file
        overrides : list[str], optional
            Hydra-style overrides (e.g., ["++run_name=exp1", "++seed=42"])
        """
        config = cls(
            config_file=config_path,
            config_name=config_path.stem,
            config_overrides=overrides,
        )

        # Extract run_name from overrides first (takes priority)
        if overrides:
            for override in overrides:
                if override.startswith("++run_name="):
                    config.run_name = override.split("=", 1)[1]
                    break

        # If not in overrides, try to read from config file
        if config.run_name is None:
            try:
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                config.run_name = cfg.get("run_name")
            except Exception:
                pass

        # Extract env_vars from config file
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            raw_env = cfg.get("env_vars")
            if isinstance(raw_env, dict):
                config.env_vars = {str(k): str(v) for k, v in raw_env.items()}
        except Exception:
            pass

        # Extract target_name (task_name) from overrides first (takes priority)
        config.target_name = _get_task_name_from_overrides(overrides)

        # If not in overrides, try to read from config file
        if config.target_name is None:
            try:
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)

                # Check generation.task_name first
                if "generation" in cfg and isinstance(cfg["generation"], dict):
                    gen_cfg = cfg["generation"]
                    task_name = gen_cfg.get("task_name")
                    if task_name and not str(task_name).startswith("${"):
                        config.target_name = task_name
                    elif "dataloader" in gen_cfg:
                        config.target_name = _get_task_name_from_dataloader(gen_cfg["dataloader"])

                # If not found, try loading the generation config defaults
                if config.target_name is None and "defaults" in cfg:
                    for default in cfg["defaults"]:
                        if isinstance(default, dict) and "generation" in default:
                            gen_file = config_path.parent / "generation" / f"{default['generation']}.yaml"
                            if gen_file.exists():
                                with open(gen_file) as f:
                                    gen_cfg = yaml.safe_load(f)
                                # Check task_name at top level first
                                task_name = gen_cfg.get("task_name")
                                if task_name and not str(task_name).startswith("${"):
                                    config.target_name = task_name
                                elif "dataloader" in gen_cfg:
                                    config.target_name = _get_task_name_from_dataloader(gen_cfg["dataloader"])
                                break
            except Exception:
                pass

        return config


# =============================================================================
# Step Execution
# =============================================================================


def _get_task_name_from_dataloader(dl: dict) -> str | None:
    """Extract task_name from a dataloader config dict.

    Checks dataset.task_name first (new format), then falls back to
    conditional_features[0].task_name (old format).
    """
    if "dataset" not in dl:
        return None

    dataset = dl["dataset"]

    # New format: task_name at dataset level
    task_name = dataset.get("task_name")
    if task_name and not task_name.startswith("${"):
        return task_name

    # Old format: task_name in conditional_features
    if "conditional_features" in dataset:
        features = dataset["conditional_features"]
        if features and len(features) > 0:
            task_name = features[0].get("task_name")
            if task_name and not task_name.startswith("${"):
                return task_name

    return None


def _get_task_name_from_overrides(overrides: list[str] | None) -> str | None:
    """Extract task_name from CLI overrides.

    Checks for ++generation.task_name=X or ++generation.dataloader.dataset.task_name=X
    """
    if not overrides:
        return None

    for override in overrides:
        # Check both possible override paths
        if override.startswith("++generation.task_name="):
            return override.split("=", 1)[1]
        if override.startswith("++generation.dataloader.dataset.task_name="):
            return override.split("=", 1)[1]

    return None


def _get_stage_njobs_key(step_name: str) -> str | None:
    if step_name in {"generate", "filter"}:
        return "gen_njobs"
    if step_name in {"evaluate", "analyze"}:
        return "eval_njobs"
    return None


def _get_njobs_from_overrides(overrides: list[str] | None, key: str) -> int | None:
    if not overrides:
        return None
    for override in overrides:
        if override.startswith(f"++{key}="):
            try:
                return int(override.split("=", 1)[1])
            except ValueError:
                return None
    return None


def _get_stage_njobs(config: StepConfig, step_name: str) -> int:
    key = _get_stage_njobs_key(step_name)
    if key is None:
        return 1

    override_value = _get_njobs_from_overrides(config.config_overrides, key)
    if override_value is not None and override_value > 0:
        return override_value

    try:
        with open(config.config_file) as f:
            cfg = yaml.safe_load(f) or {}
        value = cfg.get(key)
        return int(value) if isinstance(value, int) and value > 0 else 1
    except Exception:
        return 1


def _build_log_header(step_name: str, cmd: list[str], job_id: int | None = None) -> str:
    label = f"{step_name}"
    if job_id is not None:
        label = f"{step_name} (job {job_id})"
    return f"\n{'=' * 60}\nStep: {label}\nTime: {datetime.now().isoformat()}\nCommand: {' '.join(cmd)}\n{'=' * 60}\n\n"


def _stream_process_output(
    proc: subprocess.Popen,
    log_path: Path,
    header: str,
    should_filter_line,
) -> None:
    skip_next_empty = False
    with open(log_path, "a") as log_f:
        log_f.write(header)
        log_f.flush()
        for line in proc.stdout:
            if should_filter_line(line):
                skip_next_empty = True
                continue
            if skip_next_empty and line.strip() == "":
                skip_next_empty = False
                continue
            skip_next_empty = False
            log_f.write(line)
            log_f.flush()


def get_generate_summary(config: StepConfig, run_name: str | None = None) -> dict:
    """Get summary info for the generate step.

    Returns
    -------
    dict
        Summary with target_name, n_pdbs, and output_dir
    """
    summary = {
        "target_name": None,
        "n_pdbs": 0,
        "output_dir": None,
    }

    try:
        # First check CLI overrides (highest priority)
        override_task_name = _get_task_name_from_overrides(config.config_overrides)
        if override_task_name:
            summary["target_name"] = override_task_name
        else:
            # Load the config to get target info
            with open(config.config_file) as f:
                cfg = yaml.safe_load(f)

            # Try to get target name from the config
            # Check generation.task_name first (top-level), then dataloader
            if "generation" in cfg and isinstance(cfg["generation"], dict):
                gen_cfg = cfg["generation"]

                # Check for task_name at generation level (new simplified format)
                task_name = gen_cfg.get("task_name")
                if task_name and not str(task_name).startswith("${"):
                    summary["target_name"] = task_name
                # Fall back to dataloader.dataset.task_name
                elif "dataloader" in gen_cfg:
                    summary["target_name"] = _get_task_name_from_dataloader(gen_cfg["dataloader"])

            # If not found, try loading the generation config defaults
            if summary["target_name"] is None and "defaults" in cfg:
                for default in cfg["defaults"]:
                    if isinstance(default, dict) and "generation" in default:
                        gen_file = config.config_file.parent / "generation" / f"{default['generation']}.yaml"
                        if gen_file.exists():
                            with open(gen_file) as f:
                                gen_cfg = yaml.safe_load(f)
                            if "dataloader" in gen_cfg:
                                summary["target_name"] = _get_task_name_from_dataloader(gen_cfg["dataloader"])
                            break

        # Find the output directory
        # Format: ./inference/{config_name}_{task_name}_{timestamp}[_{run_name}]/
        # e.g. ./inference/search_binder_local_02_PDL1_Y2025_M12_D13_H19_M53_S43/
        # or   ./inference/search_binder_local_02_PDL1_Y2025_M12_D13_H19_M53_S43_experiment1/
        if DEFAULT_INFERENCE_DIR.exists():
            config_name = config.config_name
            task_name = summary["target_name"]
            base_name = config_name
            if task_name:
                base_name = f"{base_name}_{task_name}"
            if run_name:
                base_name = f"{base_name}_{run_name}"

            matching_dirs = []
            direct_path = DEFAULT_INFERENCE_DIR / base_name
            if direct_path.exists():
                matching_dirs.append(direct_path)
            else:
                for run_dir in DEFAULT_INFERENCE_DIR.iterdir():
                    if not run_dir.is_dir():
                        continue
                    dir_name = run_dir.name
                    if dir_name.startswith(config_name):
                        if task_name is None or task_name in dir_name:
                            if run_name:
                                if dir_name.endswith(f"_{run_name}"):
                                    matching_dirs.insert(0, run_dir)
                                else:
                                    matching_dirs.append(run_dir)
                            else:
                                matching_dirs.append(run_dir)

            if matching_dirs:
                matching_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                output_dir = matching_dirs[0]
                summary["output_dir"] = output_dir
                # Count PDBs, excluding _binder.pdb files (intermediate outputs)
                summary["n_pdbs"] = len([p for p in output_dir.rglob("*.pdb") if not p.name.endswith("_binder.pdb")])
    except Exception:
        pass  # Silently handle any config parsing errors

    return summary


def print_generate_summary(config: StepConfig) -> None:
    """Print summary after generate step completes.

    This function is designed to never raise exceptions - any errors
    are caught and displayed cleanly without killing the process.
    """
    try:
        summary = get_generate_summary(config, run_name=config.run_name)

        print(f"\n  {'─' * 56}")
        print("  📊 Generation Summary:")

        if summary["target_name"]:
            print(f"     Target:     {summary['target_name']}")

        if summary["n_pdbs"] > 0:
            print(f"     PDB files:  {summary['n_pdbs']} generated")
            if summary["output_dir"]:
                print(f"     Output:     {summary['output_dir']}")
        else:
            print("     PDB files:  (check logs for output location)")

        print(f"  {'─' * 56}")
    except Exception as e:
        # Never let summary errors kill the process
        print(f"\n  {'─' * 56}")
        print("  📊 Generation Summary:")
        print(f"     (Could not parse summary: {e})")
        print("     Check ./inference/ for output files")
        print(f"  {'─' * 56}")


def run_step(
    step_name: str,
    config: StepConfig,
    current: int | None = None,
    total: int | None = None,
) -> None:
    """Run a single pipeline step.

    Parameters
    ----------
    step_name : str
        Name of the step to run (generate, filter, evaluate, analyze)
    config : StepConfig
        Configuration for the step
    current : int, optional
        Current step number (for progress display)
    total : int, optional
        Total number of steps (for progress display)
    """
    # Lazy import progress bar and quiet startup (only when running steps)
    clear_pipeline_step, set_pipeline_step = _import_progress_bar()
    _quiet_startup()

    module = STEP_MODULES[step_name]
    description = STEP_DESCRIPTIONS[step_name]
    gpu_icon = "🔥💻" if STEP_REQUIRES_GPU[step_name] else "💻"

    # Print step header
    print(f"\n{'─' * 60}")
    if current and total:
        print(f"  {gpu_icon} Step {current}/{total}: {step_name}")
        set_pipeline_step(step_name, current=current, total=total)
        os.environ["PIPELINE_STEP"] = step_name
        os.environ["PIPELINE_PROGRESS"] = f"{current}/{total}"
    else:
        print(f"  {gpu_icon} Running: {step_name}")
        set_pipeline_step(step_name)
        os.environ["PIPELINE_STEP"] = step_name
    print(f"  {description}")

    stage_njobs = _get_stage_njobs(config, step_name)
    use_parallel_execution = stage_njobs > 1 and not config.verbose and step_name in {"generate", "evaluate"}
    design_parallel = use_parallel_execution and config.pipeline_log_dir is not None
    stage_log_dir = config.pipeline_log_dir or LOG_DIR

    # Ensure we have a log file when not using design-level stage logs
    if config.log_file is None and not config.verbose:
        config.log_file = create_log_file(step_name, run_name=config.run_name, target_name=config.target_name)
    base_log_path = config.log_file

    # Show log file location if logging
    if not config.verbose:
        if design_parallel:
            # Multi-job steps: logs go to pipeline_log_dir/{step}_job{N}.log
            print(f"  📝 Stage logs: {config.pipeline_log_dir}")
        elif config.pipeline_log_dir is not None:
            # Single-job step within a pipeline: log goes to pipeline_log_dir/{step}.log
            print(f"  📝 Log: {config.pipeline_log_dir / f'{step_name}.log'}")
        elif config.log_file:
            print(f"  📝 Log: {config.log_file}")

    print(f"{'─' * 60}")
    # Build command with aggressive warning suppression
    cmd = [
        sys.executable,
        "-W",
        "ignore",  # Suppress ALL warnings
        "-W",
        "ignore::DeprecationWarning",  # Extra: suppress deprecation
        "-W",
        "ignore::FutureWarning",  # Extra: suppress future warnings
        "-W",
        "ignore::UserWarning",  # Extra: suppress user warnings
        "-m",
        module,
        "--config-path",
        str(config.config_file.parent.absolute()),
        "--config-name",
        config.config_file.stem,
        f"++job_id={config.job_id}",
        f"++base_config_name={config.config_name}",
    ]

    # Add any extra config overrides (but avoid duplicating run_name)
    if config.config_overrides:
        for override in config.config_overrides:
            # Skip run_name if we'll add it explicitly
            if override.startswith("++run_name="):
                continue
            cmd.append(override)

    # Add run_name if provided (after filtering from overrides to avoid duplication)
    if config.run_name:
        cmd.append(f"++run_name={config.run_name}")

    if step_name == "generate":
        cmd.append(f"++gen_njobs={stage_njobs}")
    elif step_name == "evaluate":
        cmd.append(f"++eval_njobs={stage_njobs}")

    # Run step
    step_start = time.time()

    # Set up environment to suppress ALL warnings
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"

    # Inject pipeline-level env_vars from config
    if config.env_vars:
        env.update(config.env_vars)

    # Patterns to filter from log output (noisy warnings we don't care about)
    FILTER_PATTERNS = [
        "DeprecationWarning:",
        "FutureWarning:",
        "jax.tree_map is deprecated",
        "jax.tree_flatten is deprecated",
        "jax.tree_unflatten is deprecated",
        "optax.dpsgd is deprecated",
        "optax.noisy_sgd",
        "optax.dpsgd",
        "backend and device argument on jit is deprecated",
        "jax.numpy.clip is deprecated",
        "Passing arguments 'a', 'a_min', or 'a_max'",
        "builtin type Swig",
        "builtin type swig",
        "swigvarlink",
        "has no __module__ attribute",
        "[jax",
        "[optax",
        "jax.tree_map(",
        "jax.tree_flatten(",
        "jax.tree_unflatten(",
    ]

    def should_filter_line(line: str) -> bool:
        """Check if a line should be filtered from the log."""
        return any(pattern in line for pattern in FILTER_PATTERNS)

    try:
        if use_parallel_execution:
            if design_parallel:
                stage_log_dir = config.pipeline_log_dir or (config.log_file.parent if config.log_file else LOG_DIR)
                stage_log_dir.mkdir(parents=True, exist_ok=True)
            procs = []
            threads = []
            for job_id in range(stage_njobs):
                job_cmd = cmd + [f"++job_id={job_id}"]
                job_env = env.copy()
                job_env["CUDA_VISIBLE_DEVICES"] = str(job_id)
                if design_parallel:
                    job_log = stage_log_dir / f"{step_name}_job{job_id}.log"
                    if job_log.exists():
                        job_log.unlink()
                else:
                    stem = base_log_path.stem if base_log_path else step_name
                    suffix = base_log_path.suffix if base_log_path else ".log"
                    job_log = base_log_path.with_name(f"{stem}_job{job_id}{suffix}")
                proc = subprocess.Popen(
                    job_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=job_env,
                    bufsize=1,
                )
                header = _build_log_header(step_name, job_cmd, job_id=job_id)
                thread = threading.Thread(
                    target=_stream_process_output,
                    args=(proc, job_log, header, should_filter_line),
                    daemon=True,
                )
                thread.start()
                procs.append(proc)
                threads.append(thread)

            for proc, thread in zip(procs, threads, strict=False):
                proc.wait()
                thread.join()
                if proc.returncode != 0:
                    raise subprocess.CalledProcessError(proc.returncode, cmd)
        else:
            if config.verbose:
                print(f"\n  Command: {' '.join(cmd)}\n")
                subprocess.run(cmd, check=True, env=env)
            else:
                log_path = config.log_file
                if log_path is None:
                    log_path = create_log_file(
                        step_name,
                        run_name=config.run_name,
                        target_name=config.target_name,
                    )
                    config.log_file = log_path
                if config.pipeline_log_dir is not None:
                    log_path = config.pipeline_log_dir / f"{step_name}.log"
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                header = _build_log_header(step_name, cmd, job_id=config.job_id)
                with open(log_path, "a") as log_f:
                    log_f.write(header)
                    log_f.flush()
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=env,
                        bufsize=1,
                    )
                    skip_next_empty = False
                    for line in process.stdout:
                        if should_filter_line(line):
                            skip_next_empty = True
                            continue
                        if skip_next_empty and line.strip() == "":
                            skip_next_empty = False
                            continue
                        skip_next_empty = False
                        log_f.write(line)
                        log_f.flush()
                    process.wait()
                    if process.returncode != 0:
                        raise subprocess.CalledProcessError(process.returncode, cmd)
    except subprocess.CalledProcessError as e:
        print(f"\n  ✗ {step_name} failed with exit code {e.returncode}")
        if config.log_file:
            print(f"  📝 Check log for details: {config.log_file}")
        raise

    elapsed = time.time() - step_start
    print(f"  ✓ {step_name} completed in {elapsed:.1f}s")

    # Print summary for generate step
    if step_name == "generate":
        print_generate_summary(config)


def run_design_pipeline(
    config: StepConfig,
    steps: list[str] | None = None,
    verbose: bool = False,
    pipeline_name: str = "design",
) -> None:
    """Run a multi-step pipeline.

    Parameters
    ----------
    config : StepConfig
        Configuration for all pipeline steps
    steps : list[str], optional
        Subset of steps to run (default: all steps for design pipeline)
    verbose : bool
        If True, show all output; if False, log to file
    pipeline_name : str
        Name of the pipeline for logging/display (default: "design")
    """
    steps_to_run = steps or DESIGN_PIPELINE_STEPS
    total_steps = len(steps_to_run)

    # Create log file for the pipeline (include target_name and run_name if provided)
    log_file = None
    if not verbose:
        log_file = create_pipeline_log_file(
            run_name=config.run_name,
            target_name=config.target_name,
            pipeline_name=pipeline_name,
        )
        config.log_file = log_file

    config.verbose = verbose
    gen_njobs = _get_stage_njobs(config, "generate")
    eval_njobs = _get_stage_njobs(config, "evaluate")
    pipeline_dir = None
    if log_file:
        pipeline_dir = log_file.with_suffix("")
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        config.pipeline_log_dir = pipeline_dir

    # Pipeline-specific display settings
    if pipeline_name == "analysis":
        banner_title = "🔬 Complexa Analysis Pipeline"
        completion_msg = "Analysis pipeline"
    else:
        banner_title = "🧬 Complexa Binder Design Pipeline"
        completion_msg = "Design pipeline"

    # Print banner
    print(f"\n{'=' * 60}")
    print(f"  {banner_title}")
    print(f"{'=' * 60}")
    print(f"  Config:       {config.config_file}")
    if config.target_name:
        print(f"  Target:       {config.target_name}")
    if config.run_name:
        print(f"  Run:          {config.run_name}")

    # Show GPU info
    max_jobs = max(gen_njobs, eval_njobs)
    if max_jobs > 1:
        print(f"  GPUs:         {max_jobs}")
    else:
        print("  GPUs:         1")
    print(f"  Steps:        {', '.join(steps_to_run)}")
    if log_file:
        if config.pipeline_log_dir:
            print(f"  🗂️  Stage logs: {config.pipeline_log_dir}")
        else:
            print(f"  📝 Log:        {log_file}")
    print(f"{'=' * 60}")

    start_time = time.time()

    # Run each step with the same config
    for idx, step_name in enumerate(steps_to_run, start=1):
        run_step(step_name, config, current=idx, total=total_steps)

    clear_pipeline_step, _ = _import_progress_bar()
    clear_pipeline_step()
    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print(f"  ✓ {completion_msg} completed in {elapsed / 60:.1f} minutes")
    if config.pipeline_log_dir:
        print(f"  📝 Logs: {config.pipeline_log_dir}/")
    elif log_file:
        print(f"  📝 Full log: {log_file}")
    print(f"{'=' * 60}\n")


# =============================================================================
# CLI Argument Parsing
# =============================================================================


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to a parser."""
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the configuration YAML file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show full output in terminal (don't log to file)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra config overrides (e.g., ++run_name=exp1 ++seed=42)",
    )


def add_job_args(parser: argparse.ArgumentParser) -> None:
    """Add job-related arguments to a parser."""
    parser.add_argument(
        "--job-id",
        type=int,
        default=0,
        help="Job ID for parallel execution (default: 0)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="complexa",
        description="Complexa Binder Design Pipeline - Single GPU execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # -------------------------------------------------------------------------
    # design - Run full pipeline
    # -------------------------------------------------------------------------
    design_parser = subparsers.add_parser(
        "design",
        help="Run the full binder design pipeline",
        description="Execute generate → filter → evaluate → analyze sequentially",
    )
    design_parser.add_argument(
        "config",
        type=Path,
        help="Path to the config YAML file (used for all steps)",
    )
    design_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show full output in terminal (don't log to file)",
    )
    design_parser.add_argument(
        "--steps",
        nargs="+",
        choices=DESIGN_PIPELINE_STEPS,
        help="Run only specific steps (default: all)",
    )
    design_parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra config overrides applied to ALL steps (e.g., ++run_name=exp1)",
    )

    # -------------------------------------------------------------------------
    # generate - Generate structures
    # -------------------------------------------------------------------------
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate binder structures",
        description="Generate binder structures using the partially latent flow matching model",
    )
    add_common_args(generate_parser)
    add_job_args(generate_parser)

    # -------------------------------------------------------------------------
    # filter - Filter samples
    # -------------------------------------------------------------------------
    filter_parser = subparsers.add_parser(
        "filter",
        help="Filter generated samples",
        description="Filter and rank generated samples by reward scores",
    )
    add_common_args(filter_parser)

    # -------------------------------------------------------------------------
    # evaluate - Evaluate samples
    # -------------------------------------------------------------------------
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate samples",
        description="Evaluate samples with refolding metrics",
    )
    add_common_args(evaluate_parser)
    add_job_args(evaluate_parser)

    # -------------------------------------------------------------------------
    # analyze - Analyze results
    # -------------------------------------------------------------------------
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze results",
        description="Aggregate and analyze pipeline results",
    )
    add_common_args(analyze_parser)

    # -------------------------------------------------------------------------
    # analysis - Run evaluate + analyze together
    # -------------------------------------------------------------------------
    analysis_parser = subparsers.add_parser(
        "analysis",
        help="Run evaluate → analyze pipeline",
        description="Execute evaluate → analyze sequentially (for evaluating PDB files)",
    )
    analysis_parser.add_argument(
        "config",
        type=Path,
        help="Path to the config YAML file (used for both steps)",
    )
    analysis_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show full output in terminal (don't log to file)",
    )
    analysis_parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra config overrides applied to both steps (e.g., ++dataset.task_name=02_PDL1)",
    )

    # -------------------------------------------------------------------------
    # status - Check status
    # -------------------------------------------------------------------------
    status_parser = subparsers.add_parser(
        "status",
        help="Check pipeline status",
        description="Check the status of pipeline outputs",
    )
    status_parser.add_argument(
        "--logs",
        action="store_true",
        help="Show recent log files",
    )

    # -------------------------------------------------------------------------
    # download - Download model weights
    # -------------------------------------------------------------------------
    download_parser = subparsers.add_parser(
        "download",
        help="Download model weights",
        description=(
            "Download Complexa and community model weights. "
            "Without arguments, launches an interactive wizard. "
            "Use flags to download specific models non-interactively."
        ),
    )
    download_parser.add_argument(
        "--complexa",
        action="store_true",
        help="Download Complexa Protein weights (protein binder design)",
    )
    download_parser.add_argument(
        "--complexa-ligand",
        action="store_true",
        help="Download Complexa Ligand weights (ligand binder design)",
    )
    download_parser.add_argument(
        "--complexa-ame",
        action="store_true",
        help="Download Complexa AME weights (enzyme/motif scaffolding)",
    )
    download_parser.add_argument(
        "--complexa-all",
        action="store_true",
        help="Download all Complexa weights (protein + ligand + AME)",
    )
    download_parser.add_argument(
        "--all",
        action="store_true",
        help="Download all community model weights (ProteinMPNN, LigandMPNN, AF2, ESM2, RF3)",
    )
    download_parser.add_argument(
        "--everything",
        action="store_true",
        help="Download all models (Complexa + community + optional)",
    )
    download_parser.add_argument(
        "--status",
        action="store_true",
        help="Show installation status of all models",
    )

    # -------------------------------------------------------------------------
    # demo - Show usage examples
    # -------------------------------------------------------------------------
    subparsers.add_parser(
        "demo",
        help="Show usage examples and explanations",
        description="Display example commands and explain what each step does",
    )

    # -------------------------------------------------------------------------
    # init - Initialize environment configuration
    # -------------------------------------------------------------------------
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize environment configuration",
        description=(
            "Phase 1: Create .env from .env_example (if .env is missing).\n"
            "Phase 2: Generate env.sh for a specific runtime (if .env exists).\n\n"
            "Usage:\n"
            "  complexa init                  # Create .env (first time)\n"
            "  complexa init uv               # Generate env.sh for UV runtime\n"
            "  complexa init docker           # Generate env.sh for Docker runtime\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    init_parser.add_argument(
        "runtime",
        nargs="?",
        choices=["uv", "docker"],
        default=None,
        help="Runtime to configure: uv or docker",
    )
    init_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite existing .env or env.sh file",
    )

    # -------------------------------------------------------------------------
    # target - Manage target configurations
    # -------------------------------------------------------------------------
    target_parser = subparsers.add_parser(
        "target",
        help="Manage target configurations",
        description="List, add, or show target configurations for binder design",
    )
    target_subparsers = target_parser.add_subparsers(dest="target_command")

    # target list
    target_list_parser = target_subparsers.add_parser(
        "list",
        help="List available targets",
        description="List all targets in the targets dictionary",
    )
    target_list_parser.add_argument(
        "--dict",
        type=Path,
        help="Path to custom targets dict YAML file",
    )
    target_list_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed information for each target",
    )
    target_list_parser.add_argument(
        "--ligand",
        action="store_true",
        help="Show only ligand targets",
    )
    target_list_parser.add_argument(
        "--protein",
        action="store_true",
        help="Show only protein targets",
    )

    # target add
    target_add_parser = target_subparsers.add_parser(
        "add",
        help="Add a new target",
        description="Add a new target configuration",
    )
    target_add_parser.add_argument(
        "name",
        nargs="?",
        help="Target name (required for non-interactive mode)",
    )
    target_add_parser.add_argument(
        "--dict",
        type=Path,
        help="Path to custom targets dict YAML file",
    )
    target_add_parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Open editor for interactive configuration",
    )
    target_add_parser.add_argument(
        "-e",
        "--editor",
        type=str,
        help="Editor to use for interactive mode (e.g., 'code', 'nano', 'vim', 'subl')",
    )
    target_add_parser.add_argument(
        "--source",
        type=str,
        help="Source directory under target_data/ (default: 'custom_targets')",
    )
    target_add_parser.add_argument(
        "--target-filename",
        type=str,
        help="Target PDB filename without extension (default: same as name)",
    )
    target_add_parser.add_argument(
        "--target-path",
        type=str,
        help="Full path to target PDB file (overrides source/target_filename)",
    )
    target_add_parser.add_argument(
        "--target-input",
        type=str,
        help="Chain and residue range, e.g., 'A1-115' (default: 'A1-100')",
    )
    target_add_parser.add_argument(
        "--hotspot-residues",
        nargs="+",
        help="Hotspot residues, e.g., A33 A95 A97",
    )
    target_add_parser.add_argument(
        "--binder-length",
        type=int,
        nargs="+",
        help="Binder length range [min max] or single value (default: 60 120)",
    )
    target_add_parser.add_argument(
        "--pdb-id",
        type=str,
        help="Reference PDB ID (optional)",
    )
    target_add_parser.add_argument(
        "--ligand",
        type=str,
        nargs="?",
        const="YOUR_LIGAND",
        help="Ligand residue name(s) — marks target as ligand (e.g., 'FAD', 'OQO'). Use without a value in interactive mode to get a placeholder.",
    )
    target_add_parser.add_argument(
        "--ligand-only",
        action="store_true",
        default=None,
        help="Generate binding pocket around ligand only (default for ligand targets)",
    )
    target_add_parser.add_argument(
        "--smiles",
        type=str,
        help="SMILES string for the ligand molecule",
    )
    target_add_parser.add_argument(
        "--use-bonds-from-file",
        action="store_true",
        default=None,
        help="Use bond information from PDB file (default for ligand targets)",
    )
    target_add_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite existing target without prompting",
    )

    # target show
    target_show_parser = target_subparsers.add_parser(
        "show",
        help="Show target details",
        description="Show detailed information about a specific target",
    )
    target_show_parser.add_argument(
        "name",
        help="Target name to show",
    )
    target_show_parser.add_argument(
        "--dict",
        type=Path,
        help="Path to custom targets dict YAML file",
    )

    # -------------------------------------------------------------------------
    # validate - Validate configuration and check prerequisites
    # -------------------------------------------------------------------------
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate configuration and check prerequisites",
        description="Check configuration, model weights, and file paths before running",
    )
    validate_parser.add_argument(
        "type",
        choices=["env", "target", "generate", "evaluate", "analyze", "design"],
        help="What to validate: env, target, generate, evaluate, analyze, or design (all)",
    )
    validate_parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        help="Path to config YAML file (required for generate/evaluate/analyze/design)",
    )
    validate_parser.add_argument(
        "--target",
        "-t",
        type=str,
        help="Target name to validate (for target validation)",
    )

    return parser


# =============================================================================
# Command Handlers
# =============================================================================


def handle_design(args: argparse.Namespace) -> None:
    """Handle the design command."""
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    # Parse overrides - these are applied to all steps
    overrides = args.overrides if args.overrides else None

    # Create config for all steps
    config = StepConfig.from_yaml(args.config, overrides=overrides)

    steps = args.steps if args.steps else None
    run_design_pipeline(config, steps, verbose=args.verbose)


def handle_analysis(args: argparse.Namespace) -> None:
    """Handle the analysis command (evaluate + analyze)."""
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    # Parse overrides - these are applied to all steps
    overrides = args.overrides if args.overrides else None

    # Create config for all steps
    config = StepConfig.from_yaml(args.config, overrides=overrides)

    # Run evaluate → analyze pipeline
    run_design_pipeline(
        config,
        ANALYSIS_PIPELINE_STEPS,
        verbose=args.verbose,
        pipeline_name="analysis",
    )


def handle_step(step_name: str, args: argparse.Namespace) -> None:
    """Handle a single step command."""
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    overrides = args.overrides if args.overrides else None
    config = StepConfig.from_yaml(args.config, overrides=overrides)
    config.verbose = args.verbose

    # Create log file for this step (include target_name and run_name if provided)
    if not config.verbose:
        config.log_file = create_log_file(
            step_name,
            run_name=config.run_name,
            target_name=config.target_name,
        )

    # Apply job args if available
    if hasattr(args, "job_id"):
        config.job_id = args.job_id

    # Print header for single step
    gpu_icon = "🔥" if STEP_REQUIRES_GPU[step_name] else "💻"
    print(f"\n{'=' * 60}")
    print(f"  {gpu_icon} Complexa: {step_name}")
    print(f"{'=' * 60}")
    print(f"  Config:  {config.config_file}")
    if config.target_name:
        print(f"  Target:  {config.target_name}")
    if config.run_name:
        print(f"  Run:     {config.run_name}")
    if config.log_file:
        print(f"  📝 Log:   {config.log_file}")
    print(f"{'=' * 60}")

    start_time = time.time()
    run_step(step_name, config)
    clear_pipeline_step, _ = _import_progress_bar()
    clear_pipeline_step()

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  ✓ Completed in {elapsed:.1f}s")
    if config.log_file:
        print(f"  📝 Full log: {config.log_file}")
    print(f"{'=' * 60}\n")


def handle_demo() -> None:
    """Handle the demo command - show usage examples."""
    print(f"""
{"=" * 70}
  🧬 Complexa CLI - Demo & Examples
{"=" * 70}

  Complexa is a binder design pipeline that generates protein binders
  for target proteins using a partially latent flow matching model.

{"─" * 70}
  ⚙️  STEP 0: Initialize Environment
{"─" * 70}

  First, create and activate the environment:

    complexa init                  # Create .env from .env_example
    # Edit .env with your paths and credentials, then:
    complexa init uv               # Generate env.sh for UV runtime
    source env.sh                  # Activate the environment

  The .env file configures paths to:
    • Data directory (DATA_PATH)
    • External tools (foldseek, sc, mmseqs)  
    • Model checkpoints (AF2, RF3)

{"─" * 70}
  📥 STEP 1: Download Model Weights
{"─" * 70}

  Download required model weights:

    complexa download              # Interactive menu
    complexa download --complexa-all  # Download all Complexa weights (protein + ligand + AME)
    complexa download --all        # Download all community model weights
    complexa download --status     # Check what's installed

{"─" * 70}
  🚀 QUICK START: Run Full Pipeline
{"─" * 70}

  Run all 4 steps (generate → filter → evaluate → analyze):

    # Recommended: Use v2 config with modular interface metrics
    complexa design configs/search_binder_local_v2.yaml

    # Legacy: Original config (still supported)
    complexa design configs/search_binder_local.yaml

  This will:
    1. Generate binder structures for the target
    2. Filter samples by reward scores  
    3. Evaluate with refolding metrics (AF2/RF3/Boltz2)
    4. Aggregate results into final CSV
  
  Overrides are applied to ALL steps:

    complexa design configs/search_binder_local_v2.yaml ++run_name=my_experiment

{"─" * 70}
  🔧 INDIVIDUAL STEPS
{"─" * 70}

  Run steps separately for more control:

  1️⃣  GENERATE - Create binder structures
      complexa generate configs/search_binder_local_v2.yaml
      
      • Uses flow matching to generate backbone + sequence
      • Output: ./inference/{{config}}_{{target}}_{{timestamp}}/

  2️⃣  FILTER - Rank by reward scores
      complexa filter configs/search_binder_local_v2.yaml
      
      • Scores: H-bonds, shape complementarity, hydrophobicity
      • Output: ./inference/.../filtered/

  3️⃣  EVALUATE - Refold and validate (v2)
      complexa evaluate configs/search_binder_local_v2.yaml
      
      • Refolds with AF2/RF3/Boltz2 to check self consistency
      • Computes scRMSD, pLDDT, iPAE metrics
      • Modular interface metrics (bioinformatics, TMOL)

  4️⃣  ANALYZE - Aggregate results
      complexa analyze configs/search_binder_local_v2.yaml
      
      • Combines all metrics into final CSV
      • Output: ./evaluation_results/.../results_*.csv

{"─" * 70}
  ⚙️  CONFIG OVERRIDES (Hydra-style)
{"─" * 70}

  Override any config value with ++key=value:

    # Change target
    complexa generate configs/search_binder_local_v2.yaml \\
        ++generation.task_name=02_PDL1

    # Change run name (affects output dir naming)
    complexa generate configs/search_binder_local_v2.yaml \\
        ++run_name=experiment_v2

    # Change sampling parameters
    complexa generate configs/search_binder_local_v2.yaml \\
        ++generation.args.nsteps=200 \\
        ++seed=42

    # Enable/disable specific interface metrics
    complexa evaluate configs/search_binder_local_v2.yaml \\
        ++metric.pre_refolding.madrax=True \\
        ++metric.refolded.tmol=False

    # For design command, overrides apply to ALL steps:
    complexa design configs/search_binder_local_v2.yaml \\
        ++run_name=full_experiment \\
        ++generation.task_name=05_CD45

{"─" * 70}
  📊 CHECK STATUS
{"─" * 70}

    complexa status            # Show output directories & counts
    complexa status --logs     # Also show recent log files

{"─" * 70}
  🎯 TARGET MANAGEMENT
{"─" * 70}

  List, add, or view target configurations:

    # List all available targets
    complexa target list
    complexa target list --verbose     # With detailed info
    complexa target list --ligand      # Only ligand targets
    complexa target list --protein     # Only protein targets

    # Add a new target interactively (opens editor)
    complexa target add -i

    # Add a target via command line
    complexa target add MyTarget \\
        --source custom_targets \\
        --target-input A1-150 \\
        --hotspot-residues A45 A67

    # Add a target with full path to PDB
    complexa target add MyTarget --target-path /data/targets/my.pdb

    # Show details for a specific target
    complexa target show 02_PDL1

{"─" * 70}
  🔍 VALIDATION
{"─" * 70}

  Validate configuration before running:

    complexa validate env                      # Check .env file
    complexa validate target config.yaml -t 02_PDL1  # Check target file exists
    complexa validate generate config.yaml     # Check generate prerequisites
    complexa validate evaluate config.yaml     # Check evaluate prerequisites  
    complexa validate design config.yaml       # Check everything (recommended)

  This checks for:
    • Missing required parameters
    • Missing model weights (prompts to download)
    • Missing files (target PDB, checkpoints)

{"─" * 70}
  💡 TIPS
{"─" * 70}

  • Use --verbose / -v to see full output (no log file)
  • Logs are saved to ./logs/ with timestamps
  • Use 'complexa target list' to see available targets
  • Use 'complexa target add -i' to interactively add new targets
  • Use 'complexa validate design config.yaml' before running

{"=" * 70}
""")


VALID_RUNTIMES = ("uv", "docker")

# Tool variables that have per-runtime variants (UV_*, DOCKER_*)
_TOOL_VARS = [
    "FOLDSEEK_EXEC",
    "RF3_EXEC_PATH",
    "SC_EXEC",
    "MMSEQS_EXEC",
    "DSSP_EXEC",
    "TMOL_PATH",
]


def _find_env_example() -> Path | None:
    """Locate .env_example starting from cwd up to project root."""
    for parent in [Path.cwd()] + list(Path.cwd().parents):
        candidate = parent / ".env_example"
        if candidate.exists():
            return candidate
    return None


def _generate_env_sh(runtime: str, env_path: Path, env_sh_path: Path) -> None:
    """Generate env.sh that sources .env and maps runtime-specific tool vars.

    The generated script:
    1. Sources .env to load all variables
    2. Overrides active tool paths with the selected runtime's variants
    3. For docker: overrides CKPT_PATH with DOCKER_CHECKPOINT_PATH
    4. Exports COMPLEXA_INIT=1 so other commands know init was run
    """
    runtime_upper = runtime.upper()

    lines = [
        "#!/bin/bash",
        f"# Generated by: complexa init {runtime}",
        f"# Runtime: {runtime}",
        "#",
        "# Source this file before running complexa commands:",
        "#   source env.sh",
        "",
        "# Load base configuration (resolve .env relative to this script)",
        '_ENVSH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'source "$_ENVSH_DIR/.env"',
        "",
        f"# Override active tool paths for {runtime} runtime",
    ]

    for var in _TOOL_VARS:
        src = f"{runtime_upper}_{var}"
        lines.append(f'export {var}="${{{src}:-${var}}}"')

    # Runtime-specific overrides
    if runtime == "docker":
        lines += [
            "",
            "# Docker: override paths with container paths",
            "# LOCAL_CODE_PATH uses ${HOME} which differs inside the container,",
            "# so re-derive it and all dependents from DOCKER_REPO_PATH.",
            'export LOCAL_CODE_PATH="${DOCKER_REPO_PATH:-$LOCAL_CODE_PATH}"',
            'export COMMUNITY_MODELS_PATH="${LOCAL_CODE_PATH}/community_models"',
            'export LOCAL_CACHE_DIR="${LOCAL_CODE_PATH}/.cache"',
            'export CKPT_PATH="${DOCKER_CHECKPOINT_PATH:-$CKPT_PATH}"',
            'export DATA_PATH="${DOCKER_DATA_PATH:-$DATA_PATH}"',
        ]

    lines += [
        "",
        "# Mark environment as initialized",
        f'export COMPLEXA_INIT="{runtime}"',
        "",
        f'echo "Complexa environment initialized for {runtime} runtime."',
        "",
    ]

    env_sh_path.write_text("\n".join(lines))
    env_sh_path.chmod(0o755)


def handle_init(args: argparse.Namespace) -> None:
    """Handle the init command.

    Phase 1 (no .env): copy .env_example -> .env, prompt user to edit.
    Phase 2 (.env exists + runtime arg): generate env.sh for the runtime.
    """
    env_path = Path(".env")
    env_sh_path = Path("env.sh")
    runtime = args.runtime

    # --- Phase 1: Create .env if missing ---
    if not env_path.exists():
        example_path = _find_env_example()
        if example_path is None:
            print("\n  .env_example not found. Cannot initialize .env.")
            print("  Make sure you are in the project root directory.")
            sys.exit(1)

        import shutil

        shutil.copy2(example_path, env_path)
        print(f"\n  Created .env from {example_path}")
        print(f"  Location: {env_path.absolute()}")
        print("\n  Next steps:")
        print("    1. Edit .env and fill in your credentials and paths")
        print("    2. Run: complexa init <uv|docker>")
        print()
        return

    # --- Phase 2: Generate env.sh ---
    if runtime is None:
        print("\n  .env already exists. Specify a runtime to generate env.sh:")
        print()
        print("    complexa init uv       # UV venv (default for local dev)")
        print("    complexa init docker   # Docker container")
        print()
        print("  Use --force to recreate .env from .env_example.")
        if args.force:
            example_path = _find_env_example()
            if example_path is None:
                print("\n  .env_example not found.")
                sys.exit(1)
            import shutil

            shutil.copy2(example_path, env_path)
            print(f"\n  Recreated .env from {example_path}")
            return
        sys.exit(1)

    if env_sh_path.exists() and not args.force:
        print("\n  env.sh already exists. Use --force to overwrite.")
        sys.exit(1)

    _generate_env_sh(runtime, env_path, env_sh_path)
    print(f"\n  Generated env.sh for {runtime} runtime.")
    print("\n  Run this before using complexa commands:")
    print("    source env.sh")
    print()


def handle_status(args: argparse.Namespace) -> None:
    """Handle the status command."""
    print(f"\n{'=' * 60}")
    print("  📊 Pipeline Status")
    print(f"{'=' * 60}\n")

    # Check default output directories
    print("  Output Directories:")

    if DEFAULT_INFERENCE_DIR.exists():
        sample_dirs = list(DEFAULT_INFERENCE_DIR.glob("*/"))
        print(f"    ✓ ./inference/ ({len(sample_dirs)} runs)")
    else:
        print("    ○ ./inference/ (not found)")

    if DEFAULT_EVAL_DIR.exists():
        eval_dirs = [d for d in DEFAULT_EVAL_DIR.glob("*/") if d.is_dir()]
        eval_csvs = list(DEFAULT_EVAL_DIR.glob("**/*.csv"))
        print(f"    ✓ ./evaluation_results/ ({len(eval_dirs)} runs, {len(eval_csvs)} CSVs)")
    else:
        print("    ○ ./evaluation_results/ (not found)")

    # Check for output files
    print("\n  Pipeline Outputs:")
    checks = [
        ("Samples", "./inference/*/samples", "🔥"),
        ("Filtered", "./inference/*/filtered", "🔥"),
        ("Results CSV (inference)", "./inference/**/*.csv", "📊"),
        ("Results CSV (eval)", "./evaluation_results/**/*.csv", "📊"),
        ("PDB files (inference)", "./inference/**/*.pdb", "🧬"),
        ("PDB files (eval)", "./evaluation_results/**/*.pdb", "🧬"),
    ]

    for description, pattern, icon in checks:
        matches = glob.glob(pattern, recursive=True)
        # Exclude _updated.pdb files from PDB counts
        if ".pdb" in pattern:
            matches = [m for m in matches if not m.endswith("_updated.pdb")]
        status = "✓" if matches else "○"
        count = len(matches) if matches else 0
        print(f"    {status} {icon} {description}: {count} found")

    # Show recent logs if requested
    if args.logs and LOG_DIR.exists():
        print(f"\n  Recent Logs ({LOG_DIR}):")
        log_files = sorted(LOG_DIR.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)[:5]
        if log_files:
            for log_file in log_files:
                mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                print(f"    📝 {log_file.name} ({mtime.strftime('%Y-%m-%d %H:%M')})")
        else:
            print("    No log files found")
    elif LOG_DIR.exists():
        log_count = len(list(LOG_DIR.glob("*.log")))
        if log_count > 0:
            print(f"\n  📝 {log_count} log files in ./logs/ (use --logs to show)")

    print(f"\n{'=' * 60}\n")


def handle_target(args: argparse.Namespace) -> None:
    """Handle the target command and its subcommands."""
    # Lazy import target manager only when target command is used
    add_target_cli, add_target_interactive, list_targets, show_target = _import_target_manager()

    if args.target_command == "list":
        # Determine filter
        filter_ligand = None
        if args.ligand:
            filter_ligand = True
        elif args.protein:
            filter_ligand = False

        list_targets(
            dict_path=args.dict,
            verbose=args.verbose,
            filter_ligand=filter_ligand,
        )

    elif args.target_command == "add":
        ligand_val = getattr(args, "ligand", None)
        is_ligand = ligand_val is not None

        if args.interactive or not args.name:
            defaults = {}
            if args.source:
                defaults["source"] = args.source
            if args.target_filename:
                defaults["target_filename"] = args.target_filename
            if args.target_path:
                defaults["target_path"] = args.target_path
            if args.target_input:
                defaults["target_input"] = args.target_input
            if args.hotspot_residues:
                defaults["hotspot_residues"] = args.hotspot_residues
            if args.binder_length:
                defaults["binder_length"] = args.binder_length
            if args.pdb_id:
                defaults["pdb_id"] = args.pdb_id
            if ligand_val:
                defaults["ligand"] = ligand_val
            if getattr(args, "smiles", None):
                defaults["SMILES"] = args.smiles
            if getattr(args, "ligand_only", None) is not None:
                defaults["ligand_only"] = args.ligand_only
            if getattr(args, "use_bonds_from_file", None) is not None:
                defaults["use_bonds_from_file"] = args.use_bonds_from_file

            success = add_target_interactive(
                dict_path=args.dict,
                name=args.name,
                defaults=defaults if defaults else None,
                is_ligand=is_ligand,
                editor=getattr(args, "editor", None),
            )
            if not success:
                sys.exit(1)
        else:
            success = add_target_cli(
                name=args.name,
                dict_path=args.dict,
                source=args.source,
                target_filename=args.target_filename,
                target_path=args.target_path,
                target_input=args.target_input,
                hotspot_residues=args.hotspot_residues,
                binder_length=args.binder_length,
                pdb_id=args.pdb_id,
                ligand=ligand_val,
                ligand_only=getattr(args, "ligand_only", None),
                smiles=getattr(args, "smiles", None),
                use_bonds_from_file=getattr(args, "use_bonds_from_file", None),
                force=args.force,
            )
            if not success:
                sys.exit(1)

    elif args.target_command == "show":
        show_target(name=args.name, dict_path=args.dict)

    else:
        # No subcommand - show help
        print(f"\n{'=' * 60}")
        print("  🎯 Complexa Target Management")
        print(f"{'=' * 60}")
        print("""
  Usage:
    complexa target list              List all targets
    complexa target list --verbose    List with details
    complexa target list --ligand     List only ligand targets
    complexa target list --protein    List only protein targets
    complexa target list --dict PATH  Use custom targets dict

    complexa target add NAME          Add protein target with defaults
    complexa target add -i            Interactive editor (uses $EDITOR or nano)
    complexa target add -i -e code    Interactive with VS Code
    complexa target add -i -e nano    Interactive with nano
    complexa target add --ligand -i   Ligand target template
    complexa target add NAME --target-path /path/to/target.pdb

    complexa target show NAME         Show target details

  Examples:
    # List all available targets
    complexa target list

    # Add a new protein target interactively
    complexa target add -i

    # Add a new ligand target interactively
    complexa target add --ligand FAD -i

    # Add a new protein target via command line
    complexa target add MyTarget \\
        --source custom_targets \\
        --target-filename MyTarget \\
        --target-input A1-150 \\
        --hotspot-residues A45 A67 A89 \\
        --binder-length 60 120

    # Add a ligand target via command line
    complexa target add MyLigand \\
        --ligand OQO \\
        --target-path /data/ligands/my_ligand.pdb \\
        --smiles "Fc1ccc(cc1)C"

    # Show details for a specific target
    complexa target show 02_PDL1
""")
        print(f"{'=' * 60}\n")


def handle_validate(args: argparse.Namespace) -> None:
    """Handle the validate command."""
    run_validation = _import_validate()

    validation_type = args.type
    config_path = args.config
    target_name = getattr(args, "target", None)

    # Check if config is required
    if validation_type in {"generate", "evaluate", "analyze", "design"} and not config_path:
        print(f"\n  ✗ Error: Config file required for '{validation_type}' validation")
        print(f"  Usage: complexa validate {validation_type} <config.yaml>")
        print()
        sys.exit(1)

    # Check config exists if provided
    if config_path and not config_path.exists():
        print(f"\n  ✗ Error: Config file not found: {config_path}")
        sys.exit(1)

    # Run validation
    report = run_validation(validation_type, config_path, target_name)
    report.print_report()

    # Exit with error code if validation failed
    if report.has_errors:
        sys.exit(1)


# =============================================================================
# Main Entry Point
# =============================================================================


_INIT_EXEMPT_COMMANDS = {"init", "demo", "download", "validate", "status"}


def _check_complexa_init(command: str) -> None:
    """Ensure the environment was initialized via 'source env.sh'.

    Commands in _INIT_EXEMPT_COMMANDS are exempt from this check.
    """
    if command in _INIT_EXEMPT_COMMANDS:
        return
    if not os.environ.get("COMPLEXA_INIT"):
        print("\n  Environment not initialized. Run:")
        print("    complexa init <uv|docker>")
        print("    source env.sh")
        print()
        sys.exit(1)


def main() -> None:
    """Main entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args()

    _check_complexa_init(args.command)

    if args.command == "init":
        handle_init(args)
    elif args.command == "demo":
        handle_demo()
    elif args.command == "design":
        handle_design(args)
    elif args.command == "analysis":
        handle_analysis(args)
    elif args.command in STEP_MODULES:
        handle_step(args.command, args)
    elif args.command == "status":
        handle_status(args)
    elif args.command == "target":
        handle_target(args)
    elif args.command == "validate":
        handle_validate(args)
    elif args.command == "download":
        # Pass remaining args after "download" to the script
        download_main(extra_args=sys.argv[2:])
    else:
        parser.print_help()


def download_main(extra_args: list[str] | None = None) -> None:
    """Entry point for complexa-download command.

    Parameters
    ----------
    extra_args : list[str], optional
        Additional arguments to pass to the download script.
        If None, uses sys.argv[1:] (for complexa-download entry point).
    """
    import shutil

    # Find the download script (env/ is at project root, not in src/)
    script_path = Path(__file__).parent.parent.parent.parent / "env" / "download_startup.sh"

    if not script_path.exists():
        print(f"Error: Download script not found at {script_path}")
        sys.exit(1)

    # Check if bash is available
    bash_path = shutil.which("bash")
    if not bash_path:
        print("Error: bash not found. Please run the script directly:")
        print(f"  bash {script_path}")
        sys.exit(1)

    # Build command - use extra_args if provided, otherwise sys.argv[1:]
    args_to_pass = extra_args if extra_args is not None else sys.argv[1:]
    cmd = [bash_path, str(script_path)] + args_to_pass

    try:
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nDownload cancelled.")
        sys.exit(1)


if __name__ == "__main__":
    main()
