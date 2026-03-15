import os
import shutil
import sys
from typing import Any

import hydra
import lightning as L
import pandas as pd
import torch
from dotenv import load_dotenv
from loguru import logger


def _get_filter_cfg(cfg) -> Any:
    """Return the filter config, preferring ``generation.filter`` over legacy
    ``generation.search`` keys.

    Older configs stored filter-only settings (``filter_samples_limit``,
    ``delete_non_top_n_samples``, ``dedup_sequence``) under ``search:``.
    New configs have a dedicated ``filter:`` section.  This helper reads
    from ``filter`` when present and falls back to ``search`` with a
    deprecation warning so existing configs keep working.
    """
    gen = cfg.generation
    if hasattr(gen, "filter") and gen.filter is not None:
        return gen.filter

    logger.warning(
        "No 'filter' section found in generation config — falling back to "
        "legacy keys under 'search'.  Please move filter_samples_limit, "
        "delete_non_top_n_samples, dedup_sequence, and reward_threshold "
        "into a 'filter:' section."
    )
    return gen.search


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
        root_path = f"./inference/{config_name}_{task_name}"
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


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="inference_base",
)
def main(cfg):
    load_dotenv()

    # Parse arguments and load config
    # args, cfg, config_name = parse_args_and_cfg()
    # the base config name in case the config file is generated from sweep script.
    config_name = cfg.get("base_config_name", hydra.core.hydra_config.HydraConfig.get().job.config_name)
    job_id = cfg.get("job_id", 0)
    root_path = cfg.get("root_path", None)
    run_name = cfg.get("run_name", None)
    task_name = cfg.generation.dataloader.dataset.get("task_name", None)

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

    if not os.path.exists(root_path):
        raise ValueError(f"Results path {root_path} does not exist. Run generation first.")

    # Collect and combine reward files
    reward_files = []
    for file in os.listdir(root_path):
        if file.startswith(f"rewards_{config_name}_") and file.endswith(".csv"):
            reward_files.append(os.path.join(root_path, file))

    if not reward_files:
        raise ValueError("No reward files found!")

    logger.info(f"Found {len(reward_files)} reward file(s) to process")

    # Load and combine all reward data
    combined_rewards = pd.DataFrame()
    for reward_file in reward_files:
        try:
            df = pd.read_csv(reward_file)
            logger.info(f"  Loaded {len(df)} samples from {os.path.basename(reward_file)}")
            combined_rewards = pd.concat([combined_rewards, df], ignore_index=True)
        except Exception as e:
            logger.error(f"Error loading {reward_file}: {e!s}")

    initial_count = len(combined_rewards)
    logger.info(f"Total samples loaded: {initial_count}")

    combined_rewards = combined_rewards.dropna(subset=["total_reward"])
    after_dropna_count = len(combined_rewards)
    if after_dropna_count < initial_count:
        logger.info(f"Dropped {initial_count - after_dropna_count} samples with missing total_reward")
        logger.info(f"Samples after dropping NaN: {after_dropna_count}")

    # Resolve filter config (new `filter:` section, or legacy `search:` keys)
    filter_cfg = _get_filter_cfg(cfg)

    ## Deduplicate based on sequence
    after_dedup = after_dropna_count  # Default if dedup not used
    if filter_cfg.get("dedup_sequence", True):
        before_dedup = len(combined_rewards)
        combined_rewards = combined_rewards.drop_duplicates(subset=["aatype"])
        after_dedup = len(combined_rewards)
        logger.info(
            f"Sequence deduplication: {before_dedup} -> {after_dedup} samples ({before_dedup - after_dedup} duplicates removed)"
        )

    # Select top N samples or samples above the threshold
    try:
        default_samples = cfg.generation.dataloader.dataset.nres.nsamples
    except (AttributeError, KeyError):
        default_samples = 1000
    total_samples = filter_cfg.get("filter_samples_limit", default_samples)
    reward_threshold = filter_cfg.get("reward_threshold", None)
    delete_files = filter_cfg.get("delete_non_top_n_samples", False)
    logger.info("Filtering configuration:")
    logger.info(f"  Available samples: {len(combined_rewards)}")
    logger.info(f"  Max to keep:       {total_samples}")
    logger.info(f"  Reward threshold:  {reward_threshold}")
    logger.info(f"  Delete unselected: {delete_files}")

    # Initialize top_samples (will be filtered if needed)
    top_samples = combined_rewards.sort_values("total_reward", ascending=False)

    if len(combined_rewards) > total_samples:
        logger.info(f"Filtering {len(combined_rewards)} samples down to {total_samples}...")

        # Apply reward threshold if set
        if reward_threshold is not None:
            top_samples = top_samples[top_samples["total_reward"] >= reward_threshold]
            logger.info(f"After reward threshold: {len(top_samples)} samples")

        # Limit to top N samples
        top_samples = top_samples.head(total_samples)
        logger.info(f"Selected {len(top_samples)} samples after filtering")

        if len(top_samples) == 0:
            logger.warning("No samples passed filtering — skipping directory deletion to avoid removing all data")
        else:
            # Get directories to keep
            keep_dirs = set()
            for pdb_path in top_samples["pdb_path"].tolist():
                if os.path.exists(pdb_path):
                    keep_dirs.add(os.path.abspath(os.path.dirname(pdb_path)))

            # Delete unselected directories if requested
            if delete_files:
                logger.info(f"Keeping {len(keep_dirs)} directories, deleting the rest...")
                deleted_count = 0
                for item in os.listdir(root_path):
                    item_path = os.path.abspath(os.path.join(root_path, item))
                    if os.path.isdir(item_path) and item_path not in keep_dirs:
                        try:
                            shutil.rmtree(item_path)
                            deleted_count += 1
                        except Exception as e:
                            logger.error(f"Error deleting {item_path}: {e!s}")
                logger.info(f"Deleted {deleted_count} directories")
            else:
                logger.info(f"Keeping {len(keep_dirs)} directories, moving rest to filtered_out_samples...")
                moved_count = 0
                filtered_root = os.path.join(os.path.abspath(root_path), "filtered_out_samples")
                os.makedirs(filtered_root, exist_ok=True)
                for item in os.listdir(root_path):
                    item_path = os.path.abspath(os.path.join(root_path, item))
                    if item_path == filtered_root:
                        continue
                    if os.path.isdir(item_path) and item_path not in keep_dirs:
                        dest_path = os.path.join(filtered_root, os.path.basename(item_path))
                        suffix = 1
                        while os.path.exists(dest_path):
                            dest_path = os.path.join(filtered_root, f"{os.path.basename(item_path)}_{suffix}")
                            suffix += 1
                        try:
                            shutil.move(item_path, dest_path)
                            moved_count += 1
                        except Exception as e:
                            logger.error(f"Error moving {item_path} to {dest_path}: {e!s}")
                logger.info(f"Moved {moved_count} directories to {filtered_root}")
    else:
        logger.info(f"No filtering needed: {len(combined_rewards)} samples <= {total_samples} limit")

    # Save all rewards
    combined_rewards.to_csv(os.path.join(root_path, f"all_rewards_{config_name}.csv"), index=False)
    logger.info(f"Saved all rewards to {os.path.join(root_path, f'all_rewards_{config_name}.csv')}")
    top_samples.to_csv(os.path.join(root_path, f"top_samples_{config_name}.csv"), index=False)
    logger.info(f"Saved top samples to {os.path.join(root_path, f'top_samples_{config_name}.csv')}")

    # Final summary
    logger.info("")
    logger.info(f"{'=' * 50}")
    logger.info("FILTER SUMMARY")
    logger.info(f"{'=' * 50}")
    logger.info(f"  Initial samples:    {initial_count}")
    logger.info(f"  After NaN removal:  {after_dropna_count}")
    if filter_cfg.get("dedup_sequence", True):
        logger.info(f"  After dedup:        {after_dedup}")
    logger.info(f"  Final top samples:  {len(top_samples)}")
    if len(top_samples) > 0:
        logger.info(f"  Best reward:        {top_samples['total_reward'].iloc[0]:.4f}")
        logger.info(f"  Worst kept reward:  {top_samples['total_reward'].iloc[-1]:.4f}")
    logger.info(f"{'=' * 50}")


if __name__ == "__main__":
    main()
