import argparse
import glob
import os
import shutil

import pandas as pd
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate successful samples")
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing evaluation results",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Show what would be done without copying files",
    )
    return parser.parse_args()


def aggregate_for_seq_type(inf_dir: str, seq_type: str, path: str, dryrun: bool):
    """Aggregate successful samples for a given sequence type within a result directory."""
    cluster_dir = os.path.join(inf_dir, f"clusters_binder_successful_{seq_type}")
    output_dir = os.path.join(inf_dir, f"successful_{seq_type}_samples")

    if not os.path.isdir(cluster_dir):
        return
    if os.path.exists(output_dir):
        return

    txt_files = glob.glob(os.path.join(cluster_dir, "*.txt"))
    if not txt_files:
        logger.warning(f"Directory {cluster_dir} exists but contains no .txt files — skipping")
        return

    success_path = txt_files[0]
    logger.info(f"Aggregating successful samples for {seq_type} in {inf_dir}")

    if dryrun:
        logger.info(f"  [DRY RUN] Would read {success_path} and copy samples to {output_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    success_df = pd.read_csv(success_path, sep="\t", header=None)

    def fix_path(x):
        x = x.replace("./evaluation_results", path, 1)
        if "/job_" in x:
            x = x.replace("/job_", "/all_samples/job_", 1)
        elif "/tmp_j" in x:
            x = x.replace("/tmp_j", "/all_samples/tmp_j", 1)
        return x

    paths = success_df[1].apply(fix_path)
    copied = 0
    for spath in paths:
        src_dir = os.path.dirname(spath)
        dest_dir = os.path.join(output_dir, os.path.basename(src_dir))
        if not os.path.exists(dest_dir):
            if os.path.exists(src_dir):
                shutil.copytree(src_dir, dest_dir)
                copied += 1
            else:
                logger.warning(f"Source directory not found: {src_dir}")

    logger.info(f"  Copied {copied} sample directories to {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    path = args.results_dir
    if not path.startswith("/"):
        path = os.path.join(os.getcwd(), "results_downloaded", path)

    result_dirs = glob.glob(os.path.join(path, "inf_*")) + glob.glob(os.path.join(path, "eval_*"))
    if not result_dirs:
        logger.warning(f"No inf_* or eval_* directories found in {path}")

    for inf_dir in result_dirs:
        for seq_type in ["self", "mpnn", "mpnn_fixed"]:
            aggregate_for_seq_type(inf_dir, seq_type, path, args.dryrun)
