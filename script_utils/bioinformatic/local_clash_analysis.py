import argparse
import logging
import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from pathlib import Path
from typing import Literal

import pandas as pd
from pymol import cmd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(processName)s %(message)s")


# Defaults
DEFAULT_ROOT = "/path/to/data"
DEFAULT_CSV_PATTERN = "*/inf_0/protein_binder_results_eval_0_combined.csv"
DEFAULT_TARGET_DATA_ROOT = "/path/to/target_data/truncated_structures"


def fix_path(current_path, df_parent_path):
    return current_path.replace("./evaluation_results/inf_0", os.path.join(df_parent_path, "all_samples"))


def extract_task_name(run_dir_name: str) -> str:
    match = re.search(r"_([^_]+_[^_]+)_A\d", run_dir_name)
    if not match:
        raise ValueError(f"unexpected run dir name: {run_dir_name}")
    return match.group(1)


def compute_clash_count(
    binder: str,
    refolded_data: str,
    mode: Literal["backbone", "all", "both"] = "both",
    cutoff: float = 2.0,
) -> dict:
    logger.debug(f"Computing clash counts for {binder} and {refolded_data} with mode {mode} and cutoff {cutoff}")
    try:
        cmd.delete("all")
        cmd.load(refolded_data, "target")
        cmd.load(binder, "mobile")
        rmsd = cmd.align("mobile and chain A", "target")
        logger.debug("Alignment RMSD: %s", rmsd)
        results = {}
        if mode == "backbone":
            cmd.select("target_backbone", "target and name N+CA+C+O")
            cmd.select("mobile_chainB", "mobile and chain B")
            cmd.select("mobile_backbone", "mobile_chainB and name N+CA+C+O")
            clashes = cmd.find_pairs("mobile_backbone", "target_backbone", cutoff=cutoff)
            print(f"{len(clashes)} backbone atom pairs below {cutoff} Å between mobile chain B and the target")
            results[f"backbone_{cutoff}"] = len(clashes)
        elif mode == "all":
            cmd.select("mobile_chainB", "mobile and chain B")
            clashes = cmd.find_pairs("mobile_chainB", "target", cutoff=cutoff)
            print(f"{len(clashes)} atom pairs below {cutoff} Å between mobile chain B and the target")
            results[f"all_{cutoff}"] = len(clashes)
        elif mode == "both":
            cmd.select("target_backbone", "target and name N+CA+C+O")
            cmd.select("mobile_chainB", "mobile and chain B")
            cmd.select("mobile_backbone", "mobile_chainB and name N+CA+C+O")
            clashes = cmd.find_pairs("mobile_backbone", "target_backbone", cutoff=cutoff)
            print(f"{len(clashes)} backbone atom pairs below {cutoff} Å between mobile chain B and the target")
            results[f"backbone_{cutoff}"] = len(clashes)
            clashes = cmd.find_pairs("mobile_chainB", "target", cutoff=cutoff)
            print(f"{len(clashes)} atom pairs below {cutoff} Å between mobile chain B and the target")
            results[f"all_{cutoff}"] = len(clashes)
        # cmd.clear()
        # cmd.save("clash_counts.pdb")
    except Exception as e:
        logger.error(f"Error computing clash counts: {e}")
        return {
            "backbone_1.0": -1,
            "backbone_2.0": -1,
            "all_1.0": -1,
            "all_2.0": -1,
        }
    return results


def process_combined_path(combined_path: str, target_data_root: str, cutoffs: list[float], force: bool = False) -> None:
    output_path = combined_path.replace(".csv", "_clash_counts.csv")
    if os.path.exists(output_path) and not force:
        logger.info(
            "Skipping %s because %s already exists (use --force to overwrite)",
            combined_path,
            output_path,
        )
        return
    df = pd.read_csv(combined_path)
    df["self_complex_pdb_path"] = df["self_complex_pdb_path"].apply(
        lambda x: fix_path(x, os.path.dirname(combined_path))
    )
    run_dir = Path(combined_path).parents[1]
    task_name = extract_task_name(run_dir.name)
    refolded_binders = df["self_complex_pdb_path"].tolist()
    refolded_data_path = os.path.join(target_data_root, f"{task_name}/AF*.pdb")
    refolded_data_matches = glob(refolded_data_path)
    if not refolded_data_matches:
        logger.error("No refolded data found at %s", refolded_data_path)
        return
    refolded_data = refolded_data_matches[0]

    for cutoff in cutoffs:
        logger.info("Computing clash counts for cutoff %s on %s", cutoff, combined_path)
        clash_counts = [
            compute_clash_count(refolded_binder, refolded_data, mode="both", cutoff=cutoff)
            for refolded_binder in refolded_binders
        ]
        for type in ["backbone", "all"]:
            col_name = f"clash_count_{type}_{cutoff}"
            df[col_name] = [entry[f"{type}_{cutoff}"] for entry in clash_counts]

    df.to_csv(output_path, index=False)
    logger.info("Saved clash counts to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute clash counts between refolded binders and target structures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=str,
        default=DEFAULT_ROOT,
        help="Root directory containing result folders",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=DEFAULT_CSV_PATTERN,
        help="Glob pattern for finding combined CSV files (relative to root)",
    )
    parser.add_argument(
        "--target-data-root",
        type=str,
        default=DEFAULT_TARGET_DATA_ROOT,
        help="Root directory containing target structure data",
    )
    parser.add_argument(
        "--cutoffs",
        type=float,
        nargs="+",
        default=[1.0, 2.0],
        help="Distance cutoffs (in Angstroms) for clash detection",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: min(num_files, cpu_count))",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    combined_paths = glob(os.path.join(args.root, args.pattern))
    if not combined_paths:
        print(f"No combined CSVs found matching: {os.path.join(args.root, args.pattern)}")
        return

    print(f"Found {len(combined_paths)} CSV files to process")

    max_workers = args.workers or min(len(combined_paths), os.cpu_count() or 16)

    # Use functools.partial to pass additional arguments
    from functools import partial

    process_fn = partial(
        process_combined_path,
        target_data_root=args.target_data_root,
        cutoffs=args.cutoffs,
        force=args.force,
    )

    with ProcessPoolExecutor(mp_context=mp.get_context("spawn"), max_workers=max_workers) as executor:
        list(executor.map(process_fn, combined_paths))


if __name__ == "__main__":
    main()
