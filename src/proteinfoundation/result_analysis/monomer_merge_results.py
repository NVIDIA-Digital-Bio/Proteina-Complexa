"""
Aggregate Monomer Results Across Configs

Merges all "overall_*" CSV files from different eval_* directories into single
combined files for easier analysis of sweeps over checkpoints/hyperparameters.

Usage:
    python -m proteinfoundation.result_analysis.monomer_merge_results <results_dir>
    
    # Example:
    python -m proteinfoundation.result_analysis.monomer_merge_results \
        results_downloaded/my_run-monomer-2026_02_18_11

Output:
    Creates in <results_dir>:
    - all_monomer_pass_rates.csv     (if overall_monomer_pass_rates_*.csv found)
    - all_monomer_diversity.csv      (if overall_monomer_diversity_*.csv found)
    - all_monomer_combined.csv       (merged pass rates + diversity)
"""

import argparse
import logging
import os
import re
import sys
from collections import defaultdict
from glob import glob

import pandas as pd

# Columns to exclude from output (not useful for analysis)
COLUMNS_TO_DROP = ["ckpt_path", "run_name"]

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_overall_csvs(eval_results_dir: str) -> dict[str, list[str]]:
    """
    Find all overall_*.csv files grouped by type.

    Args:
        eval_results_dir: Path to evaluation_results directory

    Returns:
        Dictionary mapping file type (e.g., "monomer_pass_rates") to list of file paths
    """
    # Pattern: overall_{type}_{config_name}.csv
    # e.g., overall_monomer_pass_rates_eval_0_run_name.csv
    pattern = os.path.join(eval_results_dir, "eval_*", "overall_*.csv")
    all_files = glob(pattern)

    if not all_files:
        logger.warning(f"No overall_*.csv files found in {eval_results_dir}/eval_*/")
        return {}

    # Group by file type (extract from filename)
    # overall_monomer_pass_rates_eval_0_... -> monomer_pass_rates
    grouped: dict[str, list[str]] = defaultdict(list)

    for filepath in all_files:
        filename = os.path.basename(filepath)
        # Extract type: overall_{TYPE}_{config_name}.csv
        match = re.match(r"overall_(.+?)_eval_\d+.*\.csv", filename)
        if match:
            file_type = match.group(1)
            grouped[file_type].append(filepath)
        else:
            logger.debug(f"Could not parse file type from: {filename}")

    return dict(grouped)


def merge_csv_files(filepaths: list[str], output_path: str) -> pd.DataFrame | None:
    """
    Merge multiple CSV files into one.

    Args:
        filepaths: List of CSV file paths to merge
        output_path: Path to save merged CSV

    Returns:
        Merged DataFrame, or None if no valid files
    """
    dfs = []
    for filepath in sorted(filepaths):
        try:
            df = pd.read_csv(filepath)
            # Add source directory for traceability
            eval_dir = os.path.basename(os.path.dirname(filepath))
            df["_source_eval_dir"] = eval_dir
            dfs.append(df)
            logger.debug(f"  Loaded {filepath} ({len(df)} rows)")
        except Exception as e:
            logger.warning(f"  Failed to read {filepath}: {e}")

    if not dfs:
        return None

    merged = pd.concat(dfs, ignore_index=True)
    # Drop columns that aren't useful for analysis
    merged = merged.drop(columns=[c for c in COLUMNS_TO_DROP if c in merged.columns])
    merged.to_csv(output_path, index=False)
    logger.info(f"  Saved: {output_path} ({len(merged)} rows)")

    return merged


def get_identifier_columns(df: pd.DataFrame) -> list[str]:
    """Get columns that are identifiers (not result metrics)."""
    return [col for col in df.columns if not col.startswith("_res_") and col != "_source_eval_dir"]


def merge_all_results(
    results_dir: str,
    merged_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame | None:
    """
    Merge all result DataFrames into a single combined file.

    Args:
        results_dir: Output directory
        merged_dfs: Dictionary of file_type -> merged DataFrame

    Returns:
        Combined DataFrame, or None if nothing to merge
    """
    if not merged_dfs:
        return None

    # Start with the first DataFrame
    df_list = list(merged_dfs.values())
    combined = df_list[0].copy()

    if len(df_list) > 1:
        # Get identifier columns from first df
        id_cols = get_identifier_columns(combined)
        # Add _source_eval_dir to merge keys
        merge_cols = [c for c in id_cols if c in combined.columns]
        if "_source_eval_dir" in combined.columns:
            merge_cols.append("_source_eval_dir")

        # Merge remaining DataFrames
        for df in df_list[1:]:
            # Find common columns to merge on
            common_cols = [c for c in merge_cols if c in df.columns]
            if common_cols:
                # Only keep result columns from df to avoid duplication
                result_cols = [c for c in df.columns if c.startswith("_res_")]
                cols_to_keep = common_cols + result_cols
                df_subset = df[cols_to_keep].drop_duplicates(subset=common_cols)
                combined = combined.merge(df_subset, on=common_cols, how="outer")
            else:
                logger.warning("No common columns to merge on, skipping")

    output_path = os.path.join(results_dir, "all_monomer_combined.csv")
    combined.to_csv(output_path, index=False)
    logger.info(f"  Saved combined: {output_path} ({len(combined)} rows, {len(combined.columns)} cols)")

    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate monomer results across configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Path to results directory (e.g., results_downloaded/my_run-monomer-2026_02_18_11)",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Show what would be done without creating files",
    )

    args = parser.parse_args()

    # Handle relative paths - check if it's relative to results_downloaded
    results_dir = args.results_dir
    if not os.path.isabs(results_dir) and not os.path.exists(results_dir):
        # Try prepending results_downloaded
        alt_path = os.path.join("results_downloaded", results_dir)
        if os.path.exists(alt_path):
            results_dir = alt_path

    if not os.path.isdir(results_dir):
        logger.error(f"Results directory not found: {results_dir}")
        sys.exit(1)

    eval_results_dir = os.path.join(results_dir, "evaluation_results")
    if not os.path.isdir(eval_results_dir):
        logger.error(f"evaluation_results directory not found in: {results_dir}")
        sys.exit(1)

    logger.info(f"Aggregating results from: {results_dir}")

    # Find all overall CSV files grouped by type
    grouped_files = find_overall_csvs(eval_results_dir)

    if not grouped_files:
        logger.error("No files to aggregate")
        sys.exit(1)

    logger.info(f"Found {len(grouped_files)} file type(s):")
    for file_type, files in grouped_files.items():
        logger.info(f"  - {file_type}: {len(files)} files")

    if args.dryrun:
        logger.info("Dryrun mode - no files will be created")
        return

    # Merge each file type
    merged_dfs: dict[str, pd.DataFrame] = {}

    for file_type, filepaths in grouped_files.items():
        output_filename = f"all_{file_type}.csv"
        output_path = os.path.join(results_dir, output_filename)

        logger.info(f"\nMerging {file_type} ({len(filepaths)} files)...")
        df = merge_csv_files(filepaths, output_path)
        if df is not None:
            merged_dfs[file_type] = df

    # Create combined file with all metrics
    if len(merged_dfs) > 1:
        logger.info("\nCreating combined file with all metrics...")
        merge_all_results(results_dir, merged_dfs)

    logger.info("\nAggregation complete!")
    logger.info(f"Output files are in: {results_dir}")


if __name__ == "__main__":
    main()
