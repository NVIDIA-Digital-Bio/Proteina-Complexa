#!/usr/bin/env python3
"""
Script to load data directly from pdb_multimer_binder_synthetic dataset CIF files,
apply secondary structure filtering, save statistics to CSV,
and save excluded sample IDs to txt file using multiprocessing.

This script reads directly from CIF files in the refolded_structures directory,
using strucio which handles CIF files natively for secondary structure analysis.

Usage Examples:
    # Process entire dataset
    python script_utils/ss_filter_synthetic_multimer.py

    # Process 1000 structures with 8 workers
    python script_utils/ss_filter_synthetic_multimer.py --num_samples 1000 --num_workers 8

    # Use different coil threshold for exclusion
    python script_utils/ss_filter_synthetic_multimer.py --coil_threshold 0.6

    # Custom output directory
    python script_utils/ss_filter_synthetic_multimer.py --output_dir ss_filtering_results

    # Set random seed for reproducibility
    python script_utils/ss_filter_synthetic_multimer.py --seed 42

    # Specify custom data directory
    python script_utils/ss_filter_synthetic_multimer.py --data_dir /path/to/data

Arguments:
    --data_dir: Data directory path (default: from DATA_PATH env variable)
    --num_samples: Number of samples to process (default: -1 for entire dataset)
    --output_dir: Output directory for results (default: ss_filtering_results)
    --seed: Random seed for reproducibility (default: 43)
    --num_workers: Number of worker processes for multiprocessing (default: 4)
    --coil_threshold: Coil fraction threshold for exclusion (default: 0.5)
"""

import argparse
import csv
import multiprocessing as mp
import os
import pathlib
import random
import tempfile
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()  # Load env variables before importing logger to set LOGURU_LEVEL correctly

from loguru import logger

from proteinfoundation.metrics.structural_metric_ss_ca_ca import compute_ss_metrics

# Import BioPython for residue counting only
try:
    from Bio.PDB import MMCIFParser, PDBParser
except ImportError:
    logger.warning("BioPython not available. Residue counting may use fallback method.")
    MMCIFParser = None
    PDBParser = None


def count_residues_from_structure(structure_path):
    """Count residues from structure file using BioPython.

    Args:
        structure_path: Path to structure file (CIF or PDB)

    Returns:
        int: Number of residues, or 0 if counting fails
    """
    try:
        if structure_path.endswith(".cif") or structure_path.endswith(".cif.gz"):
            if MMCIFParser is not None:
                parser = MMCIFParser(QUIET=True)
                structure = parser.get_structure("temp", structure_path)
                return sum(len(chain) for model in structure for chain in model)
        elif structure_path.endswith(".pdb"):
            if PDBParser is not None:
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure("temp", structure_path)
                return sum(len(chain) for model in structure for chain in model)
    except Exception:
        pass
    return 0


def process_single_cif_file(cif_file_path, coil_threshold):
    """Process a single CIF file for secondary structure filtering.

    Args:
        cif_file_path: Path to the CIF file
        coil_threshold: Coil fraction threshold for exclusion

    Returns:
        dict: Processing results including metrics and exclusion flags
    """
    try:
        # Extract sample ID from filename
        sample_id = pathlib.Path(cif_file_path).stem.replace(".cif", "")
        if sample_id.endswith(".gz"):
            sample_id = sample_id[:-3]  # Remove .gz extension

        # Handle gzipped CIF files by extracting to temporary file
        structure_file_path = cif_file_path
        temp_cif_path = None

        if cif_file_path.endswith(".gz"):
            import gzip

            temp_cif_path = tempfile.NamedTemporaryFile(mode="w", suffix=".cif", delete=False).name
            with gzip.open(cif_file_path, "rt") as gz_file:
                content = gz_file.read()
            with open(temp_cif_path, "w") as temp_cif:
                temp_cif.write(content)
            structure_file_path = temp_cif_path

        try:
            # Compute secondary structure metrics directly on the CIF file
            # strucio.load_structure() can handle CIF files natively
            complex_metrics = compute_ss_metrics(structure_file_path)

            # Count residues from the structure file
            complex_length = count_residues_from_structure(structure_file_path)

            # Fallback if we couldn't get the length
            if complex_length == 0:
                complex_length = 100  # Default fallback

            # Determine exclusion flags
            exclude_high_coil = complex_metrics.get("biot_coil", 0.0) > coil_threshold

            # Prepare secondary structure stats
            ss_stats = {
                "sample_id": sample_id,
                "protein_type": "complex",
                "coil_fraction": complex_metrics.get("biot_coil", 0.0),
                "alpha_fraction": complex_metrics.get("biot_alpha", 0.0),
                "beta_fraction": complex_metrics.get("biot_beta", 0.0),
                "length": complex_length,
                "exclude_high_coil": exclude_high_coil,
            }

            return {
                "success": True,
                "sample_id": sample_id,
                "ss_stats": ss_stats,
                "exclude_high_coil": exclude_high_coil,
            }

        finally:
            # Clean up temporary CIF file if created for gzipped files
            if temp_cif_path and os.path.exists(temp_cif_path):
                os.unlink(temp_cif_path)

    except Exception as e:
        sample_id = pathlib.Path(cif_file_path).stem if cif_file_path else "unknown"
        logger.error(f"Error processing sample {sample_id}: {e!s}")
        return {"success": False, "error": str(e), "sample_id": sample_id}


def process_wrapper_ss_filter(args):
    """Wrapper function for multiprocessing."""
    cif_file_path, coil_threshold = args
    return process_single_cif_file(cif_file_path, coil_threshold)


def main():
    """Main function for secondary structure filtering."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Apply secondary structure filtering to synthetic multimer dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Data directory path (default: from DATA_PATH env variable + synthetic_data/boltz_refold_pdb_multimer/)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of samples to process (default: -1 for entire dataset)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ss_filtering_results",
        help="Output directory for results (default: ss_filtering_results)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=43,
        help="Random seed for reproducibility (default: 43)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for multiprocessing (default: 4)",
    )
    parser.add_argument(
        "--coil_threshold",
        type=float,
        default=0.5,
        help="Coil fraction threshold for exclusion (default: 0.5)",
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Determine data directory
    if args.data_dir is None:
        data_path = os.environ.get("DATA_PATH")
        if data_path is None:
            raise ValueError("DATA_PATH environment variable not set and --data_dir not provided")
        data_dir = pathlib.Path(data_path) / "synthetic_data" / "boltz_refold_pdb_multimer"
    else:
        data_dir = pathlib.Path(args.data_dir)

    # Look for CIF files in the refolded_structures directory (raw data)
    raw_dir = data_dir / "refolded_structures"

    logger.info("=== Secondary Structure Filtering for PDB Multimer Binder Synthetic ===")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Raw structures directory: {raw_dir}")
    logger.info(f"Number of samples to process: {'entire dataset' if args.num_samples == -1 else args.num_samples}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Number of workers: {args.num_workers}")
    logger.info(f"Coil threshold: {args.coil_threshold}")

    # Check if raw directory exists
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw structures directory does not exist: {raw_dir}")

    # Find all CIF files in the raw directory
    cif_files = []
    cif_files.extend(list(raw_dir.glob("*.cif")))
    cif_files.extend(list(raw_dir.glob("*.cif.gz")))

    if len(cif_files) == 0:
        raise FileNotFoundError(f"No CIF files found in {raw_dir}")

    logger.info(f"Found {len(cif_files)} CIF files in raw directory")

    # Shuffle and limit files if needed
    random.shuffle(cif_files)
    if args.num_samples != -1 and args.num_samples < len(cif_files):
        cif_files = cif_files[: args.num_samples]
        logger.info(f"Limited to {len(cif_files)} files for processing")

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Initialize storage for results
    all_ss_stats = []
    excluded_high_coil_ids = []
    successful_processes = 0
    failed_processes = 0

    logger.info("Starting secondary structure filtering...")

    # Process files
    if args.num_workers > 1:
        logger.info(f"Using multiprocessing with {args.num_workers} workers")

        # Create a list of tasks for multiprocessing
        tasks = [(str(cif_file), args.coil_threshold) for cif_file in cif_files]

        logger.info(f"Created {len(tasks)} tasks for multiprocessing")
        logger.info(f"Starting multiprocessing with {args.num_workers} workers...")

        # Process with multiprocessing with progress bar
        with mp.Pool(processes=args.num_workers) as pool:
            # Use imap for real-time progress tracking
            results = []
            for result in tqdm(
                pool.imap(process_wrapper_ss_filter, tasks),
                total=len(tasks),
                desc="Processing samples",
            ):
                results.append(result)
    else:
        logger.info("Using single-threaded processing")

        # Process files sequentially
        results = []
        for cif_file in tqdm(cif_files, desc="Processing files"):
            result = process_single_cif_file(str(cif_file), args.coil_threshold)
            results.append(result)

    # Process results
    logger.info("Processing results...")
    for result in results:
        if not result["success"]:
            logger.warning(f"Failed to process {result['sample_id']}: {result.get('error', 'Unknown error')}")
            failed_processes += 1
            continue

        successful_processes += 1
        sample_id = result["sample_id"]

        # Store SS statistics
        all_ss_stats.append(result["ss_stats"])

        # Track exclusions
        if result["exclude_high_coil"]:
            excluded_high_coil_ids.append(sample_id)

    # All excluded IDs (only high coil now)
    all_excluded_ids = set(excluded_high_coil_ids)

    logger.info("=== Processing Summary ===")
    logger.info(f"Total samples processed: {successful_processes + failed_processes}")
    logger.info(f"Successfully processed: {successful_processes}")
    logger.info(f"Failed to process: {failed_processes}")
    logger.info(f"Samples excluded for high coil (>{args.coil_threshold}): {len(excluded_high_coil_ids)}")
    logger.info(f"Total excluded samples: {len(all_excluded_ids)}")

    # Save secondary structure statistics to CSV
    ss_csv_file = os.path.join(output_dir, "secondary_structure_stats.csv")
    logger.info(f"Saving secondary structure statistics to: {ss_csv_file}")

    with open(ss_csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample_id",
                "protein_type",
                "coil_fraction",
                "alpha_fraction",
                "beta_fraction",
                "length",
                "exclude_high_coil",
            ]
        )
        for stat in all_ss_stats:
            writer.writerow(
                [
                    stat["sample_id"],
                    stat["protein_type"],
                    f"{stat['coil_fraction']:.4f}",
                    f"{stat['alpha_fraction']:.4f}",
                    f"{stat['beta_fraction']:.4f}",
                    stat["length"],
                    stat["exclude_high_coil"],
                ]
            )

    # Save excluded sample IDs (high coil)
    excluded_coil_file = os.path.join(output_dir, "excluded_high_coil_ids.txt")
    logger.info(f"Saving excluded high coil sample IDs to: {excluded_coil_file}")

    with open(excluded_coil_file, "w") as f:
        f.write(f"# Samples excluded for high coil fraction (>{args.coil_threshold})\n")
        f.write(f"# Total excluded: {len(excluded_high_coil_ids)}\n")
        for sample_id in sorted(excluded_high_coil_ids):
            f.write(f"{sample_id}\n")

    # Save all excluded sample IDs (same as high coil since that's the only filter)
    excluded_all_file = os.path.join(output_dir, "excluded_sample_ids.txt")
    logger.info(f"Saving all excluded sample IDs to: {excluded_all_file}")

    with open(excluded_all_file, "w") as f:
        f.write(f"# Samples excluded for high coil fraction (>{args.coil_threshold})\n")
        f.write(f"# Total excluded: {len(all_excluded_ids)}\n")
        for sample_id in sorted(all_excluded_ids):
            f.write(f"{sample_id}\n")

    # Calculate and log secondary structure distribution statistics
    logger.info("\n=== Secondary Structure Distribution Statistics ===")
    df = pd.DataFrame(all_ss_stats)

    if not df.empty:
        logger.info(f"Mean coil fraction: {df['coil_fraction'].mean():.4f} ± {df['coil_fraction'].std():.4f}")
        logger.info(f"Mean alpha fraction: {df['alpha_fraction'].mean():.4f} ± {df['alpha_fraction'].std():.4f}")
        logger.info(f"Mean beta fraction: {df['beta_fraction'].mean():.4f} ± {df['beta_fraction'].std():.4f}")
        logger.info(f"Mean length: {df['length'].mean():.1f} ± {df['length'].std():.1f}")

        # Log exclusion percentages
        exclude_coil_pct = (df["exclude_high_coil"].sum() / len(df)) * 100
        logger.info(f"Exclusion rate (high coil): {exclude_coil_pct:.2f}%")

    # Save summary statistics
    summary_file = os.path.join(output_dir, "filtering_summary.txt")
    logger.info(f"Saving summary to: {summary_file}")

    with open(summary_file, "w") as f:
        f.write("=== Secondary Structure Filtering Summary ===\n\n")
        f.write("Dataset: pdb_multimer_binder_synthetic\n")
        f.write(f"Processing date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Coil threshold: {args.coil_threshold}\n")
        f.write(f"Number of workers: {args.num_workers}\n")
        f.write(f"Random seed: {args.seed}\n\n")

        f.write(f"Total samples processed: {successful_processes + failed_processes}\n")
        f.write(f"Successfully processed: {successful_processes}\n")
        f.write(f"Failed to process: {failed_processes}\n\n")

        f.write(f"Samples excluded for high coil (>{args.coil_threshold}): {len(excluded_high_coil_ids)}\n")
        f.write(f"Total excluded samples: {len(all_excluded_ids)}\n\n")

        if not df.empty:
            f.write("=== Secondary Structure Statistics ===\n")
            f.write(f"Mean coil fraction: {df['coil_fraction'].mean():.4f} ± {df['coil_fraction'].std():.4f}\n")
            f.write(f"Mean alpha fraction: {df['alpha_fraction'].mean():.4f} ± {df['alpha_fraction'].std():.4f}\n")
            f.write(f"Mean beta fraction: {df['beta_fraction'].mean():.4f} ± {df['beta_fraction'].std():.4f}\n")
            f.write(f"Mean length: {df['length'].mean():.1f} ± {df['length'].std():.1f}\n\n")

            exclude_coil_pct = (df["exclude_high_coil"].sum() / len(df)) * 100
            f.write(f"Exclusion rate (high coil): {exclude_coil_pct:.2f}%\n")

    logger.info("=== Secondary Structure Filtering Complete ===")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Secondary structure stats CSV: {ss_csv_file}")
    logger.info(f"Excluded IDs (high coil): {excluded_coil_file}")
    logger.info(f"All excluded IDs: {excluded_all_file}")
    logger.info(f"Summary report: {summary_file}")


if __name__ == "__main__":
    main()
