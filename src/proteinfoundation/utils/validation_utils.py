"""
Validation utilities for protein generation metrics and file management.
Extracted from Proteina class for separation of concerns.
"""

import os
import re
import shutil
import tempfile

import numpy as np

from proteinfoundation.metrics.novelty import novelty_from_list
from proteinfoundation.metrics.structural_metric_ss_ca_ca import compute_structural_metrics


def get_pdb_novelty_metric(
    paths: list[str],
    val_mode: str,
    ncpus: int,
) -> dict[str, float]:
    """
    Compute PDB novelty metric for a list of PDB paths.

    Args:
        paths: List of paths to PDB files
        val_mode: Validation mode name for metric keys
        ncpus: Number of workers for novelty computation

    Returns:
        Dict with keys for logging: 'nfail' and optionally 'pdb_novelty_mean'.
        Caller should log these with prefix like f"validation_nov/..._{val_mode}".
    """
    results = {}
    tmp_path = tempfile.mkdtemp(prefix="novelty_")
    try:
        pdb_novelty_metric = novelty_from_list(
            query_pdb_list=paths,
            db_type="pdb",
            tmp_path=tmp_path,
            num_workers=ncpus,
        )
        pdb_novelty_metric_good = [v for v in pdb_novelty_metric if v is not None]
        total_count = len(pdb_novelty_metric)
        fail_count = total_count - len(pdb_novelty_metric_good)

        results[f"nfail_pdb_novelty_{val_mode}"] = float(fail_count)
        if len(pdb_novelty_metric_good) > 0:
            results[f"pdb_novelty_{val_mode}"] = float(np.array(pdb_novelty_metric_good).mean())
    except Exception:
        raise
    finally:
        try:
            shutil.rmtree(tmp_path)
        except Exception:
            pass
    return results


def get_structural_metrics(paths: list[str], val_mode: str) -> dict[str, float]:
    """
    Compute structural metrics (secondary structure, etc.) for a list of PDB paths.

    Args:
        paths: List of paths to PDB files
        val_mode: Validation mode name for metric keys

    Returns:
        Dict mapping metric names to mean values. Keys use prefixes:
        - "validation_sanity" for most metrics
        - "validation_ss" for metrics containing "biot" (secondary structure)
    """
    props = []
    for pdb_path in paths:
        ss_metrics = compute_structural_metrics(pdb_path)
        props.append(ss_metrics)

    results = {}
    for k in props[0]:
        prefix = "validation_ss" if "biot" in k else "validation_sanity"
        key = f"{prefix}/{k}_{val_mode}"
        results[key] = float(np.array([p[k] for p in props]).mean())
    return results


def clean_validation_files(paths: list[str], limit: int = 2) -> None:
    """
    Remove most generated validation files, keeping a subset for visualization.

    Keeps files where num < limit and rank <= 2.

    Args:
        paths: List of file paths to clean up
        limit: Keep files with num < limit (from filename pattern _num_N_)
    """
    from loguru import logger

    pattern = re.compile(r".*_num_(\d+)_rank_(\d+)")
    for fname in paths:
        match = pattern.match(fname)
        if match:
            i = int(match.group(1))
            rank = int(match.group(2))
            if i >= limit or rank > 2:
                try:
                    os.remove(fname)
                except Exception:
                    logger.info("Could not remove validation file")
