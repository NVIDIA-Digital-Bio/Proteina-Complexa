"""
Multimer evaluation metrics.
"""

import os

import pandas as pd
from biotite.structure.io import load_structure
from loguru import logger
from omegaconf import DictConfig

from proteinfoundation.evaluation.utils import parse_cfg_for_table
from proteinfoundation.metrics.designability import run_multimer_eval


def compute_multimer_metrics(
    cfg: DictConfig,
    cfg_metric: DictConfig,
    sample_root_paths: list[str],
    job_id: int,
) -> pd.DataFrame:
    """
    Computes metrics for multimer structures.

    Args:
        cfg: Full configuration
        cfg_metric: Metric configuration
        sample_root_paths: List of directory paths containing PDB files
        job_id: Job ID for this evaluation

    Returns:
        DataFrame with computed metrics including:
            - L: Total length
            - n_chains: Number of chains
            - pLDDT: Best predicted LDDT
            - _res_scRMSD: Best self-consistency RMSD
    """
    columns, flat_dict = parse_cfg_for_table(cfg)
    columns += ["id_gen", "pdb_path", "L", "n_chains", "_res_pLDDT", "_res_pLDDT_all"]
    columns += ["_res_scRMSD", "_res_scRMSD_all"]
    results = []

    for i, sample_root_path in enumerate(sample_root_paths):
        fname = os.path.basename(sample_root_path) + ".pdb"
        pdb_path = os.path.join(sample_root_path, fname)

        try:
            structure = load_structure(pdb_path)
            n_chains = len(set(structure.chain_id.tolist()))
            # Try to extract length from filename, fallback to structure length
            try:
                n = int(fname.split("_")[3])
            except (IndexError, ValueError):
                n = len(structure)
        except Exception as e:
            logger.error(f"Failed to load structure {pdb_path}: {e}")
            continue

        row_dict = {
            **flat_dict,
            "id_gen": i,
            "pdb_path": pdb_path,
            "L": n,
            "n_chains": n_chains,
        }

        try:
            pdb_paths, stats, rmsd_results = run_multimer_eval(
                pdb_file_path=pdb_path,
                tmp_path=sample_root_path,
                folding_method=cfg_metric.get("folding_method", "esmfold"),
            )

            best_rmsd = float("inf")
            best_seq = None
            for seq_num, rmsd_result in enumerate(rmsd_results):
                for seq_name, rmsd in rmsd_result.items():
                    if rmsd < best_rmsd:
                        best_rmsd = rmsd
                        best_seq = seq_name

            if best_seq is not None:
                best_pLDDT = stats[int(best_seq.split("_")[1]) - 1][best_seq]["pLDDT"]
                row_dict.update({"_res_pLDDT": best_pLDDT, "_res_scRMSD": best_rmsd})
                row_dict.update(
                    {
                        "_res_pLDDT_all": [stat[f"seq_{i + 1}"]["pLDDT"] for i, stat in enumerate(stats)],
                        "_res_scRMSD_all": [list(result.values())[0] for result in rmsd_results],
                    }
                )
            else:
                row_dict.update(
                    {
                        "_res_pLDDT": float("inf"),
                        "_res_scRMSD": float("inf"),
                        "_res_pLDDT_all": [],
                        "_res_scRMSD_all": [],
                    }
                )
        except Exception as e:
            logger.error(f"Failed multimer evaluation for {pdb_path}: {e}")
            row_dict.update(
                {
                    "_res_pLDDT": float("inf"),
                    "_res_scRMSD": float("inf"),
                    "_res_pLDDT_all": [],
                    "_res_scRMSD_all": [],
                }
            )

        results.append(row_dict)

    df = pd.DataFrame(results)
    df = df.reindex(columns=columns)
    return df
