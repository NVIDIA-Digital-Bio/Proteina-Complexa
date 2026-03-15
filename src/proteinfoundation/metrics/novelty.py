# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import os
import random
import shutil
import subprocess
import time

from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

load_dotenv()


def _get_foldseek_path() -> str:
    """Returns path where foldseek databases are stored."""
    return os.path.join(os.getenv("DATA_PATH"), "foldseek_db")


def _get_db_paths(db_type: str = "pdb") -> tuple[str]:
    """
    Returns two paths, one to the root directory of the database, which contains all auxilairy files needed,
    and a second one to the actual database. Foldseek needs these two paths when searching or clustering,
    and we also need to check if both exist, and potentially create them.

    Args:
        db_type: Database, "pdb" and will add variant of "afdb".
    """
    foldseek_path = _get_foldseek_path()
    if db_type == "pdb":
        root_path_database = os.path.join(foldseek_path, "pdb_db")
        path_database = os.path.join(root_path_database, "pdb")
    elif db_type == "genie2":  # This is in the cluster, shared directory
        # afdb_rep_v4 filtered between 32 and 256 residues
        root_path_database = os.path.join(foldseek_path, "genie2_db")
        path_database = os.path.join(root_path_database, "genie2_db")
    elif db_type == "afdb_rep_v4":  # This is in the cluster, shared directory
        # All afdb_rep_v4, the foldcomp database (made into foldseek database)
        root_path_database = os.path.join(foldseek_path, "afdb_rep_v4_db")
        path_database = os.path.join(root_path_database, "afdb_rep_v4")
    elif db_type == "afdb_rep_v4_geniefilters_maxlen512":  # This is in the cluster, shared directory
        # All afdb_rep_v4 with genie filters up to length 512
        root_path_database = os.path.join(foldseek_path, "afdb_rep_v4_geniefilters_maxlen512")
        path_database = os.path.join(root_path_database, "afdb_rep_v4_geniefilters_maxlen512")

    else:
        raise OSError(f"Database type {db_type} incorrect")
    return root_path_database, path_database


def download_database(db_type: str = "pdb") -> None:
    """
    Downloads requested database into the appropriate path.

    Args:
        db_type: "pdb" for now. We'll extend to include "afdb" as well.
    """
    # Get necessary directories and checks if database already downloaded
    root_path_database, path_database = _get_db_paths(db_type=db_type)

    if os.path.exists(path_database):
        logger.info(f"Foldseek database {db_type} already downloaded {path_database}.")
        return
    else:
        logger.info(f"Creating directory {root_path_database}")
        os.makedirs(root_path_database)

    if db_type == "pdb":
        db_down = "PDB"
    elif db_type == "afdb":
        db_down = "Alphafold/UniProt50"
    else:
        raise OSError(f"Database type {db_type} incorrect")

    tmp_dir = "foldseek_tmp_download"
    foldseek_exec = os.getenv("FOLDSEEK_EXEC")

    # Check if Foldseek executable available
    if foldseek_exec is None:
        shutil.rmtree(path_database)
        shutil.rmtree(root_path_database)
        logger.info("No Foldseek executable available")
        return

    # Run download command
    command = [
        foldseek_exec,
        "databases",
        db_down,
        path_database,
        tmp_dir,
    ]  # Command to download database
    command_print = " ".join(command)
    logger.info(f"Downloading database {db_type} with command <{command_print}>")
    try:
        os.system(f"{foldseek_exec} databases {db_down} {path_database} {tmp_dir}")
    except Exception as e:
        print(f"Failed to download database {db_type} - {e}")

    # Remove tmp files created by foldseek
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)


def novelty(query_pdb_path: str, db_type: str, tmp_path: str, logging: bool = True) -> float:
    """
    Computes novelty for a given pdb file against a database.

    Args:
        query_pdb_path: Paths to the query PDBs we want to score.
        db_type: Database to compare against. So far only "pdb", we'll extend to a variant of "afdb".
        tmp_path: Path used to store temporary files created by foldseek.
        logging: Whether output should be logged or not.

    Returns:
        TM-PDB score of query (float).
    """
    time_start = time.time()

    # Create all tmp directories and necessary files
    tmp_path = os.path.join(tmp_path, str(random.randint(1, 1e12)))
    if os.path.exists(tmp_path):
        logger.error(f"Path {tmp_path} exists.")
    else:
        os.makedirs(tmp_path)
    out_file = os.path.join(tmp_path, query_pdb_path.split("/")[-1].replace(".pdb", ".txt"))
    db_path = _get_db_paths(db_type)[1]

    # Run foldseek
    foldseek_exec = os.getenv("FOLDSEEK_EXEC")
    if not foldseek_exec:
        logger.error("Foldseek executable not found. Please set the FOLDSEEK_EXEC environment variable.")
        return None
    command = [
        foldseek_exec,
        "easy-search",
        query_pdb_path,
        db_path,
        out_file,
        tmp_path,
        "--alignment-type",
        "1",
        "--format-output",
        # "query,target,alntmscore,qtmscore,ttmscore",
        "query,target,alntmscore",
        "--tmscore-threshold",
        "0.0",
        "--exhaustive-search",
        "--max-seqs",
        "10000000000",
        # "--threads",
        # "1",
        # "--gpu",
        # "1",
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    if logging:
        logger.info(f"Stderr of Foldseek command: {result.stderr}")
        logger.info(f"Took {time.time() - time_start} seconds")

    all_scores = []
    with open(out_file) as f:
        for line in f:
            if len(line.split()) > 2:
                aln_score = float(line.split()[2])
                all_scores.append(aln_score)
            else:
                logger.warning(f"{query_pdb_path} had error in novelty computation")

    # Remove tmp files created by foldseek
    shutil.rmtree(tmp_path)
    if len(all_scores) == 0:
        logger.warning(f"{query_pdb_path} returning None")
        return None
    return max(all_scores)  # aln score


# tqdm with multithread stuff obtained from https://stackoverflow.com/questions/51601756/use-tqdm-with-concurrent-futures
def novelty_from_list(query_pdb_list: list[str], db_type: str, tmp_path: str, num_workers: int = 32) -> list[float]:
    """
    Computes novelty of each pdb in query_pdb_list (lisg of paths) as the max TM score against
    a database.

    Args:
        query_pdb_list: List of paths to all PDBs we want to score.
        db_type: Database to compare against. So far only "pdb", we'll extend to a variant of "afdb".
        tmp_path: Path used to store temporary files created by foldseek, will be removed at end.
        num_workers: Number of CPUs used for this to parallelize over.

    Returns:
        List of TM-PDB scores (max tm against database) for each pdb file in query_pdb_list.
    """

    def _novelty_wrapper(aux):
        query_pdb_path, db_type, tmp_path = aux
        return novelty(query_pdb_path, db_type, tmp_path, logging=False)

    tmp_path_run = os.path.join(tmp_path, "novelty_run")
    aux = [(query_pdb_list[i], db_type, os.path.join(tmp_path_run, f"iter_{i}")) for i in range(len(query_pdb_list))]
    # with tqdm(total=len(aux)) as pbar:
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    #         futures = {
    #             executor.submit(_novelty_wrapper, arg): idx
    #             for (idx, arg) in enumerate(aux)
    #         }
    #         results = [None] * len(aux)
    #         for future in concurrent.futures.as_completed(futures):
    #             idx = futures[future]
    #             results[idx] = future.result()
    #             pbar.update(1)
    results = []
    for arg in tqdm(aux):
        results.append(_novelty_wrapper(arg))
    # shutil.rmtree(tmp_path)
    return results


if __name__ == "__main__":
    load_dotenv()
    download_database("pdb")
    # download_database("afdb")
