import os
import pathlib
import shutil
import subprocess

import numpy as np
import pandas as pd
from biotite.structure.io import load_structure, save_structure
from dotenv import load_dotenv
from loguru import logger

from proteinfoundation.utils.cluster_utils import cluster_sequences, df_to_fasta
from proteinfoundation.utils.pdb_utils import extract_seq_from_pdb

load_dotenv()


def diversity_foldseek(
    list_of_pdb_paths: list[str],
    tm_threshold: float = 0.5,
    tmp_path: str = "./tmp/metrics/",
    clean_tmp: bool = True,
    chains: list[str] | None = None,
    multimer_tm_threshold: float = 0.65,
    interface_lddt_threshold: float = 0.65,
    alignment_type: int = 1,
    min_seq_id: float = 0.0,
    save_cluster_file: bool = False,
    cluster_output_dir: str | None = None,
    inf_config_name: str | None = None,
) -> tuple[float, float, float] | None:
    """
    Evaluates diversity by clustering designable samples using foldseek. Diversity is measured as
    the number of clusters over number of samples. Supports both monomers and multimers.

    Args:
        list_of_pdb_paths: List of paths (strings) each one to a pdb file.
        tm_threshold: Threshold used to cluster with tm score (for monomers and chain-level in multimers)
        tmp_path: Path to store temp files during clustering
        clean_tmp: Whether to delete temp files created by the clustering after it is done
        chains: List of chain ids to use for clustering. If None, all chains are used.
        multimer_tm_threshold: TM threshold for multimer clustering (Foldseek multimer mode)
        interface_lddt_threshold: Interface lDDT threshold for multimer clustering
        alignment_type: 1 (structure) or 2 (structure+sequence), passed to Foldseek
        min_seq_id: Minimum sequence identity for clustering (monomer mode, alignment_type=2)
        save_cluster_file: Whether to save the cluster TSV file for later analysis
        cluster_output_dir: Directory to save cluster files (if save_cluster_file=True)

    Returns:
        Tuple[float, float, float], containing the diversity score, # clusters, # samples.
    """
    assert alignment_type in [
        1,
        2,
    ], f"alignment_type {alignment_type} not valid, should be either 1 or 2"
    if alignment_type == 1:
        assert min_seq_id == 0.0, f"alignment_type == 1 only admits min_seq_id = 0, but {min_seq_id} was given"
    if alignment_type == 2:
        assert min_seq_id > 0.0, f"alignment_type == 2 requires min_seq_id > 0, but {min_seq_id} was given"
    if len(list_of_pdb_paths) == 0:
        return None

    path_tmp = os.path.join(tmp_path, "diversity")
    if not os.path.exists(path_tmp):
        os.makedirs(path_tmp)

    path_designable = os.path.join(path_tmp, "samples")
    if os.path.exists(path_designable):
        shutil.rmtree(path_designable)
    os.makedirs(path_designable, exist_ok=False)

    contain_multimers = False
    for i, fpdb in enumerate(list_of_pdb_paths):
        dest_f = os.path.join(path_designable, f"{i + 1}.pdb")
        if not contain_multimers:
            pdb = load_structure(fpdb)
            if len(np.unique(pdb.chain_id)) > 1:
                contain_multimers = True

        if chains is not None:
            pdb = load_structure(fpdb)
            struct = pdb[np.isin(pdb.chain_id, chains)]
            save_structure(dest_f, struct)
        else:
            shutil.copy(fpdb, dest_f)

    foldseek_exec = os.getenv("FOLDSEEK_EXEC")
    if not foldseek_exec:
        logger.error("Foldseek executable not found. Please set the FOLDSEEK_EXEC environment variable.")
        return None
    if contain_multimers:
        # Interface clustering
        logger.info(f"Clustering {len(list_of_pdb_paths)} structures with Foldseek multimercluster")
        command = [
            foldseek_exec,
            "easy-multimercluster",
            path_designable,
            os.path.join(path_tmp, "res"),
            path_tmp,
            "--alignment-type",
            str(alignment_type),
            "--cov-mode",
            "0",
            "--multimer-tm-threshold",
            f"{multimer_tm_threshold}",
            "--chain-tm-threshold",
            f"{tm_threshold}",
            "--interface-lddt-threshold",
            f"{interface_lddt_threshold}",
        ]
    else:
        logger.info(f"Clustering {len(list_of_pdb_paths)} structures with Foldseek easy-cluster")
        command = [
            foldseek_exec,
            "easy-cluster",
            path_designable,
            os.path.join(path_tmp, "res"),
            path_tmp,
            "--alignment-type",
            str(alignment_type),
            "--cov-mode",
            "0",
            "--min-seq-id",
            f"{min_seq_id}",
            "--tmscore-threshold",
            f"{tm_threshold}",
        ]

    try:
        subprocess.run(command, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        logger.error(
            f"Foldseek diversity failed (exit code {e.returncode}):\n"
            f"  stdout: {e.stdout}\n"
            f"  stderr: {e.stderr}\n"
            f"  Returning None"
        )
    except Exception as e:
        logger.error(f"Foldseek diversity failed: {e} \n Returning None")
        if clean_tmp:
            shutil.rmtree(path_tmp)
        return None

    res_file = os.path.join(path_tmp, "res_cluster.tsv")
    with open(res_file) as f:
        reps, els = set(), set()
        for line in f:
            line = line.strip()
            if line:
                words = line.split()
                if len(words) == 2:
                    word1, word2 = words
                    reps.add(word1)
                    els.add(word2)
                else:
                    logger.error(f"Strange line parsing Foldseek clustering results: {line}")
    nclus = len(reps)
    nels = len(els)

    assert nels == len(list_of_pdb_paths), f"Foldseek clustering clustered {nels} from {len(list_of_pdb_paths)} given"

    # Save cluster file and essential files if requested
    if save_cluster_file and cluster_output_dir is not None:
        os.makedirs(cluster_output_dir, exist_ok=True)

        # Save the cluster file
        if inf_config_name is None or inf_config_name == "":
            cluster_save_path = os.path.join(cluster_output_dir, "res_cluster.tsv")
        else:
            cluster_save_path = os.path.join(cluster_output_dir, f"res_cluster_{inf_config_name}.tsv")
        # cluster_save_path = os.path.join(cluster_output_dir, "res_cluster.tsv")
        shutil.copy2(res_file, cluster_save_path)
        logger.info(f"Cluster file saved to {cluster_save_path}")

        # Save the PDB paths used for clustering
        if inf_config_name is None or inf_config_name == "":
            pdb_paths_file = os.path.join(cluster_output_dir, "original_pdb_paths.txt")
        else:
            pdb_paths_file = os.path.join(cluster_output_dir, f"original_pdb_paths_{inf_config_name}.txt")
        # pdb_paths_file = os.path.join(cluster_output_dir, "original_pdb_paths.txt")
        with open(pdb_paths_file, "w") as f:
            for i, pdb_path in enumerate(list_of_pdb_paths, 1):
                f.write(f"{i}\t{pdb_path}\n")
        logger.info(f"PDB paths saved to {pdb_paths_file}")

        # Create combined CSV by reading from saved txt and tsv files
        # Read sample_index -> path from the saved txt file
        sample_to_path = {}
        with open(pdb_paths_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    sample_idx, path = parts
                    sample_to_path[sample_idx] = path

        # Read member_id -> cluster_rep from saved tsv file
        member_to_cluster_rep = {}
        with open(cluster_save_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    rep_id, member_id = parts
                    member_to_cluster_rep[member_id] = rep_id

        # Assign cluster indices based on unique reps (preserve order of first appearance)
        unique_reps = list(dict.fromkeys(member_to_cluster_rep.values()))
        rep_to_cluster_idx = {rep: idx for idx, rep in enumerate(unique_reps)}

        # Build cluster data by matching sample_index to member_id in tsv
        cluster_data = []
        for sample_idx, path in sample_to_path.items():
            if sample_idx in member_to_cluster_rep:
                rep = member_to_cluster_rep[sample_idx]
                cluster_idx = rep_to_cluster_idx[rep]
                cluster_data.append((cluster_idx, int(sample_idx), path))

        # Sort by cluster_index, then sample_index
        cluster_data.sort(key=lambda x: (x[0], x[1]))

        # Write combined CSV
        if inf_config_name is None or inf_config_name == "":
            combined_csv_path = os.path.join(cluster_output_dir, "cluster_assignments.csv")
        else:
            combined_csv_path = os.path.join(cluster_output_dir, f"cluster_assignments_{inf_config_name}.csv")

        with open(combined_csv_path, "w") as f:
            f.write("cluster_index,sample_index,path_name\n")
            for cluster_idx, sample_idx, path in cluster_data:
                f.write(f"{cluster_idx},{sample_idx},{path}\n")
        logger.info(f"Combined cluster assignments saved to {combined_csv_path}")

    if clean_tmp:
        shutil.rmtree(path_tmp)

    return (nclus / nels, nclus, nels)


def diversity_sequence_mmseqs(
    list_of_pdb_paths: list[str],
    min_seq_id: float = 0.3,
    coverage: float = 0.8,
    silence_mmseqs_output: bool = True,
    efficient_linclust: bool = False,
    tmp_path: str = "./tmp/metrics",
    clean_tmp: bool = True,
) -> tuple[float, float, float]:
    """
    Cluster protein sequences using MMseqs2 and returns diversity, number of cluster, number of samples.

    Args:
        list_of_pdb_paths: List of paths (strings) each one to a pdb file.
        min_seq_id (float): Minimum sequence identity for clustering. Defaults to 0.3.
        coverage (float): Minimum coverage for clustering. Defaults to 0.8.
        silence_mmseqs_output (bool): Whether to silence MMseqs2 output. Defaults to True.
        efficient_linclust (bool): Whether to use efficient linclust for clustering for large datasets. Defaults to False.
        tmp_path (str): Path to store temp files during clustering. Defaults to "./tmp/metrics".
        clean_tmp (bool): Whether to delete temp files created by the clustering after it is done. Defaults to True.
    """
    mmseqs_exec = os.getenv("MMSEQS_EXEC")  # None if it does not exist

    path_tmp = os.path.join(tmp_path, "diversity_mmseqs")
    if os.path.exists(path_tmp):
        shutil.rmtree(path_tmp)
    if not os.path.exists(path_tmp):
        os.makedirs(path_tmp, exist_ok=False)

    # Get dataframe from list of PDBs
    sample_ids = []
    sequences = []
    for i, pdb_file in enumerate(list_of_pdb_paths):
        sample_ids.append(i)
        sequences.append(extract_seq_from_pdb(pdb_file))
    data = {"id": sample_ids, "sequence": sequences}
    df = pd.DataFrame(data)

    # Save fasta file that will be the input to mmseqs
    fasta_file = os.path.join(path_tmp, "sequences.fasta")
    df_to_fasta(df, fasta_file)

    # Cluster sequences
    cluster_out_file = os.path.join(path_tmp, "cluster_out.fasta")
    cluster_sequences(
        fasta_input_filepath=fasta_file,
        cluster_output_filepath=cluster_out_file,
        min_seq_id=min_seq_id,
        coverage=coverage,
        overwrite=False,
        silence_mmseqs_output=silence_mmseqs_output,
        efficient_linclust=efficient_linclust,
        mmseqs_exec=mmseqs_exec,
    )
    cluster_fasta_path = pathlib.Path(cluster_out_file)
    cluster_tsv_path = cluster_fasta_path.with_suffix(".tsv")

    # Count number of clusters and samples
    nsamples_orig = len(list_of_pdb_paths)
    line_count = 0
    unique_clusters = set()

    with open(cluster_tsv_path) as f:
        for line in f:
            elements = line.strip().split()
            if elements:  # Checking if the line is not empty
                line_count += 1
                unique_clusters.add(int(elements[0]))  # Add the first element to the set

    assert line_count == nsamples_orig, (
        f"Number of samples in the cluster file ({line_count}) does not match the number of samples in the input ({nsamples_orig})"
    )
    nels = nsamples_orig
    nclus = len(unique_clusters)

    if clean_tmp:
        shutil.rmtree(path_tmp)

    return (nclus / nels, nclus, nels)


# # Leaving this here just in case it is needed at some point, diversity with MaxCluster
# def diversity_maxcluster(
#     list_of_pdb_paths: List[str],
#     tm_threshold: int = 0.5,
#     tmp_path: str = "./tmp/metrics/",
#     path_to_exec: str = "./maxcluster"
# ) -> Tuple[float, float, float]:
#     """
#     Evaluates diversity.

#     Args:
#         list_of_pdb_paths: List of paths (strings) each one to a pdb file.
#         path_to_exec: Path to MaxCluster executable

#     Returns:
#         Tuple[floar, float, float], containing the diversity score, # clusters, # samples.
#     """
#     path_tmp = os.path.join(tmp_path, "diversity")
#     if not os.path.exists(path_tmp):
#         os.makedirs(path_tmp)

#     # Create file listing paths of all generated pdbs
#     path_to_pdb_list = os.path.join(path_tmp, "pdb.list")
#     with open(path_to_pdb_list, "w") as f:
#         for pdb_path in list_of_pdb_paths:
#             f.write(pdb_path)
#             f.write("\n")

#     command = [path_to_exec, "-l", path_to_pdb_list, "./all_by_all_lite", "-C", "2", "-in", "-Rl", "./tm_results.txt", "-Tm", str(tm_threshold)]
#     result = subprocess.run(command, check=True, text=True, capture_output=True)
#     logger.info(f"Stderr of command: {result.stderr}")

#     # Recover number of clusters and number of samples
#     # (We verify # samples is the same as # PDBs later on)
#     pattern_nchains = r"INFO\s+:\s+Successfully read (\d+) Chain structures"
#     pattern_nclus = r"INFO\s+:\s+(\d+) Clusters @ Threshold\s+(\d+\.\d+|\d+)\s+\((\d+\.\d+|\d+)\)"
#     match_nchains = re.search(pattern_nchains, result.stdout)
#     match_nclus = re.search(pattern_nclus, result.stdout)

#     if match_nchains:
#         nchains = int(match_nchains.group(1))
#         found_nchains = True

#     if match_nclus:
#         nclus = int(match_nclus.group(1))
#         thres_1 = float(match_nclus.group(2))
#         thres_2 = float(match_nclus.group(3))
#         found_nclus = True

#     # Verify whether things worked as expected
#     if not found_nchains or not found_nclus:
#         raise IOError(f"Parsing of MaxCluster output failed - {found_nchains} - {found_nclus}")
#     if len(list_of_pdb_paths) != nchains:
#         raise IOError(f"Number of generations {len(list_of_pdb_paths)} des not correspond to number of chains processed by MaxCluster {nchains}")

#     logger.info(f"MaxCluster operated successfully")
#     logger.info(f"Read {nchains} structures and identified {nclus} clusters (threshold {thres_1} {thres_2})")
#     return (nclus / nchains, nclus, nchains)
