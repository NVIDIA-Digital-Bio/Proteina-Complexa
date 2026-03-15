# inspired from https://github.com/a-r-j/graphein/blob/master/graphein/ml/datasets/pdb_data.py

import math
import os
import pathlib
import random
import shutil
import subprocess
import tempfile
from multiprocessing import Pool, cpu_count

import modin.pandas as mpd
import pandas as pd
import torch
from graphein.utils.dependencies import is_tool
from lightning.pytorch.utilities import rank_zero_only
from loguru import logger
from torch.utils.data import Sampler
from tqdm import tqdm


@rank_zero_only
def log_info(msg):
    logger.info(msg)


class ClusterSampler(Sampler):
    def __init__(
        self,
        dataset,
        clusterid_to_seqid_mapping,
        sampling_mode="cluster-random",
        shuffle=True,
        drop_last=False,
        dimer_mode=False,
    ):
        """
        Args:
            dataset: torch_geometric.data.Dataset or FoldCompDataset-like
            clusterid_to_seqid_mapping: dict {cluster_id: [seq_ids]}
            sampling_mode: 'cluster-random' or 'cluster-reps'
            shuffle: shuffle clusters each epoch
            drop_last: drop incomplete batches in distributed mode
            dimer_mode: if True, yield (index, full_dimer_id) tuples:
                        index is for dataset lookup, full_dimer_id for transforms
        """
        self.dataset = dataset
        self.clusterid_to_seqid_mapping = clusterid_to_seqid_mapping
        self.cluster_names = list(clusterid_to_seqid_mapping.keys())
        self.sampling_mode = sampling_mode
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.dimer_mode = dimer_mode
        self.len = None

        # Map from monomer ID -> dataset index
        if getattr(dataset, "database", None) in (
            "pdb",
            "scop",
            "boltz_refold_pdb_multimer",
        ):
            self.sequence_id_to_idx = {fname.split(".")[0]: i for i, fname in enumerate(dataset.file_names)}
        elif getattr(dataset, "database", None) == "pinder":
            self.sequence_id_to_idx = dataset.pinder_id_to_idx
        else:
            # FoldCompDataset or similar
            self.sequence_id_to_idx = dataset.protein_to_idx

        self.log_clusters = True

    def __iter__(self):
        """Iterate over clusters according to the sampling mode."""
        # Setup distributed context if any
        if torch.distributed.is_initialized():
            num_replicas = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            num_replicas = None
            rank = 0

        # Shuffle cluster order if needed
        indices = list(range(len(self.cluster_names)))
        if self.shuffle:
            random.shuffle(indices)

        if num_replicas is not None:
            # Partition indices across ranks
            num_samples = math.ceil(len(indices) / num_replicas)
            total_size = num_samples * num_replicas

            # pad or truncate
            if self.drop_last:
                indices = indices[:total_size]
            else:
                if total_size > len(indices):
                    indices += indices[: (total_size - len(indices))]

            # subsample this rank
            indices = indices[rank:total_size:num_replicas]
            self.len = len(indices)
        # else: non-distributed — already shuffled if requested

        # Main sampling loop
        for cluster_idx in indices:
            cluster_name = self.cluster_names[cluster_idx]

            if self.sampling_mode == "cluster-reps":
                # Representative = cluster_name itself
                seq_id = cluster_name
            elif self.sampling_mode == "cluster-random":
                seq_id = random.choice(self.clusterid_to_seqid_mapping[cluster_name])
                if self.log_clusters:
                    print(f"[ClusterSampler] First cluster: chose {seq_id} from {cluster_name}")
                    self.log_clusters = False
            else:
                raise ValueError(f"Unknown sampling_mode {self.sampling_mode}")

            if self.dimer_mode:
                # seq_id is full dimer like DIxxxxx_AF-...
                full_dimer_id = seq_id
                monomer_id = full_dimer_id.split("_", 1)[1]  # strip DI prefix
                try:
                    ds_index = self.sequence_id_to_idx[monomer_id]
                except KeyError:
                    raise KeyError(f"Monomer ID {monomer_id} not found in dataset index map")
                yield (ds_index, full_dimer_id)
            else:
                # seq_id should match exactly the dataset index key
                try:
                    ds_index = self.sequence_id_to_idx[seq_id]
                except KeyError:
                    raise KeyError(f"Sequence ID {seq_id} not found in dataset index map")
                yield ds_index

    def __len__(self):
        if self.len is None:
            return len(self.cluster_names)
        else:
            return self.len


def split_dataframe(
    df: pd.DataFrame,
    splits: list[str],
    ratios: list[float],
    leftover_split: int = 0,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Split a DataFrame into multiple parts based on specified split ratios.

    Args:
        df (pd.DataFrame): The DataFrame to split.
        splits (List[str]): Names of the resulting splits.
        ratios (List[float]): Ratios to split df into. Must sum to 1.0.
        leftover_split (int): Index of split to assign leftover rows to.
            Defaults to 0.
        seed (int): Random seed for shuffling. Defaults to 42.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping split names to
            DataFrame splits.

    Raises:
        AssertionError: If len(splits) != len(ratios) or sum(ratios) != 1.
    """
    assert len(splits) == len(ratios), "Number of splits must equal number of ratios"
    assert sum(ratios) == 1, "Split ratios must sum to 1"

    # Calculate size of each split
    split_sizes = [int(len(df) * ratio) for ratio in ratios]

    # Assign leftover rows to specified split
    split_sizes[leftover_split] += len(df) - sum(split_sizes)

    # Shuffle DataFrame rows
    df = df.sample(frac=1, random_state=seed)

    # Split DataFrame into parts
    split_dfs = {}
    start = 0
    for split, size in zip(splits, split_sizes, strict=False):
        split_dfs[split] = df.iloc[start : start + size]
        start += size

    return split_dfs


def merge_dataframe_splits(df1: pd.DataFrame, df2: pd.DataFrame, list_columns: list[str]) -> pd.DataFrame:
    """
    Merge two DataFrame splits on all columns except 'split'.

    Args:
        df1 (pd.DataFrame): First DataFrame split to merge.
        df2 (pd.DataFrame): Second DataFrame split to merge.
        list_columns (List[str]): Columns containing lists to convert to tuples.

    Returns:
        pd.DataFrame: Merged DataFrame containing rows in both splits.
    """
    # Convert list columns to tuples for merging
    for df in [df1, df2]:
        for col in list_columns:
            if col in df.columns:
                df[col] = df[col].apply(tuple)

    # Merge the two DataFrames
    merge_cols = [c for c in df1.columns if c != "split"]
    merged_df = pd.merge(df1, df2, on=merge_cols, how="inner")

    # Convert tuple columns back to lists
    for df in [df1, df2]:
        for col in list_columns:
            if col in df.columns:
                df[col] = df[col].apply(list)

    return merged_df


def cluster_sequences(
    fasta_input_filepath: str,
    cluster_output_filepath: str = None,
    min_seq_id: float = 0.3,
    coverage: float = 0.8,
    overwrite: bool = False,
    silence_mmseqs_output: bool = True,
    efficient_linclust: bool = False,
    mmseqs_exec: str = None,
) -> None:
    """
    Cluster protein sequences in a DataFrame using MMseqs2.

    Args:
        fasta_input_file (str): Fasta File path containing protein sequences.
        cluster_output_filepath (str): Path to write clustering results. If None, defaults to
            "cluster_rep_seq_id_{min_seq_id}_c_{coverage}.fasta".
        min_seq_id (float): Minimum sequence identity for clustering. Defaults to 0.3.
        coverage (float): Minimum coverage for clustering. Defaults to 0.8.
        overwrite (bool): Whether to overwrite existing cluster file. Defaults to False.
        silence_mmseqs_output (bool): Whether to silence MMseqs2 output. Defaults to True.
        efficient_linclust (bool): Whether to use efficient linclust for clustering for large datasets. Defaults to False.
        mmseqs_exec (str): Path to the mmseqs2 executable. Defaults to None. If not provided, the function will check if mmseqs2 is installed.
    """
    if cluster_output_filepath is None:
        cluster_output_filepath = f"cluster_rep_seq_id_{min_seq_id}_c_{coverage}.fasta"

    cluster_fasta_path = pathlib.Path(cluster_output_filepath)
    cluster_tsv_path = cluster_fasta_path.with_suffix(".tsv")

    if not cluster_fasta_path.exists() or overwrite:
        # Remove existing file if overwriting
        if cluster_fasta_path.exists() and overwrite:
            cluster_fasta_path.unlink()

    if not cluster_tsv_path.exists() or overwrite:
        # Remove existing file if overwriting
        if cluster_tsv_path.exists() and overwrite:
            cluster_tsv_path.unlink()

        # Run MMseqs2 clustering
        if mmseqs_exec is None and not is_tool("mmseqs"):
            logger.error("MMseqs2 not found. Set MMSEQS_EXEC in .env or ensure mmseqs is on PATH.")

        mmseqs_exec = "mmseqs" if mmseqs_exec is None else mmseqs_exec
        if efficient_linclust:  # use efficient linclust algorithm that cales linearly with input size
            cmd = f"{mmseqs_exec} easy-linclust {fasta_input_filepath} pdb_cluster tmp --min-seq-id {min_seq_id} -c {coverage} --cov-mode 1"
        else:  # use standard cascaded clustering algorithm
            cmd = f"{mmseqs_exec} easy-cluster {fasta_input_filepath} pdb_cluster tmp --min-seq-id {min_seq_id} -c {coverage} --cov-mode 1"
        if silence_mmseqs_output:
            subprocess.run(cmd.split(), stdout=subprocess.DEVNULL)
        else:
            subprocess.run(cmd.split())
        # Rename output file
        shutil.move("pdb_cluster_rep_seq.fasta", cluster_fasta_path)
        shutil.move("pdb_cluster_cluster.tsv", cluster_tsv_path)


def split_sequence_clusters(df, splits, ratios, leftover_split=0, seed=42) -> dict[str, pd.DataFrame]:
    """
    Split clustered sequences into train/val/test sets.

    Args:
        df (pd.DataFrame): DataFrame with clustered sequences.
        splits (List[str]): Names of splits, e.g. ["train", "val", "test"].
        ratios (List[float]): Ratios for each split. Must sum to 1.0.
        leftover_split (int): Index of split to assign leftover sequences.
            Defaults to 0.
        seed (int): Random seed. Defaults to 42.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping split names to DataFrames that contain randomly-split representative sequences.
    """
    # Split clusters into subsets
    cluster_splits = split_dataframe(df, splits, ratios, leftover_split, seed)
    # Get representative sequences for each split
    split_dfs = {}
    for split, cluster_df in cluster_splits.items():
        rep_seqs = cluster_df.representative_sequences()
        split_dfs[split] = rep_seqs

    return split_dfs


def expand_cluster_splits(
    cluster_rep_splits: dict[str, pd.DataFrame],
    clusterid_to_seqid_mapping: dict[str, list[str]],
    use_modin: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Expand the cluster representative splits to full cluster splits based on the provided cluster dictionary.

    Args:
        cluster_rep_splits: A dictionary containing DataFrames for each split (e.g., 'train', 'val', 'test').
            Each DataFrame should have an 'id' column representing the cluster representative IDs.
        clusterid_to_seqid_mapping: A dictionary mapping cluster representative IDs to their corresponding cluster member IDs.
        use_modin (bool): Whether to use Modin for dataframe operations, useful for big datasets. Defaults to False.

    Returns:
        A new dictionary of DataFrames with expanded 'id' columns based on the cluster dictionary.
        The 'id' column in the original DataFrames is replaced with the corresponding cluster member IDs.
        If df_sequences is provided, the additional columns from df_sequences are added to the resulting DataFrames.

    """
    full_cluster_splits = {}
    split_clusterid_to_seqid_mapping = {}

    for split_name, split_df in cluster_rep_splits.items():
        # Create a dictionary to store the cluster members for the current split
        split_cluster_members = {}

        for rep_id in split_df["id"]:
            if rep_id in clusterid_to_seqid_mapping:
                split_cluster_members[rep_id] = clusterid_to_seqid_mapping[rep_id]
            else:
                logger.warning(f"ID {rep_id} is a representative in the splits, but not in the cluster_dicts")

        # Create a DataFrame with the cluster representative IDs and their corresponding cluster member IDs for the current split
        if use_modin:
            split_cluster_members_df = mpd.DataFrame(
                [
                    (rep_id, member_id)
                    for rep_id, member_ids in split_cluster_members.items()
                    for member_id in member_ids
                ],
                columns=["cluster_id", "id"],
            )
        else:
            split_cluster_members_df = pd.DataFrame(
                [
                    (rep_id, member_id)
                    for rep_id, member_ids in split_cluster_members.items()
                    for member_id in member_ids
                ],
                columns=["cluster_id", "id"],
            )
        # Split the 'id' column into 'pdb' and 'chain' columns
        if len(split_cluster_members_df) > 0:
            split_cluster_members_df[["pdb", "chain"]] = split_cluster_members_df["id"].str.split("_", n=1, expand=True)
        # Add the expanded DataFrame to the dictionary
        full_cluster_splits[split_name] = split_cluster_members_df
        # Add the split-specific cluster_dict to the dictionary
        split_clusterid_to_seqid_mapping[split_name] = split_cluster_members
    return full_cluster_splits, split_clusterid_to_seqid_mapping


def read_cluster_tsv(cluster_tsv_filepath: pathlib.Path) -> dict[str, list[str]]:
    """
    Read the cluster TSV file that is output from mmseqs2 and construct a dictionary mapping cluster representatives to sequence IDs.

    Args:
        cluster_tsv_filepath (pathlib.Path): The path to the cluster TSV file.

    Returns:
        Dict[str, List[str]]: A dictionary mapping cluster representatives to lists of sequence IDs.
    """
    cluster_dict = {}
    with open(cluster_tsv_filepath) as file:
        for line in file:
            cluster_name, sequence_name = line.strip().split("\t")
            cluster_dict.setdefault(cluster_name, []).append(sequence_name)
    return cluster_dict


def setup_clustering_file_paths(
    data_dir: str,
    file_identifier: str,
    split_sequence_similarity: float,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """
    Set up file paths for the fasta file, cluster file, and cluster TSV file.

    Args:
        data_dir (str): The directory where the files will be stored.
        file_identifier (str): The identifier used to name the files.
        split_sequence_similarity (float): The sequence similarity threshold for splitting.

    Returns:
        Tuple[pathlib.Path, pathlib.Path, pathlib.Path]: A tuple containing the file paths for
            the input fasta file, cluster file, and cluster TSV file.
    """
    input_fasta_filepath = pathlib.Path(data_dir) / f"seq_{file_identifier}.fasta"
    cluster_filepath = (
        pathlib.Path(data_dir) / f"cluster_seqid_{split_sequence_similarity}_{file_identifier}_test.fasta"
    )
    cluster_tsv_filepath = cluster_filepath.with_suffix(".tsv")
    return input_fasta_filepath, cluster_filepath, cluster_tsv_filepath


def df_to_fasta(df: pd.DataFrame, output_file: str) -> None:
    """
    Convert a pandas DataFrame to a FASTA file.

    Args:
        df (pd.DataFrame): DataFrame containing 'id' and 'sequence' columns.
        output_file (str): Path to the output FASTA file.

    Returns:
        None
    """
    with open(output_file, "w") as f:
        for _, row in df.iterrows():
            f.write(f">{row['id']}\n{row['sequence']}\n")


def fasta_to_df(fasta_input_file: str, use_modin: bool = False) -> pd.DataFrame:
    """
    Convert a FASTA file to a pandas DataFrame.

    Args:
        fasta_input_file (str): Path to the input FASTA file.
        use_modin (bool): Whether to use Modin DataFrame or pandas DataFrame. Defaults to False (pandas).

    Returns:
        pd.DataFrame: DataFrame containing 'id' and 'sequence' columns.
    """
    data = []
    with open(fasta_input_file) as file:
        sequence_id = None
        sequence = []
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if sequence_id is not None:
                    data.append([sequence_id, "".join(sequence)])
                sequence_id = line[1:]
                sequence = []
            else:
                sequence.append(line)
        if sequence_id is not None:
            data.append([sequence_id, "".join(sequence)])

        if use_modin:
            df = mpd.DataFrame(data, columns=["id", "sequence"])
        else:
            df = pd.DataFrame(data, columns=["id", "sequence"])
    return df


def cluster_structures(
    pdb_dir: str,
    cluster_output_filepath: str = None,
    interface_lddt_threshold: float = 0.3,
    chain_tm_threshold: float = 0.7,
    overwrite: bool = False,
    silence_foldseek_output: bool = True,
    foldseek_exec: str = None,
    tmp_dir: str = None,
) -> None:
    """
    Cluster protein structures using Foldseek multimer clustering.

    Args:
        pdb_dir (str): Directory containing PDB files to cluster.
        cluster_output_filepath (str): Path to write clustering results. If None, defaults to
            "cluster_struct_ilddt_{interface_lddt_threshold}_ctm_{chain_tm_threshold}.tsv".
        interface_lddt_threshold (float): Interface lDDT threshold for clustering. Defaults to 0.3.
        chain_tm_threshold (float): Chain TM score threshold for clustering. Defaults to 0.7.
        overwrite (bool): Whether to overwrite existing cluster file. Defaults to False.
        silence_foldseek_output (bool): Whether to silence Foldseek output. Defaults to True.
        foldseek_exec (str): Path to the foldseek executable. Defaults to None. If not provided, the function will check if foldseek is installed.
        tmp_dir (str): Temporary directory for foldseek operations. If None, a temporary directory will be created.
    """
    if cluster_output_filepath is None:
        cluster_output_filepath = f"cluster_struct_ilddt_{interface_lddt_threshold}_ctm_{chain_tm_threshold}.tsv"

    cluster_tsv_path = pathlib.Path(cluster_output_filepath)

    if not cluster_tsv_path.exists() or overwrite:
        # Remove existing file if overwriting
        if cluster_tsv_path.exists() and overwrite:
            cluster_tsv_path.unlink()

        # Check if foldseek is available
        if foldseek_exec is None:
            foldseek_exec = os.getenv("FOLDSEEK_EXEC")

        if foldseek_exec is None and not is_tool("foldseek"):
            logger.error("Foldseek not found. Set FOLDSEEK_EXEC in .env or ensure foldseek is on PATH.")
            raise RuntimeError("Foldseek not found")

        foldseek_exec = "foldseek" if foldseek_exec is None else foldseek_exec

        # Create temporary directory if not provided
        if tmp_dir is None:
            tmp_dir = tempfile.mkdtemp()
        else:
            os.makedirs(tmp_dir, exist_ok=True)

        # Run Foldseek multimer clustering
        output_prefix = os.path.join(tmp_dir, "res")
        cmd = [
            foldseek_exec,
            "easy-multimercluster",
            pdb_dir,
            output_prefix,
            tmp_dir,
            "--interface-lddt-threshold",
            str(interface_lddt_threshold),
            "--chain-tm-threshold",
            str(chain_tm_threshold),
        ]

        try:
            if silence_foldseek_output:
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                subprocess.run(cmd, check=True)

            # Move the cluster TSV file to the desired location
            cluster_tsv_source = os.path.join(tmp_dir, "res_cluster.tsv")
            if os.path.exists(cluster_tsv_source):
                shutil.move(cluster_tsv_source, cluster_tsv_path)
            else:
                raise FileNotFoundError(f"Expected cluster file not found at {cluster_tsv_source}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Foldseek clustering failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during structure clustering: {e}")
            raise
        finally:
            # Clean up temporary directory if we created it
            if tmp_dir is None or tmp_dir.startswith(tempfile.gettempdir()):
                shutil.rmtree(tmp_dir, ignore_errors=True)


def setup_structure_clustering_file_paths(
    data_dir: str,
    file_identifier: str,
    interface_lddt_threshold: float,
    chain_tm_threshold: float,
) -> tuple[pathlib.Path, pathlib.Path]:
    """
    Set up file paths for structure clustering.

    Args:
        data_dir (str): The directory where the files will be stored.
        file_identifier (str): The identifier used to name the files.
        interface_lddt_threshold (float): The interface lDDT threshold for clustering.
        chain_tm_threshold (float): The chain TM score threshold for clustering.

    Returns:
        Tuple[pathlib.Path, pathlib.Path]: A tuple containing the file paths for
            the cluster TSV file and temporary PDB directory.
    """
    cluster_tsv_filepath = (
        pathlib.Path(data_dir)
        / f"cluster_struct_ilddt_{interface_lddt_threshold}_ctm_{chain_tm_threshold}_{file_identifier}.tsv"
    )
    tmp_pdb_dir = pathlib.Path(data_dir) / f"tmp_pdb_{file_identifier}"
    return cluster_tsv_filepath, tmp_pdb_dir


def _copy_single_structure(args):
    """Helper function to copy a single structure file (for multiprocessing)."""
    sample_id, raw_dir, tmp_pdb_dir, format = args

    # Try different possible file extensions
    possible_paths = [
        raw_dir / f"{sample_id}.{format}",
        raw_dir / f"{sample_id}.{format}.gz",
        raw_dir / f"{sample_id}.pdb",  # Fallback to pdb if cif conversion was done
    ]

    source_path = None
    for path in possible_paths:
        if path.exists():
            source_path = path
            break

    if source_path is None:
        return (sample_id, False, f"Could not find structure file for {sample_id}")

    # Copy to temporary directory with .pdb extension for foldseek
    dest_path = tmp_pdb_dir / f"{sample_id}.pdb"

    try:
        # If source is already pdb, just copy
        if source_path.suffix == ".pdb":
            shutil.copy2(source_path, dest_path)
            return (sample_id, True, None)

        # For CIF files, try direct copy first, then convert if needed
        if source_path.suffix in [".cif", ".cif.gz"]:
            # Try direct copy of CIF file
            try:
                shutil.copy2(source_path, dest_path.with_suffix(".cif"))
                return (sample_id, True, None)
            except Exception as e:
                # If direct copy fails, try conversion
                logger.debug(f"Direct CIF copy failed for {sample_id}, trying conversion: {e}")

        # Convert to pdb if needed
        from Bio.PDB import PDBIO, MMCIFParser

        parser = MMCIFParser()
        structure = parser.get_structure(sample_id, str(source_path))
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(dest_path))
        return (sample_id, True, None)

    except Exception as e:
        return (sample_id, False, f"Failed to process {source_path}: {e}")


def copy_structures_to_tmp_dir(
    df_data: pd.DataFrame,
    raw_dir: pathlib.Path,
    tmp_pdb_dir: pathlib.Path,
    format: str = "cif",
    overwrite: bool = False,
    num_workers: int = None,
) -> None:
    """
    Copy structure files to a temporary directory for clustering using multiprocessing.

    Args:
        df_data (pd.DataFrame): DataFrame containing sample IDs.
        raw_dir (pathlib.Path): Directory containing raw structure files.
        tmp_pdb_dir (pathlib.Path): Temporary directory to copy structures to.
        format (str): Format of the structure files (cif, pdb, etc.).
        overwrite (bool): Whether to overwrite existing files in tmp directory.
        num_workers (int): Number of workers for multiprocessing. If None, uses cpu_count().
    """
    if not overwrite and tmp_pdb_dir.exists():
        logger.info(f"Temporary PDB directory {tmp_pdb_dir} already exists, skipping copy.")
        return

    # Create temporary directory
    tmp_pdb_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Copying {len(df_data)} structures to temporary directory {tmp_pdb_dir}")

    # Prepare arguments for multiprocessing
    sample_ids = df_data["sample_id"].tolist()
    args_list = [(sample_id, raw_dir, tmp_pdb_dir, format) for sample_id in sample_ids]

    # Use multiprocessing to copy files
    num_workers = num_workers or min(cpu_count(), len(sample_ids))
    copied_count = 0
    failed_count = 0

    with Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(_copy_single_structure, args_list),
                total=len(args_list),
                desc="Copying structures",
                unit="file",
            )
        )

    # Process results
    for sample_id, success, error_msg in results:
        if success:
            copied_count += 1
        else:
            failed_count += 1
            logger.warning(f"Failed to copy {sample_id}: {error_msg}")

    logger.info(f"Successfully copied {copied_count} structures to {tmp_pdb_dir}")
    if failed_count > 0:
        logger.warning(f"Failed to copy {failed_count} structures")
