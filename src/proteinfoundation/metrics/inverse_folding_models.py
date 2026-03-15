import json
import os
import re
import subprocess
from typing import Literal

from biotite.sequence.io import fasta
from loguru import logger

from proteinfoundation.utils.pdb_utils import pdb_name_from_path


# ProteinMPNN
## ## ## ## ## ## ## ## ## ## ## ##
def extract_gen_seqs_proteinmpnn(path_to_file: str) -> list[dict[str, float]]:
    """Extracts sequences and metadata from ProteinMPNN generation files.

    Args:
        path_to_file: Path to file with ProteinMPNN output in FASTA format.

    Returns:
        List of dictionaries, each containing:
            - 'seq': The amino acid sequence
            - 'score': The score value
            - 'seqid': The sequence recovery value
    """
    seqs = []
    fasta_file = fasta.FastaFile.read(path_to_file)

    # Skip first sequence (model info)
    first = True
    for header, sequence in fasta_file.items():
        if first:
            first = False
            continue

        # Extract score and seq_recovery from header
        score_match = re.search(r"score=([\d\.]+)", header)
        seqid_match = re.search(r"seq_recovery=([\d\.]+)", header)

        if score_match and seqid_match:
            seqs.append(
                {
                    "seq": sequence,
                    "score": float(score_match.group(1)),
                    "seqid": float(seqid_match.group(1)),
                }
            )

    return seqs


def extract_gen_seqs_ligandmpnn(
    path_to_file: str,
    backbone_name: str,
) -> list[dict[str, float]]:
    """Extracts sequences and metadata from ProteinMPNN generation files.

    Args:
        path_to_file: Path to file with ProteinMPNN output in FASTA format.

    Returns:
        List of dictionaries, each containing:
            - 'seq': The amino acid sequence
            - 'score': The score value
            - 'seqid': The sequence recovery value
    """
    seqs = []
    fasta_file = fasta.FastaFile.read(path_to_file)

    # Skip first sequence (model info)
    first = True
    for header, sequence in fasta_file.items():
        if first:
            first = False
            continue

        # Extract score and seq_recovery from header
        score_match = re.search(r"overall_confidence=([\d\.]+)", header)
        seqid_match = re.search(r"seq_rec=([\d\.]+)", header)

        if score_match and seqid_match:
            seqs.append(
                {
                    "seq": sequence,
                    "score": float(score_match.group(1)),
                    "seqid": float(seqid_match.group(1)),
                    "backbone_name": backbone_name,
                }
            )

    return seqs


def write_fix_pos_file(fix_pos: list[str], all_chains: list[str], out_dir_root: str, pdb_file_path: str):
    """Writes a fixed positions file for ProteinMPNN.
    Old version of ProteinMPNN only accept jsonl file for fixed positions.

    Args:
        fix_pos: List of positions to fix in format ["ChainID-ResidueNumber"].
        out_dir_root: Directory where designed sequences will be saved.
        all_chains: All chains in the PDB file.
    """
    name = pdb_name_from_path(pdb_file_path)
    fixed_dict = {name: {chain: [] for chain in all_chains}}

    # Fill in the fixed positions
    for pos in fix_pos:
        try:
            chain = pos[0]
            residue = int(pos[1:])
            if chain in fixed_dict[name]:
                fixed_dict[name][chain].append(residue)
            else:
                raise ValueError(f"Chain {chain} not found in provided chain list: {all_chains}")
        except (IndexError, ValueError) as e:
            if isinstance(e, ValueError) and "not found in provided chain list" in str(e):
                raise
            raise ValueError(f"Invalid fix_pos format. Expected 'ChainIDNumber', got '{pos}'")
    # Create fixed positions file in output directory
    fixed_positions_path = os.path.join(out_dir_root, f"{name}_fixed_positions.jsonl")
    with open(fixed_positions_path, "w") as f:
        json.dump(fixed_dict, f)

    return fixed_positions_path


## ## ## ## ## ## ## ## ## ## ## ##


def run_proteinmpnn(
    pdb_file_path: str,
    out_dir_root: str,
    all_chains: list[str] = ["A", "B"],
    pdb_path_chains: list[str] = ["B"],
    fix_pos: list[str] = None,
    num_seq_per_target: int = 8,
    omit_AAs: str = "X",
    sampling_temp: float = 0.1,
    seed: int | None = None,
    ca_only: bool = True,
    verbose: bool = False,
) -> list[dict[str, float]]:
    """Runs ProteinMPNN for protein sequence design.

    This function provides an interface to ProteinMPNN, a deep learning model for protein
    sequence design. It handles the creation of temporary files for fixed positions and
    manages the execution of the ProteinMPNN command-line tool.

    Args:
        pdb_file_path (str): Path to the input PDB file.
        out_dir_root (str): Directory where designed sequences will be saved.
        all_chains (List[str]): All chains in the PDB file. Defaults to ["A", "B"].
        pdb_path_chains (List[str]): List of chain identifiers to be designed. Defaults to ["B"].
        fix_pos (List[str], optional): List of positions to fix in format ["ChainID-ResidueNumber"].
            Example: ["B45", "B46", "B54"]. Defaults to None.
        num_seq_per_target (int): Number of sequences to generate per target structure.
            Defaults to 8.
        omit_AAs (str, optional): String of amino acids that will not be considered in the design process (e.g. "CX").
            Defaults to "X".
        sampling_temp (float): Temperature parameter for sequence sampling. Higher values increase
            diversity. Defaults to 0.1.
        seed (Optional[int]): Random seed for reproducibility. Defaults to None.
        ca_only (bool): If True, uses only alpha carbon atoms for design. Defaults to True.
        verbose (bool): If True, prints detailed output. Defaults to False.

    Returns:
        List of dictionaries, each containing:
            - 'seq': The amino acid sequence
            - 'score': The score value
            - 'seqid': The sequence recovery value

    Raises:
        ValueError: If the fix_pos format is invalid.
        RuntimeError: If ProteinMPNN command fails.
    """
    name = pdb_name_from_path(pdb_file_path)
    python_exec = os.environ.get("PYTHON_EXEC", "python")
    # Base command without optional parameters
    base_command = f"""
    {python_exec} ./community_models/ProteinMPNN/protein_mpnn_run.py \
        --pdb_path {pdb_file_path} \
        --pdb_path_chains '{" ".join(pdb_path_chains)}' \
        --out_folder {out_dir_root} \
        --num_seq_per_target {num_seq_per_target} \
        --sampling_temp {sampling_temp} \
        --omit_AAs "{omit_AAs}" \
        --batch_size 1 \
        --suppress_print {0 if verbose else 1} \
    """

    if ca_only:
        base_command += " --ca_only"
    if seed is not None:
        base_command += f" --seed {seed}"
    if not verbose:
        base_command += " > /dev/null 2>&1"

    if fix_pos:
        fixed_positions_path = write_fix_pos_file(fix_pos, all_chains, out_dir_root, pdb_file_path)
        command = base_command + f" --fixed_positions_jsonl {fixed_positions_path}"
    else:
        command = base_command

    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"ProteinMPNN command failed with error: {result.stderr}")
            raise RuntimeError(f"ProteinMPNN command failed: {result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error(f"ProteinMPNN command failed with error: {e.stderr}")
        raise RuntimeError(f"ProteinMPNN command failed: {e.stderr}")
    except Exception as e:
        logger.error(f"Unexpected error running ProteinMPNN: {e!s}")
        raise RuntimeError(f"Unexpected error running ProteinMPNN: {e!s}")

    redesigned_seqs_info = extract_gen_seqs_proteinmpnn(os.path.join(out_dir_root, "seqs", name + ".fa"))

    return redesigned_seqs_info


def run_ligandmpnn(
    pdb_file_path: str,
    out_dir_root: str,
    all_chains: list[str] = ["A", "B"],
    pdb_path_chains: list[str] = ["B"],
    fix_pos: list[str] = None,
    num_seq_per_target: int = 1,
    omit_AAs: str = "X",
    sampling_temp: float = 0.1,
    seed: int | None = None,
    verbose: bool = False,
) -> list[dict[str, float]]:
    """Runs LigandMPNN for protein sequence design.

    This function provides an interface to LigandMPNN, a deep learning model for protein
    sequence design. It handles the creation of temporary files for fixed positions and
    manages the execution of the LigandMPNN command-line tool.

    Args:
        pdb_file_path (str): Path to the input PDB file.
        out_dir_root (str): Directory where designed sequences will be saved.
        all_chains (List[str]): All chains in the PDB file. Defaults to ["A", "B"].
        pdb_path_chains (List[str]): List of chain identifiers to be designed. Defaults to ["B"].
        fix_pos (List[str], optional): List of positions to fix in format ["ChainID-ResidueNumber"].
            Example: ["B45", "B46", "B54"]. Defaults to None.
        num_seq_per_target (int): Number of sequences to generate per target structure.
            Defaults to 8.
        omit_AAs (str, optional): String of amino acids that will not be considered in the design process (e.g. "CX").
            Defaults to "X".
        sampling_temp (float): Temperature parameter for sequence sampling. Higher values increase
            diversity. Defaults to 0.1.
        seed (Optional[int]): Random seed for reproducibility. Defaults to None.
        ca_only (bool): If True, uses only alpha carbon atoms for design. Defaults to True.
        verbose (bool): If True, prints detailed output. Defaults to False.

    Returns:
        List of dictionaries, each containing:
            - 'seq': The amino acid sequence
            - 'score': The score value
            - 'seqid': The sequence recovery value

    Raises:
        ValueError: If the fix_pos format is invalid.
        RuntimeError: If ProteinMPNN command fails.
    """
    name = pdb_name_from_path(pdb_file_path)
    python_exec = os.environ.get("PYTHON_EXEC", "python")
    chain_specificifaction = (
        f" --chains_to_design {','.join(pdb_path_chains)}"  # f" --parse_these_chains_only {pdb_path_chains}"
    )
    base_command = f"""
    {python_exec} ./community_models/LigandMPNN/run.py \
        --pdb_path {pdb_file_path} \
        --out_folder {out_dir_root} \
        --temperature {sampling_temp} \
        --omit_AA "{omit_AAs}" \
        --batch_size 1 \
        --number_of_batches {num_seq_per_target} \
        {chain_specificifaction} \
        --model_type ligand_mpnn \
        --checkpoint_ligand_mpnn "./community_models/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt" \
        --ligand_mpnn_use_atom_context 1 \
        --ligand_mpnn_cutoff_for_score 8.0 \
        --ligand_mpnn_use_side_chain_context 0 \
        --verbose {0 if verbose else 1} \
    """
    if fix_pos:
        fixed_residues = " ".join(fix_pos)
        logger.info(f"LigandMPNN fixing positions: {fixed_residues}")
        command = base_command + f' --fixed_residues "{fixed_residues}"'
    else:
        command = base_command
    if seed is not None:
        base_command += f" --seed {seed}"
    if not verbose:
        base_command += " > /dev/null 2>&1"

    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"LigandMPNN command failed with error: {result.stderr}")
            raise RuntimeError(f"LigandMPNN command failed: {result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error(f"LigandMPNN command failed with error: {e.stderr}")
        raise RuntimeError(f"LigandMPNN command failed: {e.stderr}")
    except Exception as e:
        logger.error(f"Unexpected error running LigandMPNN: {e!s}")
        raise RuntimeError(f"Unexpected error running LigandMPNN: {e!s}")

    # Maybe not need this?
    backbone_name = os.path.join(out_dir_root, "backbones", name + "_1.pdb")
    redesigned_seqs_info = extract_gen_seqs_ligandmpnn(
        path_to_file=os.path.join(out_dir_root, "seqs", name + ".fa"),
        backbone_name=backbone_name,
    )
    ### Output only the seqs of chains that were redesigned, not needed for ligand-protein design
    # all_chains = sorted(all_chains)
    # redesigned_chain_idx = [i for i in range(len(all_chains)) if all_chains[i] in pdb_path_chains]
    # for idx, seq_info in enumerate(redesigned_seqs_info):
    #     logger.info(f"Redesigned seq before post-processing: {seq_info['seq']}")
    #     seq = seq_info["seq"].split(":") # list of sequences of different chains
    #     redesigned_seqs = ':'.join([seq[i] for i in redesigned_chain_idx])
    #     redesigned_seqs_info[idx]["seq"] = redesigned_seqs

    return redesigned_seqs_info


def run_solublempnn(
    pdb_file_path: str,
    out_dir_root: str,
    all_chains: list[str] = ["A", "B"],
    pdb_path_chains: list[str] = ["B"],
    fix_pos: list[str] = None,
    num_seq_per_target: int = 1,
    omit_AAs: str = "X",
    sampling_temp: float = 0.1,
    seed: int | None = None,
    verbose: bool = False,
) -> list[dict[str, float]]:
    """Runs SolubleMPNN for protein sequence design.

    This function provides an interface to SolubleMPNN, a deep learning model for protein
    sequence design. It handles the creation of temporary files for fixed positions and
    manages the execution of the SolubleMPNN command-line tool.

    Args:
        pdb_file_path (str): Path to the input PDB file.
        out_dir_root (str): Directory where designed sequences will be saved.
        all_chains (List[str]): All chains in the PDB file. Defaults to ["A", "B"].
        pdb_path_chains (List[str]): List of chain identifiers to be designed. Defaults to ["B"].
        fix_pos (List[str], optional): List of positions to fix in format ["ChainID-ResidueNumber"].
            Example: ["B45", "B46", "B54"]. Defaults to None.
        num_seq_per_target (int): Number of sequences to generate per target structure.
            Defaults to 8.
        omit_AAs (str, optional): String of amino acids that will not be considered in the design process (e.g. "CX").
            Defaults to "X".
        sampling_temp (float): Temperature parameter for sequence sampling. Higher values increase
            diversity. Defaults to 0.1.
        seed (Optional[int]): Random seed for reproducibility. Defaults to None.
        ca_only (bool): If True, uses only alpha carbon atoms for design. Defaults to True.
        verbose (bool): If True, prints detailed output. Defaults to False.

    Returns:
        List of dictionaries, each containing:
            - 'seq': The amino acid sequence
            - 'score': The score value
            - 'seqid': The sequence recovery value

    Raises:
        ValueError: If the fix_pos format is invalid.
        RuntimeError: If ProteinMPNN command fails.
    """
    name = pdb_name_from_path(pdb_file_path)
    python_exec = os.environ.get("PYTHON_EXEC", "python")
    chain_specificifaction = (
        f" --chains_to_design {','.join(pdb_path_chains)}"  # f" --parse_these_chains_only {pdb_path_chains}"
    )
    base_command = f"""
    {python_exec} ./community_models/LigandMPNN/run.py \
        --pdb_path {pdb_file_path} \
        --out_folder {out_dir_root} \
        --temperature {sampling_temp} \
        --omit_AA "{omit_AAs}" \
        --batch_size 1 \
        --number_of_batches {num_seq_per_target} \
        {chain_specificifaction} \
        --model_type soluble_mpnn \
        --checkpoint_soluble_mpnn "./community_models/LigandMPNN/model_params/solublempnn_v_48_020.pt" \
        --verbose {0 if verbose else 1} \
    """
    if fix_pos:
        fixed_residues = " ".join(fix_pos)
        logger.info(f"SolubleMPNN fixing positions: {fixed_residues}")
        command = base_command + f' --fixed_residues "{fixed_residues}"'
    else:
        command = base_command
    if seed is not None:
        base_command += f" --seed {seed}"
    if not verbose:
        base_command += " > /dev/null 2>&1"

    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"SolubleMPNN command failed with error: {result.stderr}")
            raise RuntimeError(f"SolubleMPNN command failed: {result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error(f"SolubleMPNN command failed with error: {e.stderr}")
        raise RuntimeError(f"SolubleMPNN command failed: {e.stderr}")
    except Exception as e:
        logger.error(f"Unexpected error running SolubleMPNN: {e!s}")
        raise RuntimeError(f"Unexpected error running SolubleMPNN: {e!s}")

    # Extract redesigned sequences from SolubleMPNN output
    backbone_name = os.path.join(out_dir_root, "backbones", name + "_1.pdb")
    redesigned_seqs_info = extract_gen_seqs_ligandmpnn(
        path_to_file=os.path.join(out_dir_root, "seqs", name + ".fa"),
        backbone_name=backbone_name,
    )
    ### Output only the seqs of chains that were redesigned, not needed for ligand-protein design
    all_chains = sorted(all_chains)
    redesigned_chain_idx = [i for i in range(len(all_chains)) if all_chains[i] in pdb_path_chains]
    for idx, seq_info in enumerate(redesigned_seqs_info):
        logger.info(f"Redesigned seq before post-processing: {seq_info['seq']}")
        seq = seq_info["seq"].split(":")  # list of sequences of different chains
        redesigned_seqs = ":".join([seq[i] for i in redesigned_chain_idx])
        redesigned_seqs_info[idx]["seq"] = redesigned_seqs

    return redesigned_seqs_info


def inverse_fold(
    model_type: Literal["protein_mpnn", "ligand_mpnn", "soluble_mpnn"],
    pdb_file_path: str,
    out_dir_root: str,
    all_chains: list[str] = ["A", "B"],
    pdb_path_chains: list[str] = ["B"],  # TODO: Rename to chains_to_design
    fix_pos: list[str] = None,
    num_seq_per_target: int = 8,
    omit_AAs: list[str] = ["X"],
    sampling_temp: float = 0.1,
    seed: int | None = None,
    ca_only: bool = True,  # TODO: Remove this
    verbose: bool = False,
) -> list[dict[str, float]]:
    """Runs LigandMPNN for protein sequence design.

    This function provides an interface to LigandMPNN, a deep learning model for protein
    sequence design. It handles the creation of temporary files for fixed positions and
    manages the execution of the LigandMPNN command-line tool.

    Args:
        pdb_file_path (str): Path to the input PDB file.
        out_dir_root (str): Directory where designed sequences will be saved.
        all_chains (List[str]): All chains in the PDB file. Defaults to ["A", "B"].
        pdb_path_chains (List[str]): List of chain identifiers to be designed. Defaults to ["B"].
        fix_pos (List[str], optional): List of positions to fix in format ["ChainID-ResidueNumber"].
            Example: ["B45", "B46", "B54"]. Defaults to None.
        num_seq_per_target (int): Number of sequences to generate per target structure.
            Defaults to 8.
        omit_AAs (List[str], optional): List of amino acids that will not be considered in the design process.
            Defaults to ["X"].
        sampling_temp (float): Temperature parameter for sequence sampling. Higher values increase
            diversity. Defaults to 0.1.
        seed (Optional[int]): Random seed for reproducibility. Defaults to None.
        ca_only (bool): If True, uses only alpha carbon atoms for design. Defaults to True.
        verbose (bool): If True, prints detailed output. Defaults to False.

    Returns:
        List of dictionaries, each containing:
            - 'seq': The amino acid sequence
            - 'score': The score value
            - 'seqid': The sequence recovery value
            - 'backbone_name': The name of the backbone file

    Raises:
        ValueError: If the fix_pos format is invalid.
        RuntimeError: If ProteinMPNN command fails.
    """
    omit_AAs = "".join(omit_AAs)
    if model_type == "protein_mpnn":
        redesigned_seqs_info = run_proteinmpnn(
            pdb_file_path=pdb_file_path,
            out_dir_root=out_dir_root,
            all_chains=all_chains,
            pdb_path_chains=pdb_path_chains,
            fix_pos=fix_pos,
            num_seq_per_target=num_seq_per_target,
            omit_AAs=omit_AAs,
            sampling_temp=sampling_temp,
            seed=seed,
            ca_only=ca_only,
            verbose=verbose,
        )
    elif model_type == "ligand_mpnn":
        redesigned_seqs_info = run_ligandmpnn(
            pdb_file_path=pdb_file_path,
            out_dir_root=out_dir_root,
            all_chains=all_chains,
            pdb_path_chains=pdb_path_chains,
            fix_pos=fix_pos,
            num_seq_per_target=num_seq_per_target,
            omit_AAs=omit_AAs,
            sampling_temp=sampling_temp,
            seed=seed,
            verbose=verbose,
        )
    elif model_type == "soluble_mpnn":
        redesigned_seqs_info = run_solublempnn(
            pdb_file_path=pdb_file_path,
            out_dir_root=out_dir_root,
            all_chains=all_chains,
            pdb_path_chains=pdb_path_chains,
            fix_pos=fix_pos,
            num_seq_per_target=num_seq_per_target,
            omit_AAs=omit_AAs,
            sampling_temp=sampling_temp,
            seed=seed,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Model type {model_type} not supported")

    return redesigned_seqs_info
