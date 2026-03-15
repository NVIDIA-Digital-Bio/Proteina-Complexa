#!/usr/bin/env python3
"""
Script to rename chain IDs in PDB files from ("A", "B") to ("B", "A") and reorder them.

This script processes PDB files in a directory and:
1. Identifies PDB files with exactly 2 chains ("A" and "B")
2. Renames chain "A" to "B" and chain "B" to "A"
3. Reorders the chains so that the new "A" chain comes before the new "B" chain
4. Performs in-place modifications

Usage:
    python rename_chains.py --data_dir /path/to/pdb/files
"""

import argparse
import glob
import os
import sys

from loguru import logger

from proteinfoundation.utils.pdb_utils import from_pdb_file, write_prot_to_pdb


def get_pdb_files_from_dir(pdb_dir: str) -> list[str]:
    """
    Get all PDB files from a directory.

    Args:
        pdb_dir: Directory containing PDB files

    Returns:
        List of paths to PDB files
    """
    pdb_files = []
    for ext in ["*.pdb", "*.pdb.gz"]:
        pdb_files.extend(glob.glob(os.path.join(pdb_dir, ext)))
        # pdb_files.extend(glob.glob(os.path.join(pdb_dir, '**', ext), recursive=True))

    pdb_files = sorted(pdb_files)
    logger.info(f"Found {len(pdb_files)} PDB files in {pdb_dir}")
    return pdb_files


def get_chain_ids_from_pdb(pdb_file: str) -> list[str]:
    """
    Extract chain IDs from a PDB file.

    Args:
        pdb_file: Path to the PDB file

    Returns:
        List of chain IDs found in the file
    """
    try:
        # Use BioPython to parse the PDB file
        from Bio.PDB import PDBParser

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("temp", pdb_file)

        chain_ids = []
        for model in structure:
            for chain in model:
                chain_ids.append(chain.id)

        return sorted(chain_ids)
    except Exception as e:
        logger.warning(f"Could not parse {pdb_file}: {e}")
        return []


def rename_and_reorder_chains(pdb_file: str) -> bool:
    """
    Rename chains from ("A", "B") to ("B", "A") and reorder them.

    Args:
        pdb_file: Path to the PDB file to process

    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if this is a double-chain PDB with chains A and B
        chain_ids = get_chain_ids_from_pdb(pdb_file)

        if len(chain_ids) != 2 or set(chain_ids) != {"A", "B"}:
            logger.debug(f"Skipping {pdb_file}: not a double-chain PDB with chains A and B (found: {chain_ids})")
            return False

        logger.info(f"Processing {pdb_file} with chains {chain_ids}")

        # Read the PDB file using the existing utility
        protein = from_pdb_file(pdb_file)

        # Create a mapping to rename chains: A -> B, B -> A
        # First, we need to understand the current chain structure
        unique_chain_indices = sorted(set(protein.chain_index))

        if len(unique_chain_indices) != 2:
            logger.warning(f"Expected 2 unique chain indices, found {len(unique_chain_indices)} in {pdb_file}")
            return False

        # Create new chain indices: swap the chain indices
        new_chain_index = protein.chain_index.copy()
        old_to_new = {
            unique_chain_indices[0]: unique_chain_indices[1],  # A -> B
            unique_chain_indices[1]: unique_chain_indices[0],  # B -> A
        }

        for i in range(len(new_chain_index)):
            new_chain_index[i] = old_to_new[new_chain_index[i]]

        # Create a new protein with reordered chains
        # We need to reorder the residues so that the new "A" chain comes first
        new_chain_order = [
            unique_chain_indices[1],
            unique_chain_indices[0],
        ]  # B, A -> A, B

        # Find indices for each chain
        chain_a_indices = []
        chain_b_indices = []

        for i, chain_idx in enumerate(protein.chain_index):
            if chain_idx == unique_chain_indices[0]:  # Original A
                chain_a_indices.append(i)
            else:  # Original B
                chain_b_indices.append(i)

        # Reorder: put original B first (new A), then original A (new B)
        reorder_indices = chain_b_indices + chain_a_indices

        # Create new protein with reordered data
        new_protein = type(protein)(
            atom_positions=protein.atom_positions[reorder_indices],
            atom_mask=protein.atom_mask[reorder_indices],
            aatype=protein.aatype[reorder_indices],
            residue_index=protein.residue_index[reorder_indices],
            chain_index=new_chain_index[reorder_indices],
            b_factors=(protein.b_factors[reorder_indices] if protein.b_factors is not None else None),
        )

        # Write back to the same file
        write_prot_to_pdb(
            prot_pos=new_protein.atom_positions,
            file_path=pdb_file,
            aatype=new_protein.aatype,
            chain_index=new_protein.chain_index,
            overwrite=True,
            b_factors=new_protein.b_factors,
            no_indexing=True,
        )

        logger.info(f"Successfully processed {pdb_file}")
        return True

    except Exception as e:
        logger.error(f"Error processing {pdb_file}: {e}")
        return False


def process_directory(data_dir: str) -> tuple[int, int]:
    """
    Process all PDB files in a directory.

    Args:
        data_dir: Directory containing PDB files

    Returns:
        Tuple of (total_files, processed_files)
    """
    if not os.path.exists(data_dir):
        logger.error(f"Directory {data_dir} does not exist")
        return 0, 0

    pdb_files = get_pdb_files_from_dir(data_dir)

    if not pdb_files:
        logger.warning(f"No PDB files found in {data_dir}")
        return 0, 0

    processed_count = 0
    total_count = len(pdb_files)

    for pdb_file in pdb_files:
        if rename_and_reorder_chains(pdb_file):
            processed_count += 1

    return total_count, processed_count


def main():
    """Main function to parse arguments and process files."""
    parser = argparse.ArgumentParser(description="Rename chain IDs in PDB files from (A, B) to (B, A) and reorder them")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing PDB files to process",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    logger.info(f"Processing PDB files in directory: {args.data_dir}")

    total_files, processed_files = process_directory(args.data_dir)

    logger.info("Processing complete!")
    logger.info(f"Total PDB files found: {total_files}")
    logger.info(f"Successfully processed: {processed_files}")
    logger.info(f"Skipped: {total_files - processed_files}")


if __name__ == "__main__":
    main()
