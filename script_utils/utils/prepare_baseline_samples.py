import argparse
import os
import re


def fetch_pdb_files(root_path: str):
    """
    Fetch all pdb files in the root_path directory recursively. Returns results as a
    sorted list.
    """
    pdb_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".pdb"):
                pdb_files.append(os.path.join(root, file))
    return sorted(pdb_files)


def rename_with_job_id(pdb_files: list[str], njobs: int):
    """
    Rename pdb files with a job id. The job id is the index of the pdb file in the list
    (remainder of division by the number of jobs). The new name is `job_{job_id}_{original_name}`.
    PDB files are renamed in place.

    If the original name already has a job prefix, it is removed, and the new one is added.

    Args:
        pdb_files: List of pdb file paths.
        njobs: Number of jobs to parallelize the evaluation.

    Returns:
        None, renames in place.
    """

    def remove_job_prefix(s):
        """
        If s starts with `job_{int}_` it removes that prefix.
        """
        return re.sub(r"^job_\d+_", "", s)

    for i, file in enumerate(pdb_files):
        job_id = i % njobs
        original_name = remove_job_prefix(os.path.basename(file))
        new_name = f"job_{job_id}_{original_name}"
        new_path = os.path.join(os.path.dirname(file), new_name)
        os.rename(file, new_path)


if __name__ == "__main__":
    """
    This should be used for baseline samples. They should be placed in a directory
    `./inference/samples/{name}.pdb`. This script will rename the files to `job_{job_id}_{name}.pdb`
    in place.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--njobs",
        type=int,
        required=True,
        help="Number of jobs to parallelize the evaluation",
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="inference",
        help='Root path directory (default: "inference")',
    )
    args = parser.parse_args()

    pdb_files = fetch_pdb_files(args.root_path)
    rename_with_job_id(pdb_files, args.njobs)
