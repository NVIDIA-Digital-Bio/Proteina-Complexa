import argparse
import os
import tempfile

import foldcomp
import mdtraj as md
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger


def compute_ss_metrics(pdb_path):
    try:
        traj = md.load(pdb_path)
        pdb_ss = md.compute_dssp(traj, simplified=True)
        pdb_coil_percent = np.mean(pdb_ss == "C")
        pdb_helix_percent = np.mean(pdb_ss == "H")
        pdb_strand_percent = np.mean(pdb_ss == "E")
        pdb_ss_percent = pdb_helix_percent + pdb_strand_percent
        pdb_rg = md.compute_rg(traj)[0]
    except Exception:
        pdb_ss = -1
        pdb_coil_percent = -1
        pdb_helix_percent = -1
        pdb_strand_percent = -1
        pdb_ss_percent = -1
        pdb_rg = -1
    return {
        "length": pdb_ss.shape[1],
        "sec_structure": pdb_ss.flatten(),
        "non_coil_percent": np.float32(pdb_ss_percent),
        "coil_percent": np.float32(pdb_coil_percent),
        "helix_percent": np.float32(pdb_helix_percent),
        "strand_percent": np.float32(pdb_strand_percent),
        "radius_of_gyration": np.float32(pdb_rg),
    }


def process_and_write_chunk(db_path, output_dir, ids, task_id):
    db = foldcomp.open(db_path, ids=ids)
    results = []
    counter = 0

    for name, content in db:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb") as temp_file:
            temp_file.write(content)
            metrics = compute_ss_metrics(temp_file.name)
            results.append((name, metrics))
            counter += 1

        if counter % 1000 == 0:
            logger.info(f"Processed {counter} Ids")

        if counter % 5000 == 0:
            logger.info("Writing to parquet")
            write_to_parquet(results, output_dir, task_id)
            results = []

    logger.info("Finished iterating")
    if results:
        logger.info("Writing remaining results to parquet")
        write_to_parquet(results, output_dir, task_id)
    logger.info("Closing database")
    db.close()


def main(db_dir, db_name, output_dir, task_id, num_tasks, start_index=30000000):
    os.makedirs(output_dir, exist_ok=True)

    # Get length of database
    with open(f"{db_dir}/{db_name}.lookup") as f:
        all_ids = f.readlines()

    # Get ids and indices
    all_ids = [x.strip().split("\t")[1] for x in all_ids]
    # all_ids = all_ids[:(len(all_ids)//2)]
    # ids = np.array_split(all_ids, num_tasks)[int(task_id)]
    # ids = list(ids)
    # Determine start index based on task_id
    start_index_worker = start_index + int(task_id) * 30000
    end_index_worker = start_index_worker + 30000
    ids = all_ids[start_index_worker:end_index_worker]  # Get a maximum of 25000 IDs
    if len(ids) == 0:
        logger.warning("No ids found for task, probably out of index. Exiting now")
    else:
        logger.info(f"Processing {len(ids)} IDs in this worker from idx {start_index_worker} to {end_index_worker}")
        process_and_write_chunk(f"{db_dir}/{db_name}", output_dir, ids, task_id)


def write_to_parquet(results, output_dir, task_id):
    if not results:
        return

    logger.info(f"Writing {len(results)} results to Parquet")

    data = {
        "id": pa.array([result[0] for result in results], pa.string()),
        "length": pa.array([result[1]["length"] for result in results], pa.float32()),
        "sec_structure": pa.array([result[1]["sec_structure"] for result in results], pa.list_(pa.string())),
        "non_coil_percent": pa.array([result[1]["non_coil_percent"] for result in results], pa.float32()),
        "coil_percent": pa.array([result[1]["coil_percent"] for result in results], pa.float32()),
        "helix_percent": pa.array([result[1]["helix_percent"] for result in results], pa.float32()),
        "strand_percent": pa.array([result[1]["strand_percent"] for result in results], pa.float32()),
        "radius_of_gyration": pa.array([result[1]["radius_of_gyration"] for result in results], pa.float32()),
    }
    file_path = os.path.join(output_dir, f"partition_{int(task_id)}.parquet")
    table = pa.Table.from_pydict(data)
    if os.path.exists(file_path):
        existing_table = pq.read_table(file_path)
        table = pa.concat_tables([existing_table, table])
    pq.write_table(table, file_path, compression="snappy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PDB files and write metrics to Parquet.")
    parser.add_argument("--db_dir", type=str, required=True, help="Path to the database directory")
    parser.add_argument("--db_name", type=str, required=True, help="Database File Name")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for Parquet files",
    )

    args = parser.parse_args()

    task_id = os.environ["SLURM_ARRAY_TASK_ID"]
    logger.info(f"Starting task {task_id}")
    num_tasks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])

    main(args.db_dir, args.db_name, args.output_dir, task_id, num_tasks)
