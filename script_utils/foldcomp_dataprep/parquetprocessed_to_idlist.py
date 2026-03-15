import argparse
import os

import modin.pandas as mpd
import ray

# Initialize Ray for Modin
ray.init()


def process_and_filter_file(file_path, output_file, criteria):
    df = mpd.read_parquet(file_path)
    mask = mpd.Series(True, index=df.index)

    if criteria["max_length"] is not None:
        mask &= df["length"] <= criteria["max_length"]
    if criteria["min_length"] is not None:
        mask &= df["length"] >= criteria["min_length"]
    if criteria["max_plddt_avg"] is not None:
        mask &= df["plddt_avg"] <= criteria["max_plddt_avg"]
    if criteria["min_plddt_avg"] is not None:
        mask &= df["plddt_avg"] >= criteria["min_plddt_avg"]
    if criteria["max_plddt_std"] is not None:
        mask &= df["plddt_std"] <= criteria["max_plddt_std"]
    if criteria["min_plddt_std"] is not None:
        mask &= df["plddt_std"] >= criteria["min_plddt_std"]

    filtered_ids = df.loc[mask, "header"]

    with open(output_file, "a") as f:
        for id in filtered_ids:
            f.write(f"{id}\n")


def filter_processed_files(directory, output_file, criteria):
    for file in os.listdir(directory):
        if file.endswith("_processed.parquet"):
            file_path = os.path.join(directory, file)
            print(f"Processing and filtering {file}...")
            process_and_filter_file(file_path, output_file, criteria)
    print(f"Filtered ids written to {output_file}")


def get_output_filename(directory, criteria):
    criteria_str = "_".join([f"{k}_{v}" for k, v in criteria.items() if v is not None])
    return os.path.join(directory, f"id_list_{criteria_str}.txt")


def main():
    parser = argparse.ArgumentParser(
        description="Filter processed parquet files and create a text file with filtered IDs."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Path to the directory containing processed parquet files",
    )
    parser.add_argument("--max_length", type=int, help="Maximum length filter")
    parser.add_argument("--min_length", type=int, help="Minimum length filter")
    parser.add_argument("--max_plddt_avg", type=int, help="Maximum plddt average filter")
    parser.add_argument("--min_plddt_avg", type=int, help="Minimum plddt average filter")
    parser.add_argument("--max_plddt_std", type=int, help="Maximum plddt standard deviation filter")
    parser.add_argument("--min_plddt_std", type=int, help="Minimum plddt standard deviation filter")

    args = parser.parse_args()

    criteria = {
        "max_length": args.max_length,
        "min_length": args.min_length,
        "max_plddt_avg": args.max_plddt_avg,
        "min_plddt_avg": args.min_plddt_avg,
        "max_plddt_std": args.max_plddt_std,
        "min_plddt_std": args.min_plddt_std,
    }

    output_file = get_output_filename(args.directory, criteria)

    # Clear the output file if it already exists
    open(output_file, "w").close()

    filter_processed_files(args.directory, output_file, criteria)


if __name__ == "__main__":
    main()
