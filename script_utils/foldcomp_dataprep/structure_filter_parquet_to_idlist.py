import argparse
import os

import modin.pandas as mpd
import ray

# Initialize Ray for Modin
ray.init()


def process_and_filter_file(file_path, output_file, criteria):
    df = mpd.read_parquet(file_path)
    mask = mpd.Series(True, index=df.index)
    if criteria["max_coil"] is not None:
        mask &= df["coil_percent"] <= criteria["max_coil"]
    if criteria["min_alpha"] is not None:
        mask &= df["helix_percent"] >= criteria["min_alpha"]
    if criteria["min_beta"] is not None:
        mask &= df["strand_percent"] >= criteria["min_beta"]
    if criteria["max_rog"] is not None:
        mask &= df["radius_of_gyration"] <= criteria["max_rog"]
    if criteria["min_rog"] is not None:
        mask &= df["radius_of_gyration"] >= criteria["min_rog"]

    filtered_ids = df.loc[mask, "id"]
    filtered_ids = filtered_ids.str.replace(".cif.gz$", "", regex=True)

    with open(output_file, "a") as f:
        for id in filtered_ids:
            f.write(f"{id}\n")


def filter_processed_files(directory, output_file, criteria):
    for file in os.listdir(directory):
        if file.endswith(".parquet"):
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
    parser.add_argument("--max_coil", type=float, help="Maximum coil percentage")
    parser.add_argument("--min_alpha", type=float, help="Minimum alpha helix percentage")
    parser.add_argument("--min_beta", type=float, help="Minimum beta sheet percentage")
    parser.add_argument("--max_rog", type=float, help="Maximum radius of gyration")
    parser.add_argument("--min_rog", type=float, help="Minimum radius of gyration")
    args = parser.parse_args()

    criteria = {
        "max_coil": args.max_coil,
        "min_alpha": args.min_alpha,
        "min_beta": args.min_beta,
        "max_rog": args.max_rog,
        "min_rog": args.min_rog,
    }

    output_file = get_output_filename(args.directory, criteria)

    # Clear the output file if it already exists
    open(output_file, "w").close()

    filter_processed_files(args.directory, output_file, criteria)


if __name__ == "__main__":
    main()
