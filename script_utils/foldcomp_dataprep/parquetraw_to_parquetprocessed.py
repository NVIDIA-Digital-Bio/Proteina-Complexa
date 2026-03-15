import argparse
import os

import modin.pandas as mpd
import numpy as np


def process_parquet_file(input_file, output_file):
    # Read the parquet file
    df = mpd.read_parquet(input_file)

    # Convert 'seq' column directly to numpy array of integers
    df["seq"] = df["seq"].apply(lambda x: np.fromiter(map(int, x), dtype=int))

    # Calculate new columns
    df["length"] = df["seq"].apply(len)
    df["plddt_avg"] = df["seq"].apply(np.mean)
    df["plddt_std"] = df["seq"].apply(np.std)
    df["plddt_avg"] = df["plddt_avg"] * 10
    df["plddt_std"] = df["plddt_std"] * 10

    # Write the updated dataframe to a new parquet file
    df.to_parquet(output_file, engine="pyarrow", compression="snappy")


def get_parquet_files(directory):
    return [f for f in os.listdir(directory) if f.endswith(".parquet")]


def main():
    parser = argparse.ArgumentParser(description="Process all parquet files in a directory.")
    parser.add_argument("directory", type=str, help="Path to the directory containing parquet files")

    args = parser.parse_args()

    parquet_files = get_parquet_files(args.directory)

    for file in parquet_files:
        input_path = os.path.join(args.directory, file)
        output_filename = file.replace(".parquet", "_processed.parquet")
        output_path = os.path.join(args.directory, output_filename)

        print(f"Processing {file}...")
        process_parquet_file(input_path, output_path)
        print(f"Processed file saved as {output_filename}")


if __name__ == "__main__":
    main()
