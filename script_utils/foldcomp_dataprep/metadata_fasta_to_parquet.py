import argparse
import os

import pyarrow as pa
import pyarrow.parquet as pq
from Bio import SeqIO
from tqdm import tqdm


def process_fasta_to_parquet(fasta_file: str, parquet_dir: str, batch_size: int = 1000000) -> None:
    schema = pa.schema([pa.field("header", pa.string()), pa.field("seq", pa.string())])

    os.makedirs(parquet_dir, exist_ok=True)
    progress_bar = tqdm(unit="records", desc="Processing")

    batch_headers = []
    batch_seqs = []
    batch_num = 1

    for record in SeqIO.parse(fasta_file, "fasta"):
        header = record.id.replace(".cif.gz", "")
        seq = str(record.seq)

        batch_headers.append(header)
        batch_seqs.append(seq)

        if len(batch_headers) == batch_size:
            parquet_file = os.path.join(parquet_dir, f"batch_{batch_num}.parquet")
            batch_table = pa.Table.from_arrays([batch_headers, batch_seqs], schema=schema)
            pq.write_table(batch_table, parquet_file, compression="snappy")
            batch_headers.clear()
            batch_seqs.clear()
            progress_bar.update(batch_size)
            batch_num += 1

    if batch_headers:
        parquet_file = os.path.join(parquet_dir, f"batch_{batch_num}.parquet")
        batch_table = pa.Table.from_arrays([batch_headers, batch_seqs], schema=schema)
        pq.write_table(batch_table, parquet_file, compression="snappy")
        progress_bar.update(len(batch_headers))

    progress_bar.close()


def main():
    parser = argparse.ArgumentParser(description="Process a FASTA file and write its contents to Parquet files.")
    parser.add_argument("fasta_file", type=str, help="Path to the input FASTA file")
    parser.add_argument("parquet_dir", type=str, help="Path to the output directory for Parquet files")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000000,
        help="Number of records to process in each batch (default: 5,000,000)",
    )

    args = parser.parse_args()

    process_fasta_to_parquet(args.fasta_file, args.parquet_dir, args.batch_size)


if __name__ == "__main__":
    main()
