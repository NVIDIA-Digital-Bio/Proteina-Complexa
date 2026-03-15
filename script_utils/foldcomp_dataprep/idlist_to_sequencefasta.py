import argparse

from Bio import SeqIO


def filter_fasta(fasta_file, idx_file):
    # Load indices from idx_file
    with open(idx_file) as idx_f:
        indices = set(line.strip() for line in idx_f)

    output_file = idx_file.replace("id_list", "sequences").replace(".txt", ".fasta")

    with open(output_file, "w") as out_f:
        for record in SeqIO.parse(fasta_file, "fasta"):
            if record.id in indices:
                SeqIO.write(record, out_f, "fasta")
                indices.remove(record.id)
                if not indices:
                    break


def main():
    parser = argparse.ArgumentParser(description="Filter sequences from a FASTA file using indices from a file.")
    parser.add_argument("fasta_file", type=str, help="Input FASTA file")
    parser.add_argument("idx_file", type=str, help="File containing indices to filter")
    args = parser.parse_args()
    filter_fasta(args.fasta_file, args.idx_file)


if __name__ == "__main__":
    main()
