import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openfold.np.residue_constants import restype_1to3, restypes  # , restype_3to1

# UniProt frequencies (relative)
uniprot_freq = {
    "A": 8.92,
    "Q": 3.81,
    "L": 9.81,
    "S": 6.93,
    "R": 5.85,
    "E": 6.26,
    "K": 4.96,
    "T": 5.56,
    "N": 3.81,
    "G": 7.22,
    "M": 2.32,
    "W": 1.30,
    "D": 5.47,
    "H": 2.24,
    "F": 3.87,
    "Y": 2.87,
    "C": 1.32,
    "I": 5.48,
    "P": 5.04,
    "V": 6.84,
}
uniprot_total = sum(uniprot_freq.values())
uniprot = {k: v / uniprot_total for k, v in uniprot_freq.items()}

# Training data (your numbers, relative)
# training_data_raw = {
#     'L': 7615053, 'A': 6570809, 'E': 5096075, 'V': 5051509, 'G': 5003185,
#     'S': 4684383, 'I': 4653635, 'R': 4621064, 'K': 4313515, 'D': 4256128,
#     'T': 3978182, 'P': 3300105, 'F': 3163395, 'N': 3020920, 'Q': 2743188,
#     'Y': 2551999, 'M': 2007126, 'H': 1687764, 'W': 1169378, 'C': 1108314
# }
# training_total = sum(training_data_raw.values())
# training_data = {k: v / training_total for k, v in training_data_raw.items()}

# Comment out the old data
# # MPNN data
mpnn_raw = {
    "M": 2012,
    "T": 9838,
    "I": 11330,
    "E": 25760,
    "V": 20599,
    "R": 11784,
    "L": 25318,
    "D": 13271,
    "H": 3192,
    "A": 24741,
    "G": 17903,
    "C": 1184,
    "F": 5191,
    "Y": 4627,
    "P": 12008,
    "S": 7734,
    "W": 995,
    "Q": 2069,
    "K": 15495,
    "N": 949,
}
mpnn_total = sum(mpnn_raw.values())
mpnn = {k: v / mpnn_total for k, v in mpnn_raw.items()}

# # La-Proteina low temp
la_proteina_low_raw = {
    "M": 977,
    "N": 9572,
    "E": 23517,
    "L": 21103,
    "V": 7528,
    "R": 11321,
    "Y": 5327,
    "K": 14901,
    "G": 6291,
    "I": 15050,
    "Q": 1077,
    "A": 11806,
    "H": 914,
    "S": 4206,
    "F": 2950,
    "D": 4695,
    "P": 4205,
    "T": 4226,
    "W": 200,
    "C": 134,
}
la_proteina_low_total = sum(la_proteina_low_raw.values())
la_proteina_low = {k: v / la_proteina_low_total for k, v in la_proteina_low_raw.items()}

# # La-Proteina full temp (new data point)
# la_proteina_full_raw = {'M': 3317, 'E': 10005, 'V': 9815, 'K': 7901, 'A': 12269, 'G': 10004, 'R': 8840, 'L': 15627, 'F': 6535, 'Q': 5355, 'H': 3462, 'I': 9139, 'C': 2000, 'S': 9612, 'Y': 5251, 'T': 7457, 'N': 5962, 'D': 8939, 'W': 1915, 'P': 6595}

# la_proteina_full_total = sum(la_proteina_full_raw.values())
# la_proteina_full = {k: v / la_proteina_full_total for k, v in la_proteina_full_raw.items()}

rfdiffusion1 = [
    192,
    152,
    154,
    108,
    0,
    105,
    578,
    41,
    13,
    146,
    303,
    358,
    24,
    43,
    23,
    94,
    62,
    6,
    75,
    105,
]
rfdiffusion2 = [
    120,
    121,
    47,
    39,
    0,
    46,
    437,
    17,
    13,
    142,
    290,
    190,
    44,
    59,
    28,
    57,
    57,
    2,
    44,
    87,
]
rfdiffusion3 = [
    489,
    213,
    92,
    122,
    0,
    74,
    276,
    333,
    76,
    105,
    296,
    191,
    29,
    118,
    112,
    202,
    385,
    11,
    56,
    364,
]
rfdiffusion = []
for i in range(20):
    rfdiffusion.append(rfdiffusion1[i] + rfdiffusion2[i] + rfdiffusion3[i])
rfdiffusion_total = sum(rfdiffusion)
rfdiffusion = {k: v / rfdiffusion_total for k, v in zip(restypes, rfdiffusion, strict=False)}

bindcraft1 = [
    142,
    310,
    154,
    114,
    2,
    142,
    129,
    151,
    278,
    220,
    152,
    189,
    323,
    151,
    156,
    167,
    189,
    182,
    155,
    307,
]
bindcraft2 = [
    166,
    216,
    129,
    249,
    2,
    112,
    234,
    179,
    192,
    290,
    215,
    136,
    454,
    369,
    188,
    135,
    163,
    276,
    169,
    345,
]
bindcraft3 = [
    112,
    135,
    106,
    316,
    0,
    105,
    231,
    141,
    244,
    93,
    73,
    71,
    223,
    111,
    179,
    88,
    114,
    166,
    177,
    101,
]
bindcraft = []
for i in range(20):
    bindcraft.append(bindcraft1[i] + bindcraft2[i] + bindcraft3[i])
bindcraft_total = sum(bindcraft)
bindcraft = {k: v / bindcraft_total for k, v in zip(restypes, bindcraft, strict=False)}

# Sort amino acids by descending frequency in UniProt
labels = sorted(uniprot.keys(), key=lambda x: uniprot[x], reverse=True)
labels_3 = [restype_1to3[aa] for aa in labels]


def parse_aa_counts(counts_str):
    """Parse amino acid counts from string representation."""
    try:
        # Remove brackets and split by comma
        counts_str = counts_str.strip("[]")
        counts = [int(x.strip()) for x in counts_str.split(",")]
        return counts
    except:
        return None


def counts_to_frequencies(counts):
    """Convert counts to frequencies."""
    if counts is None or sum(counts) == 0:
        return None

    total = sum(counts)
    # Use the standard amino acid order from OpenFold (same as in the analysis)
    # This is the order: A, R, N, D, C, Q, E, G, H, I, L, K, M, F, P, S, T, W, Y, V
    aa_order = restypes
    return {aa: count / total for aa, count in zip(aa_order, counts, strict=False)}


def load_csv_data(run_name, target_task=None):
    """Load data from CSV files."""
    base_path = Path(f"./results_downloaded/{run_name}/a_results_processed")

    if not base_path.exists():
        raise FileNotFoundError(f"Results directory not found: {base_path}")

    # Look for aa_distribution CSV files, but exclude all_samples
    csv_files = []
    for csv_file in base_path.glob("res_aa_distribution_*.csv"):
        if "all_samples" not in csv_file.name:
            csv_files.append(csv_file)

    if not csv_files:
        raise FileNotFoundError(f"No aa_distribution CSV files found in {base_path} (excluding all_samples)")

    all_data = []

    for csv_file in csv_files:
        print(f"Processing file: {csv_file}")
        df = pd.read_csv(csv_file)

        # Apply target task filter if specified
        if target_task and "target_task" in df.columns:
            df = df[df["target_task"] == target_task]

        # Determine which sequence type this CSV file corresponds to
        csv_filename = csv_file.name
        sequence_type = None
        if "mpnn_fixed" in csv_filename:
            sequence_type = "mpnn_fixed"
            continue
        elif "mpnn" in csv_filename:
            sequence_type = "mpnn"
        elif "self" in csv_filename:
            sequence_type = "self"

        if sequence_type is None:
            print(f"  Skipping {csv_filename} - could not determine sequence type")
            continue

        print(f"  Detected sequence type: {sequence_type}")

        # Aggregate data for this sequence type within this CSV file
        aggregated_data = {}

        # Process each row
        for _, row in df.iterrows():
            # Only process columns for the detected sequence type
            for category in ["all", "interface"]:
                # Look for columns that match the pattern
                pattern = f"_res_{sequence_type}_aa"
                if category == "interface":
                    pattern += "_interface"
                pattern += "_counts_aggregated_"

                matching_cols = [col for col in df.columns if col.startswith(pattern)]

                for col in matching_cols:
                    if pd.notna(row[col]) and row[col] != "":
                        counts = parse_aa_counts(str(row[col]))
                        if counts:
                            if category not in aggregated_data:
                                aggregated_data[category] = []
                            aggregated_data[category].append(counts)

        # Aggregate counts for each category
        for category, counts_list in aggregated_data.items():
            if counts_list:
                # Sum all counts element-wise
                aggregated_counts = [sum(counts[i] for counts in counts_list) for i in range(20)]
                freqs = counts_to_frequencies(aggregated_counts)
                if freqs:
                    all_data.append(
                        {
                            "run_id": csv_file.stem,  # Use filename as run_id
                            "type": sequence_type,
                            "category": category,
                            "frequencies": freqs,
                            "source_file": csv_file.name,
                            "num_rows": len(counts_list),
                        }
                    )
                    print(f"  Aggregated {len(counts_list)} rows for {sequence_type} {category}")

    return all_data


def create_plot(data, category, run_name, target_task=None):
    """Create a plot for a specific category (all or interface)."""
    # Prepare reference data
    # dicts = [uniprot, mpnn, la_proteina_low, rfdiffusion, bindcraft]
    # names = ["Uniprot", "MPNN", "La-Proteina Low", "RFDiffusion", "BindCraft"]
    dicts = [rfdiffusion, bindcraft]
    names = ["RFDiffusion", "BindCraft"]

    # Add data from CSV
    for item in data:
        if item["category"] == category:
            dicts.append(item["frequencies"])
            # Include number of aggregated rows in the legend
            names.append(f"{item['type']}")  #  ({item['run_id']})

    if len(dicts) <= 2:  # Only reference data
        print(f"No {category} data found for plotting")
        return

    x = np.arange(len(labels))

    # Adjust width based on number of data points
    if len(dicts) <= 5:
        width = 0.15
    elif len(dicts) <= 10:
        width = 0.12
    else:
        width = 0.08

    # Adjust figure size based on number of data points
    fig_width = max(15, len(dicts) * 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    for i, (freqs, name) in enumerate(zip(dicts, names, strict=False)):
        values = [freqs.get(aa, 0) for aa in labels]
        ax.bar(x + i * width, values, width=width, label=name, alpha=0.8)

    ax.set_xticks(x + (len(dicts) - 1) * width / 2)
    ax.set_xticklabels(labels_3, fontsize=11)
    ax.set_xlabel("Amino Acid (sorted by frequency in UniProt)")
    ax.set_ylabel("Relative Frequency")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    title = f"Amino Acid Relative Frequencies - {category.capitalize()}"
    if target_task:
        title += f" (Target Task: {target_task})"
    plt.title(title)
    plt.tight_layout()

    # Save plot
    filename = f"aa_dist_{category}_{run_name}"
    if target_task:
        filename += f"_{target_task}"
    filename += ".png"
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved plot: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Plot amino acid distributions from CSV files")
    parser.add_argument("--run_name", required=True, help="Name of the run/directory")
    parser.add_argument("--target_task", help="Filter by target task name")

    args = parser.parse_args()

    try:
        # Load data from CSV files
        data = load_csv_data(args.run_name, args.target_task)

        if not data:
            print("No data found matching the criteria")
            return

        print(f"Loaded {len(data)} data points")

        # Create plots for both categories
        create_plot(data, "all", args.run_name, args.target_task)
        create_plot(data, "interface", args.run_name, args.target_task)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
