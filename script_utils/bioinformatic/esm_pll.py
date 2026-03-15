import argparse
import os

import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser, PPBuilder
from transformers import AutoModelForMaskedLM, AutoTokenizer


def find_pdb_files(root_dir):
    pdb_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".pdb"):
                pdb_files.append(os.path.join(dirpath, filename))
    return pdb_files


def extract_chain_b_sequence(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("X", pdb_file)
    for model in structure:
        for chain in model:
            if chain.id == "B":  # change this to 'A' for chain A
                polypeptides = PPBuilder().build_peptides(chain)
                seq = "".join([str(pp.get_sequence()) for pp in polypeptides])
                return seq
    return None


def calculate_pseudo_perplexity(model, tokenizer, sequence, device="cuda"):
    inputs = tokenizer(sequence, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    seq_length = input_ids.size(1) - 2  # Exclude BOS and EOS tokens
    log_probs = []
    for i in range(1, seq_length + 1):  # Skip BOS token
        masked_input = input_ids.clone()
        masked_input[0, i] = tokenizer.mask_token_id
        with torch.no_grad():
            outputs = model(masked_input)
            logits = outputs.logits
            log_prob = torch.log_softmax(logits[0, i], dim=-1)
            true_token = input_ids[0, i]
            log_probs.append(log_prob[true_token].item())
    avg_log_likelihood = sum(log_probs) / len(log_probs)
    pseudo_ppl = np.exp(-avg_log_likelihood)
    return pseudo_ppl, avg_log_likelihood


def main():
    parser = argparse.ArgumentParser(
        description="Score PDB chain B sequences using ESM2 pseudo-perplexity (searches folder + subfolders)"
    )
    parser.add_argument(
        "--pdb_root",
        type=str,
        required=True,
        help="Folder containing PDB files (searched recursively)",
    )
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV file for results")
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/esm2_t33_650M_UR50D",
        help="ESM2 model name",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    results = []
    pdb_files = find_pdb_files(args.pdb_root)
    print(f"Found {len(pdb_files)} PDB files.")
    for pdb_file in pdb_files:
        sequence = extract_chain_b_sequence(pdb_file)
        if sequence and len(sequence) > 0:
            pppl, log_likelihood = calculate_pseudo_perplexity(model, tokenizer, str(sequence), device)
            results.append(
                {
                    "pdb_file": pdb_file,
                    "sequence": sequence,
                    "pseudo_perplexity": pppl,
                    "log_likelihood": log_likelihood,
                }
            )
            print(f"{pdb_file}: Pseudo-perplexity = {pppl:.4f}, Log-likelihood = {log_likelihood:.4f}")
        else:
            print(f"{pdb_file}: No sequence found for chain B.")

    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()


# python script_utils/esm_pll.py \
#     --pdb_root /path/to/results/top_100_successful_self_samples \
#     --output_csv /path/to/results/top_100_esm_pll_results_self.csv
