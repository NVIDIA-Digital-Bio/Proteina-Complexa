#!/usr/bin/env python3
"""
Script to sample data from the PDB multimer binder dataset,
save the binder structures as PDB files, and calculate their structural properties.

Usage Examples:
    # Sample 50 structures (default)
    python script_utils/sample_binder_structures.py

    # Sample 100 structures
    python script_utils/sample_binder_structures.py --num_samples 100

    # Process entire dataset
    python script_utils/sample_binder_structures.py --num_samples -1

    # Use different dataset config
    python script_utils/sample_binder_structures.py --config pdb_multimer_binder --num_samples 200

    # Custom output directory
    python script_utils/sample_binder_structures.py --output_dir my_analysis --num_samples 50

    # Set random seed for reproducibility
    python script_utils/sample_binder_structures.py --seed 42 --num_samples 100

    # Use custom dataset name for plot titles
    python script_utils/sample_binder_structures.py --dataset_name "PDB Multimer Binder" --num_samples 100

    # Use multiprocessing with 4 workers
    python script_utils/sample_binder_structures.py --num_workers 4 --num_samples 100

    # Compute only complex metrics (faster)
    python script_utils/sample_binder_structures.py --complex_only --num_samples 100

    # Process whole complex structures (no binder/target separation)
    python script_utils/sample_binder_structures.py --whole_complex_mode --num_samples 100

Arguments:
    --config: Dataset configuration name (default: pdb_multimer_binder_nocrop)
    --num_samples: Number of samples to process (default: 50, use -1 for entire dataset)
    --output_dir: Output directory for analysis results (default: binder_structures_analysis)
    --seed: Random seed for reproducibility (default: 43)
    --dataset_name: Dataset name for plot titles (default: uses config name)
    --num_workers: Number of worker processes for multiprocessing (default: 1, no multiprocessing)
    --save_pdb_files: Save PDB files (default: False)
    --complex_only: Compute structural metrics only for complex structures (default: False)
    --whole_complex_mode: Process whole complex structures without binder/target separation (default: False)
"""

import argparse
import multiprocessing as mp
import os
from collections import defaultdict

import hydra
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()  # Load env variables before importing logger to set LOGURU_LEVEL correctly

from loguru import logger

from proteinfoundation.metrics.structural_metric_ss_ca_ca import compute_ca_metrics, compute_ss_metrics
from proteinfoundation.utils.constants import AA_CHARACTER_PROTORP
from proteinfoundation.utils.pdb_utils import write_prot_to_pdb


def categorize_contact_by_polarity(residue_name):
    """Categorize amino acid by polarity based on AA_CHARACTER_PROTORP.

    Args:
        residue_name (str): Three-letter amino acid code (e.g., 'ALA', 'GLU')

    Returns:
        str: Polarity category - 'A' (apolar), 'P' (polar), 'C' (charged), or 'Unknown'
    """
    return AA_CHARACTER_PROTORP.get(residue_name, "Unknown")


def compute_interface_metrics(
    binder_coords,
    target_coords,
    binder_mask,
    target_mask,
    binder_residue_types,
    target_residue_types,
    distance_threshold=8.0,
):
    """Compute interface metrics between binder and target.

    Args:
        binder_coords: [n_binder, 37, 3] binder coordinates
        target_coords: [n_target, 37, 3] target coordinates
        binder_mask: [n_binder, 37] binder coordinate mask
        target_mask: [n_target, 37] target coordinate mask
        binder_residue_types: [n_binder] binder residue types (integer indices)
        target_residue_types: [n_target] target residue types (integer indices)
        distance_threshold: float, distance threshold for interface definition (default 8.0Å)

    Returns:
        dict: Interface metrics including number of interface residues, interface area, etc.
    """
    try:
        # Extract CA coordinates (index 1 in atom37 representation)
        binder_ca_coords = binder_coords[:, 1, :]  # [n_binder, 3]
        target_ca_coords = target_coords[:, 1, :]  # [n_target, 3]

        # Apply masks to get valid CA coordinates
        binder_ca_mask = binder_mask[:, 1]  # [n_binder]
        target_ca_mask = target_mask[:, 1]  # [n_target]

        binder_ca_coords = binder_ca_coords[binder_ca_mask]  # [n_valid_binder, 3]
        target_ca_coords = target_ca_coords[target_ca_mask]  # [n_valid_target, 3]

        # Get corresponding residue types for valid coordinates
        binder_valid_residue_types = binder_residue_types[binder_ca_mask]
        target_valid_residue_types = target_residue_types[target_ca_mask]

        if len(binder_ca_coords) == 0 or len(target_ca_coords) == 0:
            return {
                "interface_residues_binder": 0,
                "interface_residues_target": 0,
                "interface_residues_total": 0,
                "interface_density": 0.0,
                "min_interface_distance": float("inf"),
                "max_interface_distance": 0.0,
                "avg_interface_distance": 0.0,
                "apolar_contacts": 0,
                "polar_contacts": 0,
                "charged_contacts": 0,
                "apolar_contact_fraction": 0.0,
                "polar_contact_fraction": 0.0,
                "charged_contact_fraction": 0.0,
            }

        # Compute pairwise distances between all CA atoms
        # [n_binder, n_target]
        distances = torch.norm(binder_ca_coords[:, None, :] - target_ca_coords[None, :, :], dim=-1)

        # Find interface residues (within distance threshold)
        interface_mask = distances < distance_threshold

        # Count interface residues
        interface_residues_binder = interface_mask.any(dim=1).sum().item()
        interface_residues_target = interface_mask.any(dim=0).sum().item()
        interface_residues_total = interface_residues_binder + interface_residues_target

        # Interface density (fraction of total residues that are interface residues)
        total_residues = len(binder_ca_coords) + len(target_ca_coords)
        interface_density = interface_residues_total / total_residues if total_residues > 0 else 0.0

        # Distance statistics for interface contacts
        interface_distances = distances[interface_mask]
        if len(interface_distances) > 0:
            min_interface_distance = interface_distances.min().item()
            max_interface_distance = interface_distances.max().item()
            avg_interface_distance = interface_distances.mean().item()
        else:
            min_interface_distance = float("inf")
            max_interface_distance = 0.0
            avg_interface_distance = 0.0

        # Analyze contact polarity
        apolar_contacts = 0
        polar_contacts = 0
        charged_contacts = 0

        # Convert residue type indices to residue names
        from openfold.np import residue_constants

        residue_names = residue_constants.restypes

        # Analyze binder interface residues
        binder_interface_indices = torch.where(interface_mask.any(dim=1))[0]
        for idx in binder_interface_indices:
            res_type_idx = binder_valid_residue_types[idx].item()
            if res_type_idx < len(residue_names):
                res_name = residue_constants.restype_1to3.get(residue_names[res_type_idx], "UNK")
                polarity = categorize_contact_by_polarity(res_name)
                if polarity == "A":
                    apolar_contacts += 1
                elif polarity == "P":
                    polar_contacts += 1
                elif polarity == "C":
                    charged_contacts += 1

        # Analyze target interface residues
        target_interface_indices = torch.where(interface_mask.any(dim=0))[0]
        for idx in target_interface_indices:
            res_type_idx = target_valid_residue_types[idx].item()
            if res_type_idx < len(residue_names):
                res_name = residue_constants.restype_1to3.get(residue_names[res_type_idx], "UNK")
                polarity = categorize_contact_by_polarity(res_name)
                if polarity == "A":
                    apolar_contacts += 1
                elif polarity == "P":
                    polar_contacts += 1
                elif polarity == "C":
                    charged_contacts += 1

        # Calculate polarity fractions
        total_contact_residues = apolar_contacts + polar_contacts + charged_contacts
        if total_contact_residues > 0:
            apolar_contact_fraction = apolar_contacts / total_contact_residues
            polar_contact_fraction = polar_contacts / total_contact_residues
            charged_contact_fraction = charged_contacts / total_contact_residues
        else:
            apolar_contact_fraction = 0.0
            polar_contact_fraction = 0.0
            charged_contact_fraction = 0.0

        return {
            "interface_residues_binder": interface_residues_binder,
            "interface_residues_target": interface_residues_target,
            "interface_residues_total": interface_residues_total,
            "interface_density": interface_density,
            "min_interface_distance": min_interface_distance,
            "max_interface_distance": max_interface_distance,
            "avg_interface_distance": avg_interface_distance,
            "apolar_contacts": apolar_contacts,
            "polar_contacts": polar_contacts,
            "charged_contacts": charged_contacts,
            "apolar_contact_fraction": apolar_contact_fraction,
            "polar_contact_fraction": polar_contact_fraction,
            "charged_contact_fraction": charged_contact_fraction,
        }

    except Exception as e:
        logger.warning(f"Failed to compute interface metrics: {e}")
        return {
            "interface_residues_binder": 0,
            "interface_residues_target": 0,
            "interface_residues_total": 0,
            "interface_density": 0.0,
            "min_interface_distance": float("inf"),
            "max_interface_distance": 0.0,
            "avg_interface_distance": 0.0,
            "apolar_contacts": 0,
            "polar_contacts": 0,
            "charged_contacts": 0,
            "apolar_contact_fraction": 0.0,
            "polar_contact_fraction": 0.0,
            "charged_contact_fraction": 0.0,
        }


def compute_selected_structural_metrics(pdb_path):
    """Computes selected structural metrics, excluding pairwise CA-CA collision metrics."""
    metrics_ss = compute_ss_metrics(pdb_path)
    metrics_ca_ca = compute_ca_metrics(pdb_path)

    # Remove the pairwise collision metric
    if "ca_ca_collisions(2A)" in metrics_ca_ca:
        del metrics_ca_ca["ca_ca_collisions(2A)"]

    return {**metrics_ss, **metrics_ca_ca}


def plot_length_distributions(
    binder_length_distribution,
    target_length_distribution,
    complex_length_distribution,
    output_dir,
    dataset_name="Dataset",
):
    """Plot length distributions for binder, target, and complex structures using coarser bins."""

    def create_binned_distribution(length_distribution, bin_size=50):
        """Create binned distribution from individual length counts."""
        binned_counts = defaultdict(int)
        for length, count in length_distribution.items():
            bin_start = (length // bin_size) * bin_size
            bin_end = bin_start + bin_size - 1
            bin_label = f"{bin_start}-{bin_end}"
            binned_counts[bin_label] += count
        return binned_counts

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Length Distributions - {dataset_name}", fontsize=16, fontweight="bold")

    # Binder length distribution with coarser bins
    binder_binned = create_binned_distribution(binder_length_distribution, bin_size=50)
    binder_bins = list(binder_binned.keys())
    binder_counts = list(binder_binned.values())
    axes[0].bar(
        range(len(binder_bins)),
        binder_counts,
        alpha=0.7,
        color="skyblue",
        edgecolor="navy",
    )
    axes[0].set_title("Binder Length Distribution", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Length Range (residues)", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_xticks(range(len(binder_bins)))
    axes[0].set_xticklabels(binder_bins, rotation=45, ha="right")
    axes[0].grid(True, alpha=0.3)

    # Target length distribution with coarser bins
    target_binned = create_binned_distribution(target_length_distribution, bin_size=50)
    target_bins = list(target_binned.keys())
    target_counts = list(target_binned.values())
    axes[1].bar(
        range(len(target_bins)),
        target_counts,
        alpha=0.7,
        color="lightcoral",
        edgecolor="darkred",
    )
    axes[1].set_title("Target Length Distribution", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Length Range (residues)", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].set_xticks(range(len(target_bins)))
    axes[1].set_xticklabels(target_bins, rotation=45, ha="right")
    axes[1].grid(True, alpha=0.3)

    # Complex length distribution with coarser bins
    complex_binned = create_binned_distribution(complex_length_distribution, bin_size=50)
    complex_bins = list(complex_binned.keys())
    complex_counts = list(complex_binned.values())
    axes[2].bar(
        range(len(complex_bins)),
        complex_counts,
        alpha=0.7,
        color="lightgreen",
        edgecolor="darkgreen",
    )
    axes[2].set_title("Complex Length Distribution", fontsize=14, fontweight="bold")
    axes[2].set_xlabel("Length Range (residues)", fontsize=12)
    axes[2].set_ylabel("Count", fontsize=12)
    axes[2].set_xticks(range(len(complex_bins)))
    axes[2].set_xticklabels(complex_bins, rotation=45, ha="right")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "length_distributions.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    logger.info("Length distribution plots saved to length_distributions.png")


def plot_secondary_structure_distributions(
    binder_ss_distribution,
    target_ss_distribution,
    complex_ss_distribution,
    output_dir,
    dataset_name="Dataset",
):
    """Plot secondary structure distributions for binder, target, and complex structures."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Secondary Structure Distributions - {dataset_name}",
        fontsize=16,
        fontweight="bold",
    )

    # Prepare data for plotting
    ss_types = ["biot_alpha", "biot_beta", "biot_coil"]
    ss_labels = ["Alpha Helix", "Beta Sheet", "Coil"]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

    # Binder secondary structure
    binder_means = [np.mean(binder_ss_distribution[ss]) if binder_ss_distribution[ss] else 0 for ss in ss_types]
    binder_stds = [np.std(binder_ss_distribution[ss]) if binder_ss_distribution[ss] else 0 for ss in ss_types]

    bars1 = axes[0].bar(
        ss_labels,
        binder_means,
        yerr=binder_stds,
        capsize=5,
        alpha=0.7,
        color=colors,
        edgecolor="black",
    )
    axes[0].set_title("Binder Secondary Structure Distribution", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Fraction", fontsize=12)
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, mean_val in zip(bars1, binder_means, strict=False):
        height = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{mean_val:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Target secondary structure
    target_means = [np.mean(target_ss_distribution[ss]) if target_ss_distribution[ss] else 0 for ss in ss_types]
    target_stds = [np.std(target_ss_distribution[ss]) if target_ss_distribution[ss] else 0 for ss in ss_types]

    bars2 = axes[1].bar(
        ss_labels,
        target_means,
        yerr=target_stds,
        capsize=5,
        alpha=0.7,
        color=colors,
        edgecolor="black",
    )
    axes[1].set_title("Target Secondary Structure Distribution", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Fraction", fontsize=12)
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, mean_val in zip(bars2, target_means, strict=False):
        height = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{mean_val:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Complex secondary structure
    complex_means = [np.mean(complex_ss_distribution[ss]) if complex_ss_distribution[ss] else 0 for ss in ss_types]
    complex_stds = [np.std(complex_ss_distribution[ss]) if complex_ss_distribution[ss] else 0 for ss in ss_types]

    bars3 = axes[2].bar(
        ss_labels,
        complex_means,
        yerr=complex_stds,
        capsize=5,
        alpha=0.7,
        color=colors,
        edgecolor="black",
    )
    axes[2].set_title("Complex Secondary Structure Distribution", fontsize=14, fontweight="bold")
    axes[2].set_ylabel("Fraction", fontsize=12)
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, mean_val in zip(bars3, complex_means, strict=False):
        height = bar.get_height()
        axes[2].text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{mean_val:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "secondary_structure_distributions.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    logger.info("Secondary structure distribution plots saved to secondary_structure_distributions.png")


def plot_secondary_structure_proportion_distributions(
    binder_ss_distribution,
    target_ss_distribution,
    complex_ss_distribution,
    output_dir,
    dataset_name="Dataset",
):
    """Plot distributions of secondary structure proportions across all proteins."""

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(
        f"Secondary Structure Proportion Distributions - {dataset_name}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Prepare data for plotting
    ss_types = ["biot_alpha", "biot_beta", "biot_coil"]
    ss_labels = ["Alpha Helix", "Beta Sheet", "Coil"]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    protein_types = ["Binder", "Target", "Complex"]

    for i, protein_type in enumerate(protein_types):
        # Get the appropriate distribution data
        if protein_type == "Binder":
            ss_data = binder_ss_distribution
        elif protein_type == "Target":
            ss_data = target_ss_distribution
        else:  # Complex
            ss_data = complex_ss_distribution

        for j, (ss_type, ss_label, color) in enumerate(zip(ss_types, ss_labels, colors, strict=False)):
            if ss_data[ss_type]:
                values = ss_data[ss_type]
                axes[i, j].hist(values, bins=30, alpha=0.7, color=color, edgecolor="black")
                axes[i, j].set_title(f"{protein_type} - {ss_label}", fontsize=12, fontweight="bold")
                axes[i, j].set_xlabel("Proportion", fontsize=10)
                axes[i, j].set_ylabel("Count", fontsize=10)
                axes[i, j].set_xlim(0, 1)
                axes[i, j].grid(True, alpha=0.3)

                # Add statistics
                mean_val = np.mean(values)
                std_val = np.std(values)
                median_val = np.median(values)
                axes[i, j].axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.3f}")
                axes[i, j].axvline(
                    median_val,
                    color="blue",
                    linestyle=":",
                    label=f"Median: {median_val:.3f}",
                )
                axes[i, j].legend(fontsize=8)

                # Add statistics text
                axes[i, j].text(
                    0.02,
                    0.98,
                    f"Std: {std_val:.3f}\nN: {len(values)}",
                    transform=axes[i, j].transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                    fontsize=8,
                )
            else:
                axes[i, j].text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=axes[i, j].transAxes,
                    fontsize=12,
                )
                axes[i, j].set_title(f"{protein_type} - {ss_label}", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "secondary_structure_proportion_distributions.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    logger.info(
        "Secondary structure proportion distribution plots saved to secondary_structure_proportion_distributions.png"
    )


def plot_interface_metrics(interface_metrics, output_dir, dataset_name="Dataset"):
    """Plot interface metrics distribution."""

    # Filter out metrics that are not useful for plotting
    plot_metrics = {
        "interface_residues_binder": "Interface Residues (Binder)",
        "interface_residues_target": "Interface Residues (Target)",
        "interface_residues_total": "Total Interface Residues",
        "interface_density": "Interface Density",
        "avg_interface_distance": "Average Interface Distance (Å)",
    }

    n_metrics = len(plot_metrics)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"Interface Metrics Distributions - {dataset_name}",
        fontsize=16,
        fontweight="bold",
    )
    axes = axes.flatten()

    for idx, (metric_key, metric_label) in enumerate(plot_metrics.items()):
        if interface_metrics.get(metric_key):
            values = interface_metrics[metric_key]

            # Create histogram
            axes[idx].hist(values, bins=30, alpha=0.7, color="purple", edgecolor="black")
            axes[idx].set_title(metric_label, fontsize=12, fontweight="bold")
            axes[idx].set_xlabel(metric_label, fontsize=10)
            axes[idx].set_ylabel("Count", fontsize=10)
            axes[idx].grid(True, alpha=0.3)

            # Add statistics text
            mean_val = np.mean(values)
            std_val = np.std(values)
            median_val = np.median(values)
            axes[idx].text(
                0.02,
                0.98,
                f"Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nMedian: {median_val:.3f}",
                transform=axes[idx].transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "interface_metrics_distributions.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    logger.info("Interface metrics plots saved to interface_metrics_distributions.png")


def plot_contact_polarity_distributions(interface_metrics, output_dir, dataset_name="Dataset"):
    """Plot contact polarity distributions."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"Contact Polarity Distributions - {dataset_name}",
        fontsize=16,
        fontweight="bold",
    )

    # Plot 1: Contact counts by polarity
    polarity_metrics = ["apolar_contacts", "polar_contacts", "charged_contacts"]
    polarity_labels = ["Apolar Contacts", "Polar Contacts", "Charged Contacts"]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

    for i, (metric, label, color) in enumerate(zip(polarity_metrics, polarity_labels, colors, strict=False)):
        if interface_metrics[metric]:
            values = interface_metrics[metric]
            axes[0, i].hist(values, bins=30, alpha=0.7, color=color, edgecolor="black")
            axes[0, i].set_title(f"{label} Distribution", fontsize=12, fontweight="bold")
            axes[0, i].set_xlabel("Number of Contacts", fontsize=10)
            axes[0, i].set_ylabel("Count", fontsize=10)
            axes[0, i].grid(True, alpha=0.3)

            # Add statistics
            mean_val = np.mean(values)
            axes[0, i].axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.1f}")
            axes[0, i].legend()

    # Plot 2: Contact fractions by polarity
    fraction_metrics = [
        "apolar_contact_fraction",
        "polar_contact_fraction",
        "charged_contact_fraction",
    ]
    fraction_labels = ["Apolar Fraction", "Polar Fraction", "Charged Fraction"]

    for i, (metric, label, color) in enumerate(zip(fraction_metrics, fraction_labels, colors, strict=False)):
        if interface_metrics[metric]:
            values = interface_metrics[metric]
            axes[1, i].hist(values, bins=30, alpha=0.7, color=color, edgecolor="black")
            axes[1, i].set_title(f"{label} Distribution", fontsize=12, fontweight="bold")
            axes[1, i].set_xlabel("Fraction", fontsize=10)
            axes[1, i].set_ylabel("Count", fontsize=10)
            axes[1, i].set_xlim(0, 1)
            axes[1, i].grid(True, alpha=0.3)

            # Add statistics
            mean_val = np.mean(values)
            axes[1, i].axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.3f}")
            axes[1, i].legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "contact_polarity_distributions.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    logger.info("Contact polarity distribution plots saved to contact_polarity_distributions.png")


def plot_contact_polarity_comparison(interface_metrics, output_dir, dataset_name="Dataset"):
    """Plot comparison of contact polarity distributions."""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Contact Polarity Comparison - {dataset_name}", fontsize=16, fontweight="bold")

    # Plot 1: Average contact counts by polarity
    polarity_metrics = ["apolar_contacts", "polar_contacts", "charged_contacts"]
    polarity_labels = ["Apolar", "Polar", "Charged"]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

    means = []
    stds = []
    for metric in polarity_metrics:
        if interface_metrics[metric]:
            values = interface_metrics[metric]
            means.append(np.mean(values))
            stds.append(np.std(values))
        else:
            means.append(0)
            stds.append(0)

    bars1 = axes[0].bar(
        polarity_labels,
        means,
        yerr=stds,
        capsize=5,
        alpha=0.7,
        color=colors,
        edgecolor="black",
    )
    axes[0].set_title("Average Contact Counts by Polarity", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Average Number of Contacts", fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, mean_val in zip(bars1, means, strict=False):
        height = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{mean_val:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Plot 2: Average contact fractions by polarity
    fraction_metrics = [
        "apolar_contact_fraction",
        "polar_contact_fraction",
        "charged_contact_fraction",
    ]

    means = []
    stds = []
    for metric in fraction_metrics:
        if interface_metrics[metric]:
            values = interface_metrics[metric]
            means.append(np.mean(values))
            stds.append(np.std(values))
        else:
            means.append(0)
            stds.append(0)

    bars2 = axes[1].bar(
        polarity_labels,
        means,
        yerr=stds,
        capsize=5,
        alpha=0.7,
        color=colors,
        edgecolor="black",
    )
    axes[1].set_title("Average Contact Fractions by Polarity", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Average Fraction", fontsize=12)
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, mean_val in zip(bars2, means, strict=False):
        height = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{mean_val:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "contact_polarity_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    logger.info("Contact polarity comparison plot saved to contact_polarity_comparison.png")


def plot_ca_distance_metrics(
    binder_ca_metrics,
    target_ca_metrics,
    complex_ca_metrics,
    output_dir,
    dataset_name="Dataset",
):
    """Plot CA-CA distance metrics for binder, target, and complex structures."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"CA-CA Distance Metrics - {dataset_name}", fontsize=16, fontweight="bold")

    metric_names = ["ca_ca_dist_avg", "ca_ca_dist_median", "ca_ca_dist_std"]
    metric_labels = [
        "Average CA-CA Distance",
        "Median CA-CA Distance",
        "Std CA-CA Distance",
    ]
    colors = ["skyblue", "lightcoral", "lightgreen"]

    for i, (metric_name, metric_label) in enumerate(zip(metric_names, metric_labels, strict=False)):
        # Binder metrics
        if binder_ca_metrics[metric_name]:
            axes[0, i].hist(
                binder_ca_metrics[metric_name],
                bins=30,
                alpha=0.7,
                color=colors[0],
                edgecolor="navy",
                label="Binder",
            )
            axes[0, i].set_title(f"Binder {metric_label}", fontsize=12, fontweight="bold")
            axes[0, i].set_xlabel("Distance (Å)", fontsize=10)
            axes[0, i].set_ylabel("Count", fontsize=10)
            axes[0, i].grid(True, alpha=0.3)

            # Add statistics
            mean_val = np.mean(binder_ca_metrics[metric_name])
            axes[0, i].axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.2f}Å")
            axes[0, i].legend()

        # Target metrics
        if target_ca_metrics[metric_name]:
            axes[1, i].hist(
                target_ca_metrics[metric_name],
                bins=30,
                alpha=0.7,
                color=colors[1],
                edgecolor="darkred",
                label="Target",
            )
            axes[1, i].set_title(f"Target {metric_label}", fontsize=12, fontweight="bold")
            axes[1, i].set_xlabel("Distance (Å)", fontsize=10)
            axes[1, i].set_ylabel("Count", fontsize=10)
            axes[1, i].grid(True, alpha=0.3)

            # Add statistics
            mean_val = np.mean(target_ca_metrics[metric_name])
            axes[1, i].axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.2f}Å")
            axes[1, i].legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "ca_distance_metrics.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    logger.info("CA-CA distance metrics plots saved to ca_distance_metrics.png")


def plot_comparison_summary(
    binder_ss_distribution,
    target_ss_distribution,
    complex_ss_distribution,
    binder_length_distribution,
    target_length_distribution,
    complex_length_distribution,
    interface_metrics,
    output_dir,
    dataset_name="Dataset",
):
    """Create a comprehensive comparison summary plot."""

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        f"Comprehensive Structural Analysis - {dataset_name}",
        fontsize=18,
        fontweight="bold",
        y=0.95,
    )

    # Set up the grid with reduced spacing
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.35)

    # Helper function to create binned distribution
    def create_binned_distribution(length_distribution, bin_size=50):
        """Create binned distribution from individual length counts."""
        binned_counts = defaultdict(int)
        for length, count in length_distribution.items():
            bin_start = (length // bin_size) * bin_size
            bin_end = bin_start + bin_size - 1
            bin_label = f"{bin_start}-{bin_end}"
            binned_counts[bin_label] += count
        return binned_counts

    # 1. Length comparison (top left) - using binned distributions for better comparison
    ax1 = fig.add_subplot(gs[0, :2])

    # Create binned distributions
    binder_binned = create_binned_distribution(binder_length_distribution, bin_size=50)
    target_binned = create_binned_distribution(target_length_distribution, bin_size=50)

    # Get all unique bin labels and sort them
    all_bins = sorted(
        set(list(binder_binned.keys()) + list(target_binned.keys())),
        key=lambda x: int(x.split("-")[0]),
    )

    binder_counts = [binder_binned.get(bin_label, 0) for bin_label in all_bins]
    target_counts = [target_binned.get(bin_label, 0) for bin_label in all_bins]

    x = np.arange(len(all_bins))
    width = 0.35

    ax1.bar(
        x - width / 2,
        binder_counts,
        width,
        alpha=0.7,
        color="skyblue",
        label="Binder",
        edgecolor="navy",
    )
    ax1.bar(
        x + width / 2,
        target_counts,
        width,
        alpha=0.7,
        color="lightcoral",
        label="Target",
        edgecolor="darkred",
    )
    ax1.set_title("Length Distribution Comparison", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Length Range (residues)", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_bins, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Secondary structure comparison (top right)
    ax2 = fig.add_subplot(gs[0, 2:])
    ss_types = ["biot_alpha", "biot_beta", "biot_coil"]
    ss_labels = ["Alpha", "Beta", "Coil"]

    x = np.arange(len(ss_labels))
    width = 0.25

    binder_means = [np.mean(binder_ss_distribution[ss]) if binder_ss_distribution[ss] else 0 for ss in ss_types]
    target_means = [np.mean(target_ss_distribution[ss]) if target_ss_distribution[ss] else 0 for ss in ss_types]
    complex_means = [np.mean(complex_ss_distribution[ss]) if complex_ss_distribution[ss] else 0 for ss in ss_types]

    ax2.bar(x - width, binder_means, width, label="Binder", alpha=0.7, color="skyblue")
    ax2.bar(x, target_means, width, label="Target", alpha=0.7, color="lightcoral")
    ax2.bar(x + width, complex_means, width, label="Complex", alpha=0.7, color="lightgreen")

    ax2.set_title("Secondary Structure Comparison", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Secondary Structure Type", fontsize=12)
    ax2.set_ylabel("Fraction", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(ss_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Interface metrics (bottom left)
    ax3 = fig.add_subplot(gs[1, :2])
    if interface_metrics["interface_residues_total"]:
        ax3.hist(
            interface_metrics["interface_residues_total"],
            bins=30,
            alpha=0.7,
            color="purple",
            edgecolor="black",
        )
        ax3.set_title("Total Interface Residues Distribution", fontsize=14, fontweight="bold")
        ax3.set_xlabel("Number of Interface Residues", fontsize=12)
        ax3.set_ylabel("Count", fontsize=12)
        ax3.grid(True, alpha=0.3)

        # Add statistics
        mean_val = np.mean(interface_metrics["interface_residues_total"])
        ax3.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.1f}")
        ax3.legend()

    # 4. Interface density (bottom right)
    ax4 = fig.add_subplot(gs[1, 2:])
    if interface_metrics["interface_density"]:
        ax4.hist(
            interface_metrics["interface_density"],
            bins=30,
            alpha=0.7,
            color="orange",
            edgecolor="darkorange",
        )
        ax4.set_title("Interface Density Distribution", fontsize=14, fontweight="bold")
        ax4.set_xlabel("Interface Density", fontsize=12)
        ax4.set_ylabel("Count", fontsize=12)
        ax4.grid(True, alpha=0.3)

        # Add statistics
        mean_val = np.mean(interface_metrics["interface_density"])
        ax4.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.3f}")
        ax4.legend()

    # 5. Average interface distance (bottom)
    ax5 = fig.add_subplot(gs[2, :])
    if interface_metrics["avg_interface_distance"]:
        ax5.hist(
            interface_metrics["avg_interface_distance"],
            bins=30,
            alpha=0.7,
            color="teal",
            edgecolor="darkgreen",
        )
        ax5.set_title("Average Interface Distance Distribution", fontsize=14, fontweight="bold")
        ax5.set_xlabel("Average Interface Distance (Å)", fontsize=12)
        ax5.set_ylabel("Count", fontsize=12)
        ax5.grid(True, alpha=0.3)

        # Add statistics
        mean_val = np.mean(interface_metrics["avg_interface_distance"])
        ax5.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.2f}Å")
        ax5.legend()

    # 6. Summary statistics table (bottom)
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis("off")

    # Create summary table
    summary_data = []

    # Length statistics
    if binder_length_distribution:
        binder_lengths = list(binder_length_distribution.keys())
        summary_data.append(
            [
                "Binder Length",
                f"{np.mean(binder_lengths):.1f} ± {np.std(binder_lengths):.1f}",
            ]
        )

    if target_length_distribution:
        target_lengths = list(target_length_distribution.keys())
        summary_data.append(
            [
                "Target Length",
                f"{np.mean(target_lengths):.1f} ± {np.std(target_lengths):.1f}",
            ]
        )

    if complex_length_distribution:
        complex_lengths = list(complex_length_distribution.keys())
        summary_data.append(
            [
                "Complex Length",
                f"{np.mean(complex_lengths):.1f} ± {np.std(complex_lengths):.1f}",
            ]
        )

    # Secondary structure statistics
    ss_types = ["biot_alpha", "biot_beta", "biot_coil"]
    ss_labels = ["Alpha", "Beta", "Coil"]

    if binder_ss_distribution["biot_alpha"]:
        binder_alpha_mean = np.mean(binder_ss_distribution["biot_alpha"])
        binder_alpha_std = np.std(binder_ss_distribution["biot_alpha"])
        summary_data.append(["Binder Alpha", f"{binder_alpha_mean:.3f} ± {binder_alpha_std:.3f}"])

    if binder_ss_distribution["biot_beta"]:
        binder_beta_mean = np.mean(binder_ss_distribution["biot_beta"])
        binder_beta_std = np.std(binder_ss_distribution["biot_beta"])
        summary_data.append(["Binder Beta", f"{binder_beta_mean:.3f} ± {binder_beta_std:.3f}"])

    if target_ss_distribution["biot_alpha"]:
        target_alpha_mean = np.mean(target_ss_distribution["biot_alpha"])
        target_alpha_std = np.std(target_ss_distribution["biot_alpha"])
        summary_data.append(["Target Alpha", f"{target_alpha_mean:.3f} ± {target_alpha_std:.3f}"])

    if target_ss_distribution["biot_beta"]:
        target_beta_mean = np.mean(target_ss_distribution["biot_beta"])
        target_beta_std = np.std(target_ss_distribution["biot_beta"])
        summary_data.append(["Target Beta", f"{target_beta_mean:.3f} ± {target_beta_std:.3f}"])

    # Interface statistics
    if interface_metrics["interface_residues_total"]:
        summary_data.append(
            [
                "Interface Residues",
                f"{np.mean(interface_metrics['interface_residues_total']):.1f} ± {np.std(interface_metrics['interface_residues_total']):.1f}",
            ]
        )

    if interface_metrics["interface_density"]:
        summary_data.append(
            [
                "Interface Density",
                f"{np.mean(interface_metrics['interface_density']):.3f} ± {np.std(interface_metrics['interface_density']):.3f}",
            ]
        )

    if interface_metrics["avg_interface_distance"]:
        summary_data.append(
            [
                "Avg Interface Distance",
                f"{np.mean(interface_metrics['avg_interface_distance']):.2f} ± {np.std(interface_metrics['avg_interface_distance']):.2f}Å",
            ]
        )

    # Contact polarity statistics
    if interface_metrics["apolar_contacts"]:
        summary_data.append(
            [
                "Apolar Contacts",
                f"{np.mean(interface_metrics['apolar_contacts']):.1f} ± {np.std(interface_metrics['apolar_contacts']):.1f}",
            ]
        )

    if interface_metrics["polar_contacts"]:
        summary_data.append(
            [
                "Polar Contacts",
                f"{np.mean(interface_metrics['polar_contacts']):.1f} ± {np.std(interface_metrics['polar_contacts']):.1f}",
            ]
        )

    if interface_metrics["charged_contacts"]:
        summary_data.append(
            [
                "Charged Contacts",
                f"{np.mean(interface_metrics['charged_contacts']):.1f} ± {np.std(interface_metrics['charged_contacts']):.1f}",
            ]
        )

    if interface_metrics["apolar_contact_fraction"]:
        summary_data.append(
            [
                "Apolar Fraction",
                f"{np.mean(interface_metrics['apolar_contact_fraction']):.3f} ± {np.std(interface_metrics['apolar_contact_fraction']):.3f}",
            ]
        )

    if interface_metrics["polar_contact_fraction"]:
        summary_data.append(
            [
                "Polar Fraction",
                f"{np.mean(interface_metrics['polar_contact_fraction']):.3f} ± {np.std(interface_metrics['polar_contact_fraction']):.3f}",
            ]
        )

    if interface_metrics["charged_contact_fraction"]:
        summary_data.append(
            [
                "Charged Fraction",
                f"{np.mean(interface_metrics['charged_contact_fraction']):.3f} ± {np.std(interface_metrics['charged_contact_fraction']):.3f}",
            ]
        )

    # Create table only if there's data
    if summary_data:
        table = ax6.table(
            cellText=summary_data,
            colLabels=["Metric", "Mean ± Std"],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)

        # Style the table
        for i in range(len(summary_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header row
                    cell.set_facecolor("#4CAF50")
                    cell.set_text_props(weight="bold", color="white")
                else:
                    cell.set_facecolor("#E8F5E8" if i % 2 == 0 else "white")

        ax6.set_title("Summary Statistics", fontsize=16, fontweight="bold", pad=20)
    else:
        # No data available
        ax6.text(
            0.5,
            0.5,
            "No data available for summary statistics",
            ha="center",
            va="center",
            transform=ax6.transAxes,
            fontsize=14,
        )
        ax6.set_title("Summary Statistics", fontsize=16, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "comprehensive_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    logger.info("Comprehensive comparison plot saved to comprehensive_comparison.png")


def process_wrapper(task_args):
    """Wrapper function for multiprocessing that unpacks task arguments."""
    return process_single_sample(*task_args)


def process_single_sample(
    batch_data,
    sample_idx,
    output_dir,
    sample_id,
    save_pdb_files=False,
    complex_only=False,
    whole_complex_mode=False,
):
    """Process a single sample and return results for multiprocessing."""
    try:
        # Unpack batch data and keep as numpy arrays to avoid memory issues
        coords_atom37 = batch_data["coords_nm"][sample_idx] * 10  # [n_complex, 37, 3] - numpy
        coord_mask = batch_data["coord_mask"][sample_idx]  # [n_complex, 37] - numpy
        residue_mask = batch_data["residue_mask"][sample_idx]  # [n_complex] - numpy
        residue_type = batch_data["residue_type"][sample_idx]  # [n_complex] - numpy

        # Handle whole complex mode (no binder/target separation)
        if whole_complex_mode:
            # Apply masks to get valid residues
            valid_mask = residue_mask

            if not valid_mask.any():
                return {
                    "success": False,
                    "error": "No valid residues found",
                    "sample_id": sample_id,
                }

            # Extract complex coordinates and residue types
            complex_coords = coords_atom37[valid_mask]  # [n_complex, 37, 3] - numpy
            complex_coord_mask = coord_mask[valid_mask]  # [n_complex, 37] - numpy
            complex_residue_type = residue_type[valid_mask]  # [n_complex] - numpy

            # Apply coordinate mask
            complex_coords = complex_coords * complex_coord_mask[..., None]  # numpy operation

            # Save complex structure if requested
            complex_pdb_path = None
            if save_pdb_files:
                complex_dir = os.path.join(output_dir, "complex_structures")
                os.makedirs(complex_dir, exist_ok=True)
                complex_pdb_filename = f"{sample_id}_complex.pdb"
                complex_pdb_path = os.path.join(complex_dir, complex_pdb_filename)

                write_prot_to_pdb(
                    prot_pos=complex_coords,  # Already numpy
                    file_path=complex_pdb_path,
                    aatype=complex_residue_type,  # Already numpy
                    overwrite=True,
                    no_indexing=True,
                )

            # Compute complex structural metrics
            import tempfile

            complex_metrics = None

            try:
                if save_pdb_files and complex_pdb_path:
                    # Use the already saved PDB file
                    temp_complex_path = complex_pdb_path
                    cleanup_temp = False
                else:
                    # Create temporary PDB file for metrics calculation
                    temp_fd, temp_complex_path = tempfile.mkstemp(suffix=".pdb")
                    os.close(temp_fd)  # Close the file descriptor

                    # Write temporary complex structure
                    write_prot_to_pdb(
                        prot_pos=complex_coords,  # Already numpy
                        file_path=temp_complex_path,
                        aatype=complex_residue_type,  # Already numpy
                        overwrite=True,
                        no_indexing=True,
                    )
                    cleanup_temp = True

                # Calculate structural metrics for complex
                complex_metrics = compute_selected_structural_metrics(temp_complex_path)

                # Clean up temporary file if we created one
                if cleanup_temp and os.path.exists(temp_complex_path):
                    os.unlink(temp_complex_path)

            except Exception as e:
                logger.warning(f"Failed to compute complex metrics for {sample_id}: {e}")
                complex_metrics = None

            # Prepare results for whole complex mode
            result = {
                "success": True,
                "sample_id": sample_id,
                "complex_length": valid_mask.sum().item(),
                "binder_metrics": None,  # Not applicable in whole complex mode
                "target_metrics": None,  # Not applicable in whole complex mode
                "complex_metrics": complex_metrics,
                "interface_metrics": None,  # Not applicable in whole complex mode
                "has_target": False,  # No separate target in whole complex mode
                "whole_complex_mode": True,
            }

            return result

        # Check if target data is available (for binder/target mode)
        has_target = batch_data.get("has_target", False)
        target_coords = None
        target_residue_type = None
        target_coord_mask = None
        target_residue_mask = None

        if has_target:
            target_coords = batch_data["x_target"][sample_idx] * 10  # [n_target, 37, 3] - numpy
            target_mask = batch_data["target_mask"][sample_idx]  # [n_target, 37] - numpy
            target_residue_type = batch_data["seq_target"][sample_idx]  # [n_target] - numpy
            target_residue_mask = batch_data["seq_target_mask"][sample_idx]  # [n_target] - numpy
            target_coord_mask = target_mask  # Store for later use

        # Apply masks to get valid binder residues
        valid_mask = residue_mask

        if not valid_mask.any():
            return {
                "success": False,
                "error": "No valid binder residues found",
                "sample_id": sample_id,
            }

        # Extract binder coordinates and residue types
        binder_coords = coords_atom37[valid_mask]  # [n_binder, 37, 3] - numpy
        binder_coord_mask = coord_mask[valid_mask]  # [n_binder, 37] - numpy
        binder_residue_type = residue_type[valid_mask]  # [n_binder] - numpy

        # Apply coordinate mask
        binder_coords = binder_coords * binder_coord_mask[..., None]  # numpy operation

        # Initialize PDB paths
        complex_pdb_path = None

        # Create subdirectories and save PDB files only if requested
        if save_pdb_files:
            complex_dir = os.path.join(output_dir, "complex_structures")
            os.makedirs(complex_dir, exist_ok=True)

        # Process target if available
        if has_target and target_coords is not None and target_coords.shape[0] > 0:
            # Apply masks
            target_coords = target_coords * target_coord_mask[..., None]  # numpy operation
            target_coords = target_coords[target_residue_mask]
            target_residue_type = target_residue_type[target_residue_mask]
            target_coord_mask = target_coord_mask[target_residue_mask]

            # Create complex structure (always needed for interface metrics)
            import numpy as np

            complex_coords = np.concatenate([binder_coords, target_coords], axis=0)
            complex_residue_type = np.concatenate([binder_residue_type, target_residue_type], axis=0)

            # Create chain indices: 0 for binder, 1 for target
            binder_length = len(binder_coords)
            target_length = len(target_coords)
            chain_index = np.concatenate(
                [
                    np.zeros(binder_length, dtype=np.int32),
                    np.ones(target_length, dtype=np.int32),
                ]
            )

            # Save complex structure only if requested
            if save_pdb_files:
                complex_pdb_filename = f"{sample_id}_complex.pdb"
                complex_pdb_path = os.path.join(complex_dir, complex_pdb_filename)

                write_prot_to_pdb(
                    prot_pos=complex_coords,  # Already numpy
                    file_path=complex_pdb_path,
                    aatype=complex_residue_type,  # Already numpy
                    chain_index=chain_index,  # Already numpy
                    overwrite=True,
                    no_indexing=True,
                )

            # Compute interface metrics (convert to torch tensors for this function)
            interface_metrics_result = compute_interface_metrics(
                torch.from_numpy(binder_coords),
                torch.from_numpy(target_coords),
                torch.from_numpy(binder_coord_mask),
                torch.from_numpy(target_coord_mask),
                torch.from_numpy(binder_residue_type),
                torch.from_numpy(target_residue_type),
            )
        else:
            interface_metrics_result = None
            complex_pdb_path = None

        # Calculate structural metrics (create temporary PDB files if needed)
        binder_metrics = None
        target_metrics = None
        complex_metrics = None

        # Always compute structural metrics, create temporary PDB files if not saving permanently
        import tempfile

        # Compute binder metrics (unless complex_only is True)
        if not complex_only:
            try:
                temp_fd, temp_binder_path = tempfile.mkstemp(suffix=".pdb")
                os.close(temp_fd)  # Close the file descriptor

                # Write temporary binder structure
                write_prot_to_pdb(
                    prot_pos=binder_coords,  # Already numpy
                    file_path=temp_binder_path,
                    aatype=binder_residue_type,  # Already numpy
                    overwrite=True,
                    no_indexing=True,
                )

                # Calculate structural metrics for binder
                binder_metrics = compute_selected_structural_metrics(temp_binder_path)

                # Clean up temporary file
                if os.path.exists(temp_binder_path):
                    os.unlink(temp_binder_path)

            except Exception as e:
                logger.warning(f"Failed to compute binder metrics for {sample_id}: {e}")
                binder_metrics = None

        # Compute target and complex metrics if target is available
        if has_target:
            # Compute target metrics (unless complex_only is True)
            if not complex_only:
                try:
                    temp_fd, temp_target_path = tempfile.mkstemp(suffix=".pdb")
                    os.close(temp_fd)  # Close the file descriptor

                    # Write temporary target structure
                    write_prot_to_pdb(
                        prot_pos=target_coords,  # Already numpy
                        file_path=temp_target_path,
                        aatype=target_residue_type,  # Already numpy
                        overwrite=True,
                        no_indexing=True,
                    )

                    # Calculate structural metrics for target
                    target_metrics = compute_selected_structural_metrics(temp_target_path)

                    # Clean up temporary file
                    if os.path.exists(temp_target_path):
                        os.unlink(temp_target_path)

                except Exception as e:
                    logger.warning(f"Failed to compute target metrics for {sample_id}: {e}")
                    target_metrics = None

            try:
                # Create temporary complex structure for metrics calculation
                if save_pdb_files and complex_pdb_path:
                    # Use the already saved PDB file
                    temp_complex_path = complex_pdb_path
                    cleanup_temp_complex = False
                else:
                    # Create temporary PDB file for metrics calculation
                    temp_fd, temp_complex_path = tempfile.mkstemp(suffix=".pdb")
                    os.close(temp_fd)  # Close the file descriptor

                    # Write temporary complex structure
                    write_prot_to_pdb(
                        prot_pos=complex_coords,  # Already numpy
                        file_path=temp_complex_path,
                        aatype=complex_residue_type,  # Already numpy
                        chain_index=chain_index,  # Already numpy
                        overwrite=True,
                        no_indexing=True,
                    )
                    cleanup_temp_complex = True

                # Calculate structural metrics for complex
                complex_metrics = compute_selected_structural_metrics(temp_complex_path)

                # Clean up temporary file if we created one
                if cleanup_temp_complex and os.path.exists(temp_complex_path):
                    os.unlink(temp_complex_path)

            except Exception as e:
                logger.warning(f"Failed to compute complex metrics for {sample_id}: {e}")
                complex_metrics = None

        # Prepare results
        result = {
            "success": True,
            "sample_id": sample_id,
            "binder_length": valid_mask.sum().item(),
            "binder_metrics": binder_metrics,
            "target_metrics": target_metrics,
            "complex_metrics": complex_metrics,
            "interface_metrics": interface_metrics_result,
            "has_target": has_target,
        }

        if has_target and target_residue_mask is not None:
            result["target_length"] = target_residue_mask.sum().item()
            result["complex_length"] = result["binder_length"] + result["target_length"]

        return result

    except Exception as e:
        return {"success": False, "error": str(e), "sample_id": sample_id}


def main():
    """Main function to sample binder structures and analyze secondary structure distribution."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Sample binder structures and analyze structural properties")
    parser.add_argument(
        "--config",
        type=str,
        default="pdb_multimer_binder_nocrop",
        help="Dataset configuration name (default: pdb_multimer_binder_nocrop)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of samples to process (default: 50, use -1 for entire dataset)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="binder_structures_analysis",
        help="Output directory for analysis results (default: binder_structures_analysis)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=43,
        help="Random seed for reproducibility (default: 43)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Dataset name for plot titles (default: uses config name)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes for multiprocessing (default: 1, no multiprocessing)",
    )
    parser.add_argument(
        "--save_pdb_files",
        action="store_true",
        default=False,
        help="Save PDB files (default: False, set to True to save complex structures)",
    )
    parser.add_argument(
        "--complex_only",
        action="store_true",
        default=False,
        help="Compute structural metrics only for complex structures, skip individual binder/target metrics (default: False)",
    )
    parser.add_argument(
        "--whole_complex_mode",
        action="store_true",
        default=False,
        help="Process whole complex structures (no binder/target separation). Only computes complex metrics and secondary structure filtering (default: False)",
    )

    args = parser.parse_args()

    # Set random seed
    L.seed_everything(args.seed)

    # Determine dataset name for plots
    dataset_name = args.dataset_name if args.dataset_name is not None else args.config

    # Load configuration using absolute path
    config_path = os.path.abspath("configs/dataset/pdb")

    # Initialize hydra with the config directory
    version_base = hydra.__version__
    hydra.initialize_config_dir(config_dir=config_path, version_base=version_base)

    logger.info(f"Loading dataset with config: {args.config}")
    logger.info(f"Number of samples to process: {'entire dataset' if args.num_samples == -1 else args.num_samples}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Number of workers: {args.num_workers}")
    logger.info(f"Save PDB files: {args.save_pdb_files}")
    logger.info(f"Complex only metrics: {args.complex_only}")
    logger.info(f"Whole complex mode: {args.whole_complex_mode}")

    cfg = hydra.compose(
        config_name=args.config,
        return_hydra_config=True,
    )

    # Instantiate the datamodule
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data()
    # datamodule.process()
    datamodule.setup("fit")
    train_dataloader = datamodule.train_dataloader()

    # Create output directory for PDB files
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories only if saving PDB files
    if args.save_pdb_files:
        complex_dir = os.path.join(output_dir, "complex_structures")
        os.makedirs(complex_dir, exist_ok=True)

    # Initialize counters and storage for analysis
    total_samples = 0
    successful_saves = 0
    failed_saves = 0

    # Storage for secondary structure analysis
    binder_ss_distribution = {"biot_alpha": [], "biot_beta": [], "biot_coil": []}

    target_ss_distribution = {"biot_alpha": [], "biot_beta": [], "biot_coil": []}

    complex_ss_distribution = {"biot_alpha": [], "biot_beta": [], "biot_coil": []}

    # Storage for CA-CA distance analysis (consecutive distances only)
    binder_ca_metrics = {
        "ca_ca_dist_avg": [],
        "ca_ca_dist_median": [],
        "ca_ca_dist_std": [],
        "ca_ca_dist_min": [],
        "ca_ca_dist_max": [],
    }

    target_ca_metrics = {
        "ca_ca_dist_avg": [],
        "ca_ca_dist_median": [],
        "ca_ca_dist_std": [],
        "ca_ca_dist_min": [],
        "ca_ca_dist_max": [],
    }

    complex_ca_metrics = {
        "ca_ca_dist_avg": [],
        "ca_ca_dist_median": [],
        "ca_ca_dist_std": [],
        "ca_ca_dist_min": [],
        "ca_ca_dist_max": [],
    }

    # Storage for interface analysis
    interface_metrics = {
        "interface_residues_binder": [],
        "interface_residues_target": [],
        "interface_residues_total": [],
        "interface_density": [],
        "min_interface_distance": [],
        "max_interface_distance": [],
        "avg_interface_distance": [],
        "apolar_contacts": [],
        "polar_contacts": [],
        "charged_contacts": [],
        "apolar_contact_fraction": [],
        "polar_contact_fraction": [],
        "charged_contact_fraction": [],
    }

    # Storage for proteins to exclude (>50% coil in complex only)
    proteins_to_exclude = []

    # Storage for proteins to exclude (<10 interface residues)
    proteins_to_exclude_interface = []

    # Storage for all secondary structure statistics
    all_ss_stats = []

    # Length distribution
    binder_length_distribution = defaultdict(int)
    target_length_distribution = defaultdict(int)
    complex_length_distribution = defaultdict(int)

    logger.info("Starting to sample binder structures...")

    # Get total dataset size for progress tracking
    if args.num_samples == -1:
        try:
            total_dataset_size = len(train_dataloader.dataset)
            logger.info(f"Total dataset size: {total_dataset_size} samples")
        except:
            logger.info("Could not determine total dataset size, will process until completion")

    # Prepare batch data for processing
    all_batch_data = []
    all_sample_ids = []

    logger.info("Preparing batch data for processing...")
    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Preparing batches")):
        batch_size = len(batch.id) if hasattr(batch, "id") else 1

        # Check if we've reached the desired number of samples
        if args.num_samples != -1 and len(all_sample_ids) >= args.num_samples:
            logger.info(f"Reached target number of samples ({args.num_samples}), stopping...")
            break

        # Prepare batch data for multiprocessing - convert tensors to numpy to avoid memory issues
        batch_data = {
            "coords_nm": batch.coords_nm.detach().cpu().numpy(),
            "coord_mask": batch.coord_mask.detach().cpu().numpy(),
            "residue_mask": batch.mask_dict["residue_type"].detach().cpu().numpy(),
            "residue_type": batch.residue_type.detach().cpu().numpy(),
            "has_target": False,
        }

        # Add target data if available
        if hasattr(batch, "x_target") and batch.x_target is not None and batch.x_target.shape[1] > 0:
            batch_data.update(
                {
                    "x_target": batch.x_target.detach().cpu().numpy(),
                    "target_mask": batch.target_mask.detach().cpu().numpy(),
                    "seq_target": batch.seq_target.detach().cpu().numpy(),
                    "seq_target_mask": batch.seq_target_mask.detach().cpu().numpy(),
                    "has_target": True,
                }
            )

        all_batch_data.append(batch_data)

        # Add sample IDs
        for i in range(batch_size):
            sample_id = batch.id[i] if hasattr(batch, "id") else f"batch_{batch_idx}_sample_{i}"
            all_sample_ids.append(sample_id)

    # Limit to requested number of samples
    if args.num_samples != -1:
        all_sample_ids = all_sample_ids[: args.num_samples]

        # We need to keep the batches that contain the limited samples
        # Find which batches we need to keep
        remaining_samples = args.num_samples
        needed_batch_data = []

        for batch_data in all_batch_data:
            batch_size = batch_data["coords_nm"].shape[0]
            if remaining_samples <= 0:
                break
            needed_batch_data.append(batch_data)
            remaining_samples -= batch_size

        all_batch_data = needed_batch_data

    logger.info(f"Prepared {len(all_sample_ids)} samples for processing")

    # Process samples
    if args.num_workers > 1:
        logger.info(f"Using multiprocessing with {args.num_workers} workers")

        # Create a list of tasks for multiprocessing
        tasks = []
        sample_idx = 0
        for batch_idx, batch_data in enumerate(all_batch_data):
            batch_size = batch_data["coords_nm"].shape[0]
            for i in range(batch_size):
                if sample_idx < len(all_sample_ids):
                    tasks.append(
                        (
                            batch_data,
                            i,
                            output_dir,
                            all_sample_ids[sample_idx],
                            args.save_pdb_files,
                            args.complex_only,
                            args.whole_complex_mode,
                        )
                    )
                    sample_idx += 1

        logger.info(f"Created {len(tasks)} tasks for multiprocessing")
        logger.info(f"Starting multiprocessing with {args.num_workers} workers...")

        # Process with multiprocessing with progress bar
        with mp.Pool(processes=args.num_workers) as pool:
            # Use imap for real-time progress tracking
            results = []
            for result in tqdm(
                pool.imap(process_wrapper, tasks),
                total=len(tasks),
                desc="Processing samples",
            ):
                results.append(result)
    else:
        logger.info("Using single-threaded processing")

        # Process samples sequentially
        results = []
        sample_idx = 0
        for batch_idx, batch_data in enumerate(tqdm(all_batch_data, desc="Processing batches")):
            batch_size = batch_data["coords_nm"].shape[0]
            for i in range(batch_size):
                if sample_idx < len(all_sample_ids):
                    result = process_single_sample(
                        batch_data,
                        i,
                        output_dir,
                        all_sample_ids[sample_idx],
                        args.save_pdb_files,
                        args.complex_only,
                        args.whole_complex_mode,
                    )
                    results.append(result)
                    sample_idx += 1

    # Process results
    logger.info("Processing results...")
    for result in results:
        if not result["success"]:
            logger.warning(f"Failed to process {result['sample_id']}: {result.get('error', 'Unknown error')}")
            failed_saves += 1
            continue

        sample_id = result["sample_id"]
        successful_saves += 1

        # Process binder metrics
        if result["binder_metrics"]:
            binder_metrics = result["binder_metrics"]

            # Store secondary structure metrics for binder
            for key in binder_ss_distribution:
                if key in binder_metrics:
                    binder_ss_distribution[key].append(binder_metrics[key])

            # Store CA-CA distance metrics for binder
            for key in binder_ca_metrics:
                if key in binder_metrics:
                    binder_ca_metrics[key].append(binder_metrics[key])

            # Store length information for binder
            binder_length = result["binder_length"]
            binder_length_distribution[binder_length] += 1

            # Store all SS stats for binder
            all_ss_stats.append(
                {
                    "sample_id": sample_id,
                    "protein_type": "binder",
                    "coil_fraction": binder_metrics.get("biot_coil", 0.0),
                    "alpha_fraction": binder_metrics.get("biot_alpha", 0.0),
                    "beta_fraction": binder_metrics.get("biot_beta", 0.0),
                    "length": binder_length,
                }
            )

        # Process target metrics
        if result["target_metrics"]:
            target_metrics = result["target_metrics"]

            # Store secondary structure metrics for target
            for key in target_ss_distribution:
                if key in target_metrics:
                    target_ss_distribution[key].append(target_metrics[key])

            # Store CA-CA distance metrics for target
            for key in target_ca_metrics:
                if key in target_metrics:
                    target_ca_metrics[key].append(target_metrics[key])

            # Store length information for target
            target_length = result["target_length"]
            target_length_distribution[target_length] += 1

            # Store all SS stats for target
            all_ss_stats.append(
                {
                    "sample_id": sample_id,
                    "protein_type": "target",
                    "coil_fraction": target_metrics.get("biot_coil", 0.0),
                    "alpha_fraction": target_metrics.get("biot_alpha", 0.0),
                    "beta_fraction": target_metrics.get("biot_beta", 0.0),
                    "length": target_length,
                }
            )

        # Process complex metrics and interface metrics
        # In whole complex mode, we only have complex metrics (no interface metrics)
        if result["complex_metrics"] and (result["interface_metrics"] or result.get("whole_complex_mode", False)):
            complex_metrics = result["complex_metrics"]
            interface_metrics_result = result["interface_metrics"]

            # Check for >50% coil in complex (exclusion criterion)
            if "biot_coil" in complex_metrics and complex_metrics["biot_coil"] > 0.5:
                proteins_to_exclude.append(
                    {
                        "sample_id": sample_id,
                        "protein_type": "complex",
                        "coil_fraction": complex_metrics["biot_coil"],
                        "alpha_fraction": complex_metrics.get("biot_alpha", 0.0),
                        "beta_fraction": complex_metrics.get("biot_beta", 0.0),
                    }
                )

            # Check for <10 interface residues (exclusion criterion) - only in binder/target mode
            if interface_metrics_result and interface_metrics_result["interface_residues_total"] < 10:
                proteins_to_exclude_interface.append(
                    {
                        "sample_id": sample_id,
                        "protein_type": "complex",
                        "interface_residues_total": interface_metrics_result["interface_residues_total"],
                        "interface_residues_binder": interface_metrics_result["interface_residues_binder"],
                        "interface_residues_target": interface_metrics_result["interface_residues_target"],
                        "interface_density": interface_metrics_result["interface_density"],
                    }
                )

            # Store interface metrics (only in binder/target mode)
            if interface_metrics_result:
                for key in interface_metrics:
                    if key in interface_metrics_result:
                        interface_metrics[key].append(interface_metrics_result[key])

            # Store secondary structure metrics for complex
            for key in complex_ss_distribution:
                if key in complex_metrics:
                    complex_ss_distribution[key].append(complex_metrics[key])

            # Store CA-CA distance metrics for complex
            for key in complex_ca_metrics:
                if key in complex_metrics:
                    complex_ca_metrics[key].append(complex_metrics[key])

            # Store all SS stats for complex
            complex_length = result.get(
                "complex_length",
                result.get("binder_length", 0) + result.get("target_length", 0),
            )
            complex_length_distribution[complex_length] += 1

            all_ss_stats.append(
                {
                    "sample_id": sample_id,
                    "protein_type": "complex",
                    "coil_fraction": complex_metrics.get("biot_coil", 0.0),
                    "alpha_fraction": complex_metrics.get("biot_alpha", 0.0),
                    "beta_fraction": complex_metrics.get("biot_beta", 0.0),
                    "length": complex_length,
                }
            )

        total_samples += 1

    # Calculate summary statistics
    logger.info("Processing complete!")
    logger.info(f"Total samples processed: {total_samples}")
    logger.info(f"Successfully saved: {successful_saves}")
    logger.info(f"Failed to save: {failed_saves}")

    if args.num_samples == -1:
        logger.info("Processed entire dataset")
    else:
        logger.info(f"Target samples: {args.num_samples}, Actual processed: {total_samples}")

    # Log and save proteins to exclude (>50% coil in complex only)
    logger.info("\n=== Proteins to Exclude (>50% coil in complex) ===")
    logger.info(f"Total complexes with >50% coil: {len(proteins_to_exclude)}")

    # Save exclusion list to file (detailed version)
    exclusion_file = os.path.join(output_dir, "proteins_to_exclude_coil_50_percent.txt")
    with open(exclusion_file, "w") as f:
        f.write("# Proteins to exclude (>50% coil in complex)\n")
        f.write(f"# Total complexes with >50% coil: {len(proteins_to_exclude)}\n")
        f.write("# Sample_ID\tProtein_Type\tCoil_Fraction\tAlpha_Fraction\tBeta_Fraction\n")

        for protein in proteins_to_exclude:
            f.write(
                f"{protein['sample_id']}\t{protein['protein_type']}\t{protein['coil_fraction']:.4f}\t{protein['alpha_fraction']:.4f}\t{protein['beta_fraction']:.4f}\n"
            )

    logger.info(f"Detailed exclusion list saved to: {exclusion_file}")

    # Save simple exclusion list with only sample IDs
    exclusion_ids_file = os.path.join(output_dir, "proteins_to_exclude_ids.txt")
    with open(exclusion_ids_file, "w") as f:
        for protein in proteins_to_exclude:
            f.write(f"{protein['sample_id']}\n")

    logger.info(f"Simple exclusion list (IDs only) saved to: {exclusion_ids_file}")

    # Log and save proteins to exclude (<10 interface residues)
    logger.info("\n=== Proteins to Exclude (<10 interface residues) ===")
    logger.info(f"Total complexes with <10 interface residues: {len(proteins_to_exclude_interface)}")

    # Save interface exclusion list to file (detailed version)
    interface_exclusion_file = os.path.join(output_dir, "proteins_to_exclude_interface_less_than_10.txt")
    with open(interface_exclusion_file, "w") as f:
        f.write("# Proteins to exclude (<10 interface residues)\n")
        f.write(f"# Total complexes with <10 interface residues: {len(proteins_to_exclude_interface)}\n")
        f.write(
            "# Sample_ID\tProtein_Type\tInterface_Residues_Total\tInterface_Residues_Binder\tInterface_Residues_Target\tInterface_Density\n"
        )

        for protein in proteins_to_exclude_interface:
            f.write(
                f"{protein['sample_id']}\t{protein['protein_type']}\t{protein['interface_residues_total']}\t{protein['interface_residues_binder']}\t{protein['interface_residues_target']}\t{protein['interface_density']:.4f}\n"
            )

    logger.info(f"Detailed interface exclusion list saved to: {interface_exclusion_file}")

    # Save simple interface exclusion list with only sample IDs
    interface_exclusion_ids_file = os.path.join(output_dir, "proteins_to_exclude_interface_ids.txt")
    with open(interface_exclusion_ids_file, "w") as f:
        for protein in proteins_to_exclude_interface:
            f.write(f"{protein['sample_id']}\n")

    logger.info(f"Simple interface exclusion list (IDs only) saved to: {interface_exclusion_ids_file}")

    # Combine both exclusion lists
    all_excluded_ids = set()
    for protein in proteins_to_exclude:
        all_excluded_ids.add(protein["sample_id"])
    for protein in proteins_to_exclude_interface:
        all_excluded_ids.add(protein["sample_id"])

    combined_exclusion_ids_file = os.path.join(output_dir, "proteins_to_exclude_combined_ids.txt")
    with open(combined_exclusion_ids_file, "w") as f:
        for sample_id in sorted(all_excluded_ids):
            f.write(f"{sample_id}\n")

    logger.info(f"Combined exclusion list (IDs only) saved to: {combined_exclusion_ids_file}")
    logger.info(f"Total unique proteins to exclude: {len(all_excluded_ids)}")

    # Save all secondary structure statistics to CSV
    import csv

    ss_csv_file = os.path.join(output_dir, "all_secondary_structure_stats.csv")
    with open(ss_csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample_id",
                "protein_type",
                "coil_fraction",
                "alpha_fraction",
                "beta_fraction",
                "length",
            ]
        )
        for stat in all_ss_stats:
            writer.writerow(
                [
                    stat["sample_id"],
                    stat["protein_type"],
                    f"{stat['coil_fraction']:.4f}",
                    f"{stat['alpha_fraction']:.4f}",
                    f"{stat['beta_fraction']:.4f}",
                    stat["length"],
                ]
            )

    logger.info(f"All secondary structure statistics saved to: {ss_csv_file}")

    # Log detailed information about excluded proteins
    if proteins_to_exclude:
        logger.info("\nDetailed list of complexes with >50% coil:")
        for protein in proteins_to_exclude:
            logger.info(
                f"  {protein['sample_id']}: coil={protein['coil_fraction']:.3f}, alpha={protein['alpha_fraction']:.3f}, beta={protein['beta_fraction']:.3f}"
            )

    if proteins_to_exclude_interface:
        logger.info("\nDetailed list of complexes with <10 interface residues:")
        for protein in proteins_to_exclude_interface:
            logger.info(
                f"  {protein['sample_id']}: interface_residues={protein['interface_residues_total']}, binder={protein['interface_residues_binder']}, target={protein['interface_residues_target']}, density={protein['interface_density']:.3f}"
            )

    # Calculate and log secondary structure distribution for binders
    logger.info("\n=== Binder Secondary Structure Distribution ===")
    for ss_type, values in binder_ss_distribution.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            logger.info(f"{ss_type}: {mean_val:.4f} ± {std_val:.4f} (n={len(values)})")

    # Calculate and log CA-CA distance metrics for binders
    logger.info("\n=== Binder CA-CA Distance Metrics ===")
    for metric_name, values in binder_ca_metrics.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            logger.info(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f} (n={len(values)})")

    # Log binder length distribution
    logger.info("\n=== Binder Length Distribution ===")
    sorted_binder_lengths = sorted(binder_length_distribution.items())
    for length, count in sorted_binder_lengths:
        logger.info(f"Length {length}: {count} structures")

    # Calculate and log secondary structure distribution for targets
    logger.info("\n=== Target Secondary Structure Distribution ===")
    for ss_type, values in target_ss_distribution.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            logger.info(f"{ss_type}: {mean_val:.4f} ± {std_val:.4f} (n={len(values)})")

    # Calculate and log CA-CA distance metrics for targets
    logger.info("\n=== Target CA-CA Distance Metrics ===")
    for metric_name, values in target_ca_metrics.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            logger.info(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f} (n={len(values)})")

    # Log target length distribution
    logger.info("\n=== Target Length Distribution ===")
    sorted_target_lengths = sorted(target_length_distribution.items())
    for length, count in sorted_target_lengths:
        logger.info(f"Length {length}: {count} structures")

    # Calculate and log secondary structure distribution for complexes
    logger.info("\n=== Complex Secondary Structure Distribution ===")
    for ss_type, values in complex_ss_distribution.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            logger.info(f"{ss_type}: {mean_val:.4f} ± {std_val:.4f} (n={len(values)})")

    # Calculate and log CA-CA distance metrics for complexes
    logger.info("\n=== Complex CA-CA Distance Metrics ===")
    for metric_name, values in complex_ca_metrics.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            logger.info(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f} (n={len(values)})")

    # Log complex length distribution
    logger.info("\n=== Complex Length Distribution ===")
    sorted_complex_lengths = sorted(complex_length_distribution.items())
    for length, count in sorted_complex_lengths:
        logger.info(f"Length {length}: {count} structures")

    # Calculate and log interface metrics
    logger.info("\n=== Interface Analysis (8Å threshold) ===")
    for metric_name, values in interface_metrics.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            logger.info(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f} (n={len(values)})")

    # Log contact polarity summary
    logger.info("\n=== Contact Polarity Analysis ===")
    if (
        interface_metrics["apolar_contacts"]
        and interface_metrics["polar_contacts"]
        and interface_metrics["charged_contacts"]
    ):
        apolar_mean = np.mean(interface_metrics["apolar_contacts"])
        polar_mean = np.mean(interface_metrics["polar_contacts"])
        charged_mean = np.mean(interface_metrics["charged_contacts"])
        total_contacts = apolar_mean + polar_mean + charged_mean

        logger.info("Average Contact Distribution:")
        logger.info(f"  Apolar: {apolar_mean:.1f} ({apolar_mean / total_contacts * 100:.1f}%)")
        logger.info(f"  Polar: {polar_mean:.1f} ({polar_mean / total_contacts * 100:.1f}%)")
        logger.info(f"  Charged: {charged_mean:.1f} ({charged_mean / total_contacts * 100:.1f}%)")
        logger.info(f"  Total: {total_contacts:.1f}")

    # Save detailed results to file
    results_file = os.path.join(output_dir, "structural_analysis_results.txt")
    with open(results_file, "w") as f:
        f.write("=== Binder Structure Analysis Results ===\n\n")
        f.write(f"Total samples processed: {total_samples}\n")
        f.write(f"Successfully saved: {successful_saves}\n")
        f.write(f"Failed to save: {failed_saves}\n\n")

        f.write("=== Proteins to Exclude (>50% coil in complex) ===\n")
        f.write(f"Total complexes with >50% coil: {len(proteins_to_exclude)}\n\n")

        if proteins_to_exclude:
            f.write("Detailed list of complexes with >50% coil:\n")
            for protein in proteins_to_exclude:
                f.write(
                    f"  {protein['sample_id']}: coil={protein['coil_fraction']:.3f}, alpha={protein['alpha_fraction']:.3f}, beta={protein['beta_fraction']:.3f}\n"
                )
            f.write("\n")

        f.write("=== Proteins to Exclude (<10 interface residues) ===\n")
        f.write(f"Total complexes with <10 interface residues: {len(proteins_to_exclude_interface)}\n\n")

        if proteins_to_exclude_interface:
            f.write("Detailed list of complexes with <10 interface residues:\n")
            for protein in proteins_to_exclude_interface:
                f.write(
                    f"  {protein['sample_id']}: interface_residues={protein['interface_residues_total']}, binder={protein['interface_residues_binder']}, target={protein['interface_residues_target']}, density={protein['interface_density']:.3f}\n"
                )
            f.write("\n")

        f.write("=== Combined Exclusion Summary ===\n")
        f.write(f"Total unique proteins to exclude: {len(all_excluded_ids)}\n\n")

        f.write("=== Binder Secondary Structure Distribution ===\n")
        for ss_type, values in binder_ss_distribution.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                f.write(f"{ss_type}: {mean_val:.4f} ± {std_val:.4f} (n={len(values)})\n")

        f.write("\n=== Binder CA-CA Distance Metrics ===\n")
        for metric_name, values in binder_ca_metrics.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                f.write(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f} (n={len(values)})\n")

        f.write("\n=== Binder Length Distribution ===\n")
        for length, count in sorted(binder_length_distribution.items()):
            f.write(f"Length {length}: {count} structures\n")

        f.write("\n=== Target Secondary Structure Distribution ===\n")
        for ss_type, values in target_ss_distribution.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                f.write(f"{ss_type}: {mean_val:.4f} ± {std_val:.4f} (n={len(values)})\n")

        f.write("\n=== Target CA-CA Distance Metrics ===\n")
        for metric_name, values in target_ca_metrics.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                f.write(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f} (n={len(values)})\n")

        f.write("\n=== Target Length Distribution ===\n")
        for length, count in sorted(target_length_distribution.items()):
            f.write(f"Length {length}: {count} structures\n")

        f.write("\n=== Complex Secondary Structure Distribution ===\n")
        for ss_type, values in complex_ss_distribution.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                f.write(f"{ss_type}: {mean_val:.4f} ± {std_val:.4f} (n={len(values)})\n")

        f.write("\n=== Complex CA-CA Distance Metrics ===\n")
        for metric_name, values in complex_ca_metrics.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                f.write(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f} (n={len(values)})\n")

        f.write("\n=== Complex Length Distribution ===\n")
        for length, count in sorted(complex_length_distribution.items()):
            f.write(f"Length {length}: {count} structures\n")

        f.write("\n=== Interface Analysis (8Å threshold) ===\n")
        for metric_name, values in interface_metrics.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                f.write(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f} (n={len(values)})\n")

        f.write("\n=== Contact Polarity Analysis ===\n")
        if (
            interface_metrics["apolar_contacts"]
            and interface_metrics["polar_contacts"]
            and interface_metrics["charged_contacts"]
        ):
            apolar_mean = np.mean(interface_metrics["apolar_contacts"])
            polar_mean = np.mean(interface_metrics["polar_contacts"])
            charged_mean = np.mean(interface_metrics["charged_contacts"])
            total_contacts = apolar_mean + polar_mean + charged_mean

            f.write("Average Contact Distribution:\n")
            f.write(f"  Apolar: {apolar_mean:.1f} ({apolar_mean / total_contacts * 100:.1f}%)\n")
            f.write(f"  Polar: {polar_mean:.1f} ({polar_mean / total_contacts * 100:.1f}%)\n")
            f.write(f"  Charged: {charged_mean:.1f} ({charged_mean / total_contacts * 100:.1f}%)\n")
            f.write(f"  Total: {total_contacts:.1f}\n")

    logger.info(f"Detailed results saved to: {results_file}")
    logger.info(f"All files saved to: {output_dir}")

    # Generate plots
    logger.info("Generating visualization plots...")

    # Set matplotlib style for better-looking plots
    plt.style.use("default")
    sns.set_palette("husl")

    # Plot length distributions
    plot_length_distributions(
        binder_length_distribution,
        target_length_distribution,
        complex_length_distribution,
        output_dir,
        dataset_name,
    )

    # Plot secondary structure distributions
    plot_secondary_structure_distributions(
        binder_ss_distribution,
        target_ss_distribution,
        complex_ss_distribution,
        output_dir,
        dataset_name,
    )

    # Plot secondary structure proportion distributions
    plot_secondary_structure_proportion_distributions(
        binder_ss_distribution,
        target_ss_distribution,
        complex_ss_distribution,
        output_dir,
        dataset_name,
    )

    # Plot interface metrics
    plot_interface_metrics(interface_metrics, output_dir, dataset_name)

    # Plot contact polarity distributions
    plot_contact_polarity_distributions(interface_metrics, output_dir, dataset_name)

    # Plot contact polarity comparison
    plot_contact_polarity_comparison(interface_metrics, output_dir, dataset_name)

    # Plot CA-CA distance metrics
    plot_ca_distance_metrics(
        binder_ca_metrics,
        target_ca_metrics,
        complex_ca_metrics,
        output_dir,
        dataset_name,
    )

    # Plot comprehensive comparison summary
    plot_comparison_summary(
        binder_ss_distribution,
        target_ss_distribution,
        complex_ss_distribution,
        binder_length_distribution,
        target_length_distribution,
        complex_length_distribution,
        interface_metrics,
        output_dir,
        dataset_name,
    )

    logger.info("All visualization plots generated successfully!")
    logger.info(f"Plot files saved to: {output_dir}")
    logger.info(f"Detailed exclusion list saved to: {exclusion_file}")
    logger.info(f"Simple exclusion list (IDs only) saved to: {exclusion_ids_file}")
    logger.info(f"Detailed interface exclusion list saved to: {interface_exclusion_file}")
    logger.info(f"Simple interface exclusion list (IDs only) saved to: {interface_exclusion_ids_file}")
    logger.info(f"Combined exclusion list (IDs only) saved to: {combined_exclusion_ids_file}")
    logger.info(f"All secondary structure statistics saved to: {ss_csv_file}")

    if args.save_pdb_files:
        logger.info(f"Complex PDB files saved to: {complex_dir}")
    else:
        logger.info("No PDB files saved (use --save_pdb_files to save complex structures)")


if __name__ == "__main__":
    main()
