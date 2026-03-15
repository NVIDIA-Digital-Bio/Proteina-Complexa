import os

import matplotlib.pyplot as plt
import numpy as np

# Create output directory
os.makedirs("./scaling_results", exist_ok=True)

targets = [
    "IFNAR2",
    "BHRF1",
    "BBF14",
    "DerF21",
    "TrkA",
    "PD1",
    "Insulin",
    "DerF7",
    "PDL1",
    "IL7RA",
    "CrSAS6",
    "Claudin1",
    "VEGFA",
    "SpCas9",
    "SC2RBD",
    "CbAgo",
    "CD45",
    "BetV1",
    "HER2_AAV",
]

# Data (already in the right order for each method)
ours_mpnn = [51, 26, 24, 31, 25, 27, 16, 24, 16, 6, 12, 4, 5, 3, 0, 0, 0, 0, 0]
ours_mpnn_fixed = [52, 29, 25, 31, 20, 21, 20, 16, 14, 12, 8, 4, 2, 4, 0, 0, 0, 0, 0]
ours_self = [39, 21, 20, 22, 16, 15, 15, 10, 9, 6, 3, 2, 1, 1, 0, 0, 0, 0, 0]

apm_mpnn = [3, 20, 0, 11, 0, 1, 5, 0, 1, 1, 1, 11, 0, 0, 0, 3, 0, 0, 0]
apm_mpnn_fixed = [3, 10, 0, 8, 0, 0, 3, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0]
apm_self = [1, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

rf_diffusion = [11, 5, 2, 4, 7, 7, 8, 5, 8, 4, 8, 2, 0, 3, 2, 0, 1, 3, 1]
protpardelle_1c = [2, 4, 0, 4, 0, 0, 1, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

# Define method groups with consistent colors
mpnn_methods = [
    ("Complexa", ours_mpnn, "#E57373"),
    ("APM", apm_mpnn, "#81C784"),
    ("RFDiffusion", rf_diffusion, "#64B5F6"),
    ("Protpardelle-1c", protpardelle_1c, "#8e7cc3"),
]

mpnn_fixed_methods = [
    ("Complexa", ours_mpnn_fixed, "#E57373"),
    ("APM", apm_mpnn_fixed, "#81C784"),
]

self_methods = [
    ("Complexa", ours_self, "#E57373"),
    ("APM", apm_self, "#81C784"),
]

easy_count = 12
bar_width = 0.2
group_width = bar_width * 4 + 0.2  # 4 methods for MPNN, 2 for others
index = np.arange(len(targets)) * group_width

# Create figure with three subplots in a column
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)


def plot_subplot(ax, methods, title):
    # Draw bars per method, side by side within each target group
    for i, (label, data, color) in enumerate(methods):
        bar_pos = index + (i - 1.5) * bar_width  # Center grouping
        ax.bar(bar_pos, data, bar_width, color=color, alpha=0.8)

    # Ticks and labels adjustments
    ax.set_ylabel("Unique successes", fontsize=14)
    ax.set_xlim(index[0] - group_width / 2, index[-1] + group_width / 2)

    # Vertical dotted line to separate easy/hard
    ax.axvline(
        index[easy_count] - group_width / 2,
        color="black",
        linestyle="dotted",
        linewidth=2,
        alpha=0.5,
    )

    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="y", labelsize=14)

    # Add title as text annotation in top right corner
    ax.text(0.98, 0.95, title, transform=ax.transAxes, fontsize=18, ha="right", va="top")


# Plot each subplot
plot_subplot(ax1, mpnn_methods, "MPNN")
plot_subplot(ax2, mpnn_fixed_methods, "MPNN-FI")
plot_subplot(ax3, self_methods, "Self")

# Add x-axis labels only to the bottom subplot
ax3.set_xlabel("", fontsize=14)
ax3.set_xticks(index)
ax3.set_xticklabels(targets, fontsize=14, rotation=45, ha="center")

# Add legend for the method variants
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor="#E57373", label="Complexa"),
    Patch(facecolor="#81C784", label="APM"),
    Patch(facecolor="#64B5F6", label="RFDiffusion"),
    Patch(facecolor="#8e7cc3", label="Protpardelle-1c"),
]
fig.legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.95),
    ncol=4,
    fontsize=14,
    frameon=False,
)

plt.tight_layout()
plt.subplots_adjust(left=0.05, right=0.98, bottom=0.15, top=0.90)

# Save plots
plt.savefig("./scaling_results/hits.png", dpi=200, bbox_inches="tight")
plt.savefig("./scaling_results/hits.pdf", bbox_inches="tight")

print("Plots saved to ./scaling_results/hits.png and ./scaling_results/hits.pdf")
