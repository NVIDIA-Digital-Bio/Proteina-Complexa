import ast
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#9a6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#808080",
    "#ffffff",
    "#000000",
    "#3C1642",
    "#086375",
    "#1DD3B0",
    "#AFFC41",
    "#B2FF9E",
    "#F9D423",
    "#FF4E50",
    "#ffe119",
    "#000000",
]


# Dictionary, for each of the relevant metrics indicates if max or min is better
METRICS_MODES = {
    "codes_2A_ca_esmfold": "max",
    "codes_2A_bb3o": "max",
    "codes_2A_all_atom_esmfold": "max",
    "des_2A_ca_esmfold": "max",
    "des_2A_mpnn_1_ca_esmfold": "max",
    "_res_diversity_foldseek_filtered_samples_codes_all_atom_frac": "max",
    "_res_diversity_foldseek_filtered_samples_codes_all_atom_nclus": "max",
    "_res_diversity_foldseek_joint_filtered_samples_codes_all_atom_frac": "max",
    "_res_diversity_foldseek_joint_filtered_samples_codes_all_atom_nclus": "max",
    "_res_diversity_mmseqs_filtered_samples_codes_all_atom_frac": "max",
    "_res_diversity_mmseqs_filtered_samples_codes_all_atom_nclus": "max",
    "_res_diversity_pairwise_tm_by_len_filtered_samples_codes_all_atom": "min",
    "_res_diversity_foldseek_complex_filtered_samples_codes_nclus": "max",
    "_res_diversity_foldseek_complex_filtered_samples_codes_frac": "max",
}


# The two variables over which we do paretto front -- for now leave at two temperatures, can be changed (more variables, different ones, etc)
# The only thing we need this for is to label the points we plot, if we don't want any labels we can just let this go
H1 = "generation_model_bb_ca_simulation_step_params_sc_scale_noise"
H2 = "generation_model_local_latents_simulation_step_params_sc_scale_noise"
# H1 = "generation_model_bb_ca_simulation_step_params_sc_scale_score"
# H2 = "generation_model_local_latents_simulation_step_params_sc_scale_score"


def filter_data(
    df: pd.DataFrame,
    column_name: str,
    tuple_keep: list | None = None,
    tuple_remove: list | None = None,
):
    """
    Filter some data from dataframe.

    Note: Only tuple_keep or tuple_remove can be specified, not both.

    Args:
        df: dataframe
        column_name: what col we're using to filter
        tuple_keep: List of values we want to keep
        tuple_remove: List of values we want to remove

    Returns:
        The filtered dataframe.
    """
    assert tuple_keep is None or tuple_remove is None, "Only one of tuple_keep or tuple_remove should be provided."
    assert not (tuple_keep is None and tuple_remove is None), "One of tuple_keep or tuple_remove must be provided."

    if tuple_keep is not None:
        # Filter the DataFrame to only keep rows with column_name values in tuple_keep
        return df[df[column_name].isin(tuple_keep)]
    else:
        # Filter the DataFrame to remove rows with column_name values in tuple_remove
        return df[~df[column_name].isin(tuple_remove)]


def fix_diversity(df: pd.DataFrame):
    """
    The diversity values come as a string containing the tuple. This function adds a two columns to the dataframe,
    extracting the first and second elements of the tuple (fraction, and number of clusters).

    Args:
        df: dataframe

    Returns:
        Updated dataframe
    """
    for c in df.columns:
        if "_res_diversity_foldseek_" in c or "_res_diversity_mmseqs_" in c:
            try:
                df[c + "_frac"] = df[c].apply(lambda x: ast.literal_eval(x)[0])
                df[c + "_nclus"] = df[c].apply(lambda x: ast.literal_eval(x)[1])
            except Exception as e:
                print(f"Error processing {c}: {e}")
    return df


def find_pareto_front(df: pd.DataFrame, M1: str, M2: str):
    """
    Given a dataframe df, it finds the rows that make the pareto front for columns M1 and M2.
    """

    def is_dominated(row, others):
        for _, other in others.iterrows():
            dominated_M1 = (METRICS_MODES[M1] == "max" and other[M1] > row[M1]) or (
                METRICS_MODES[M1] == "min" and other[M1] < row[M1]
            )
            dominated_M2 = (METRICS_MODES[M2] == "max" and other[M2] > row[M2]) or (
                METRICS_MODES[M2] == "min" and other[M2] < row[M2]
            )
            if dominated_M1 and dominated_M2:
                return True
        return False

    pareto_points = []
    for index, row in df.iterrows():
        if not is_dominated(row, df.drop(index)):
            pareto_points.append(row)
    return pd.DataFrame(pareto_points)


def plot_data(
    df: pd.DataFrame,
    M1: str,
    M2: str,
    group_by_cols: list[str],
    plot_all_points: bool,
    plot_pareto_points: bool,
    store_dir: str,
    alpha_all_points: float = 0.3,
):
    """
    Plots the data. Given a dataframe df, it groups runs by `group_by_cols`, and for each group it plots
    the pareto front for columns M1 and M2.

    Args:
        df: dataframe
        M1: col name in the x axis
        M2: col name in the y axis
        group_by_cols: list of column names, that will be used to identify each line and group runs
        plot_all_points: whether to plot all points (with transparency)
        plot_pareto_points: whether to points in the pareto front
        store_dir: Directory where to store figure
        alpha_all_points: alpha value for transparency, only used if plotting all points
    """
    plt.figure(figsize=(12, 8))
    count = 0
    for name, group in df.groupby(group_by_cols):
        color = COLORS[count % len(COLORS)]

        if plot_all_points:  # If plotting all points
            if not plot_pareto_points:
                plt.scatter(
                    group[M1],
                    group[M2],
                    alpha=alpha_all_points,
                    color=color,
                    label=name,
                    marker=SYMBOLS[name[0]],
                )
            else:
                plt.scatter(
                    group[M1],
                    group[M2],
                    alpha=alpha_all_points,
                    color=color,
                    marker=SYMBOLS[name[0]],
                )

            for _, row in group.iterrows():
                plt.text(
                    row[M1],
                    row[M2],
                    f"[{row[H1]}, {row[H2]}]",
                    fontsize=6,
                    ha="right",
                    color=color,
                    # alpha=alpha_all_points,
                )

        if plot_pareto_points:
            pareto_df = find_pareto_front(group, M1, M2)

            # Sorting the points to connect them in a meaningful order
            pareto_df.sort_values(
                by=[M1, M2],
                ascending=[METRICS_MODES[M1] == "max", METRICS_MODES[M2] == "max"],
                inplace=True,
            )

            # Plot points in the Pareto frontier
            plt.scatter(
                pareto_df[M1],
                pareto_df[M2],
                color=color,
                zorder=5,
                marker=SYMBOLS[name[0]],
            )
            plt.plot(
                pareto_df[M1],
                pareto_df[M2],
                color=color,
                zorder=3,
                label=name,
                marker=SYMBOLS[name[0]],
            )

            # for _, row in pareto_df.iterrows():
            #     plt.text(
            #         row[M1],
            #         row[M2],
            #         f"[{row[H1]}, {row[H2]}]",
            #         fontsize=10,
            #         ha="right",
            #         color=color,
            #     )

        count += 1

    # plt.title("Pareto Front")
    plt.xlabel(M1)
    plt.ylabel(M2)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.04, 1))
    # plt.tight_layout()
    fname = os.path.join(store_dir, f"{M1}_{M2}.png")
    plt.savefig(fname, bbox_inches="tight")


def plot_codes_by_len(df, group_by_cols, jitter_strength=0.05):
    """
    Plots 'codes_2A_all_atom' values grouped by group_by_cols, with slight horizontal jitter.

    Args:
        df (pd.DataFrame): Input dataframe with a 'codes_2A_all_atom' column.
        group_by_cols (list): Columns to group by.
        jitter_strength (float): Maximum random shift on x-axis (default: 0.05).
    """
    # Parse the 'codes_2A_all_atom' column from string to list
    df["codes_list"] = df["codes_2A_all_atom_esmfold"].apply(ast.literal_eval)

    # Setup the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    grouped = df.groupby(group_by_cols)
    groups_list = list(grouped)
    N = len(groups_list)  # total number of groups

    colors = plt.cm.get_cmap("tab20", N)

    for idx, (name, group) in enumerate(grouped):
        color = colors(idx)
        offset = -0.25 + (idx + 0.5) * (0.5 / N)  # Calculate group offset within [-0.25, 0.25]

        for _, row in group.iterrows():
            base_x = np.array([1, 2, 3, 4, 5])
            x = base_x + offset  # Shift all x points by the group offset
            y = row["codes_list"]
            ax.scatter(x, y, color=color, marker="x", alpha=0.7, label=name)

    # Remove duplicate labels from legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    # ax.legend(by_label.values(), by_label.keys(), title="Groups", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.legend(
        by_label.values(),
        by_label.keys(),
        title="Groups",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),  # Centered above the plot
        ncol=1,  # You can adjust number of columns
        frameon=False,
        fontsize=5,
    )

    # Aesthetic settings
    ax.set_xlabel("Index (1-5)")
    ax.set_ylabel("Value")
    # ax.set_title('codes_2A_all_atom grouped by ' + ', '.join(group_by_cols))
    ax.grid(True)
    plt.tight_layout()
    fname = os.path.join(store_dir, "codes_by_len.png")
    plt.savefig(fname)


def plot_codes_by_len_violin(df, group_by_cols):
    """
    Plots 'codes_2A_all_atom' values grouped by group_by_cols
    using violin plots (one violin per group per index).

    Args:
        df (pd.DataFrame): Input dataframe with a 'codes_2A_all_atom' column.
        group_by_cols (list): Columns to group by.
    """
    # Parse the 'codes_2A_all_atom' column from string to list
    df["codes_list"] = df["codes_2A_all_atom_esmfold"].apply(ast.literal_eval)

    # Setup the plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Group the data
    grouped = df.groupby(group_by_cols)
    groups_list = list(grouped)
    N = len(groups_list)

    colors = plt.cm.get_cmap("tab20", N)

    # Prepare storage for all violins
    all_x = []
    all_y = []
    all_colors = []
    spread = 0.4

    for idx, (name, group) in enumerate(groups_list):
        color = colors(idx)
        offset = -spread + (idx + 0.5) * (2 * spread / N)

        for i in range(5):  # 5 positions: 1,2,3,4,5
            # Collect all the ith values for this group
            y = group["codes_list"].apply(lambda lst: lst[i]).values
            x_pos = (i + 1) + offset  # (i+1) because x=1,2,3,4,5

            all_x.append(np.full_like(y, x_pos))
            all_y.append(y)
            all_colors.append(color)

    # Now plot all violins
    for x, y, color in zip(all_x, all_y, all_colors, strict=False):
        parts = ax.violinplot(
            y,
            positions=[x[0]],
            widths=2 * spread / N,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_edgecolor("black")
            pc.set_alpha(0.7)

    # Create legend manually
    handles = []
    labels = []
    for idx, (name, _) in enumerate(groups_list):
        handles.append(plt.Line2D([0], [0], color=colors(idx), lw=6))
        labels.append(str(name))

    ax.legend(
        handles,
        labels,
        title="Groups",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=1,  # min(1, len(labels)),
        frameon=False,
        fontsize=5,
    )

    # Draw vertical lines at x=1,2,3,4,5
    for i in range(1, 6):
        ax.axvline(x=i, color="gray", linestyle="--", linewidth=0.5)

    # Aesthetic settings
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels([1, 2, 3, 4, 5])
    ax.set_xlabel("Index (1-5)")
    ax.set_ylabel("Value")
    # ax.set_title('Violin plot of codes_2A_all_atom grouped by ' + ', '.join(group_by_cols))
    ax.set_title("codes_2A_all_atom by length")
    ax.grid(True)

    plt.tight_layout()
    fname = os.path.join(store_dir, "codes_by_len_violin.png")
    plt.savefig(fname, bbox_inches="tight")


# What runs
run_name = [
    # ("local_latents_160m_tri_AE_zd_8_kl_4_tr_tr_60m_bis-inference-traditional-temps-2025_02_09_02_30_07", "kl4_temps"),
    # ("local_latents_160m_AE_zd_8_kl_2_tr_tr_60m_rel_io-inference-traditional-kl2-comp", "kl2_rel"),
    # ("local_latents_160m_AE_zd_8_kl_2_tr_tr_60m-inference-traditional-kl2-comp-2025_02_10_16_25_00", "kl2_abs"),
    # ("local_latents_160m_tri_AE_zd_8_kl_4_tr_tr_60m_bis-inference-traditional-temps-2025_02_09_02_30_07", "kl4_abs"),
    # ("local_latents_160m_AE_zd_8_kl_4_tr_tr_60m-inference-traditional-ckpts", "kl4_ckpts"),
    # ("local_latents_160m_tri_AE_zd_8_kl_4_tr_tr_60m_3recycle_bisbb-inference-traditional-temps-2025_02_09_21_02_16", "kl4_rec"),
    # ("local_latents_160m_AE_zd_8_kl_4_tr_tr_60m-inference-traditional-2025_02_03_19_47_05", "kl4_orig"),
    # ("local_latents_160m_AE_zd_8_kl_3_tr_tr_60m-inference-traditional-2025_02_03_21_31_32", "kl3_orig"),
    # ("local_latents_160m_AE_zd_8_kl_2_tr_tr_60m-inference-traditional-2025_02_04_00_55_57", "kl2_orig"),
    # ("local_latents_160m_AE_zd_8_kl_1_tr_tr_60m-inference-traditional-2025_02_04_01_21_34", "kl1_orig"),
    # ("local_latents_160m_AE_zd_8_kl_4_tr_tr_60m-inference-traditional-more-schs-2025_02_06_15_41_07", "kl4_schs"),
    # ("local_latents_160m_tri_AE_zd_8_kl_4_tr_tr_60m_bis-inference-traditional-temps-2025_02_09_02_30_07", "kl4_tri")
    # ("local_latents_160m_AE_zd_8_kl_2_tr_60m_ff_7m-inference-traditional-ff-fin-2025_02_11_13_37_24", "ff_kl2"),
    # ("local_latents_160m_AE_zd_8_kl_3_tr_60m_ff_7m-inference-traditional-ff-fin-2025_02_11_13_52_28", "ff_kl3"),
    # ("local_latents_160m_zd_8_kl_5_tr_tr_130m_relio-inference-traditional-testttt-2025_02_14_21_19_48", "kl5_z8_130_rel"),
    # ("local_latents_160m_zd_8_kl_5_tr_tr_130m_abso-inference-traditional-testttt-2025_02_15_11_22_54", "kl5_z8_130_abs"),
    # ("local_latents_160m_zd_8_kl_4_tr_tr_130m_relio-inference-traditional-testttt-2025_02_14_21_16_04", "kl4_z8_130_rel"),
    # ("local_latents_160m_zd_8_kl_4_tr_tr_130m_abso-inference-traditional-testttt-2025_02_15_12_25_26", "kl4_z8_130_abs"),
    # ("local_latents_160m_zd_8_kl_3_tr_tr_130m_relio-inference-traditional-testttt-2025_02_15_14_27_12", "kl3_z8_130_rel"),
    # ("local_latents_160m_zd_4_kl_5_tr_tr_130m_relio-inference-traditional-testttt-2025_02_14_22_22_03", "kl5_z4_130_rel"),
    # ("local_latents_160m_zd_4_kl_4_tr_tr_130m_relio-inference-traditional-testttt-2025_02_14_21_42_35", "kl4_z4_130_rel"),
    # ("local_latents_160m_zd_8_kl_5_tr_tr_130m_relio-inference-traditional-testttt2-2025_02_15_18_55_41", "kl5_z8_130_rel2"),
    # ("local_latents_160m_zd_8_kl_5_tr_tr_130m_relio-inference-traditional-testttt3", "kl5_z8_130_rel3"),
    # ("local_latents_160m_zd_8_kl_2_tr_tr_130m_relio-inference-traditional-testttt2-2025_02_16_14_21_56", "kl2_z8_130_rel"),
    # ("local_latents_160m_zd_8_kl_5_tr_tr_130m_relio_decloss_mean_1_1_tlim_0505-inference-traditional-auxdecloss", "aux_loss_dec_0505"),
    # ("local_latents_160m_zd_8_kl_5_tr_tr_130m_relio_decloss_mean_1_1_tlim_0402-inference-traditional-auxdecloss-2025_02_23_19_40_19", "aux_loss_dec_0402mean"),
    # ("local_latents_160m_zd_8_kl_5_tr_tr_130m_relio_g2_maxloopconsec_15_maxloopprop_0p5-inference-traditional-auxdecloss-2025_02_24_11_48_47", "kl5_z8_rel_15consecl_p5maxloop"),
    # ("local_latents_160m_zd_8_kl_5_tr_tr_130m_relio_g2_maxloopconsec_15-inference-traditional-filtersdata-2025_02_25_02_26_37", "kl5_z8_rel_15consec_filter"),
    # ("local_latents_160m_zd_8_kl_5_tr_tr_130m_relio_g2_maxloopprop_0p5-inference-traditional-filtersdata", "kl5_z8_rel_p5prop_filter"),
    # ("local_latents_160m_zd_8_kl_5_tr_tr_130m_relio_g2_maxloopconsec_15_maxloopprop_0p5-inference-traditional-2025_02_24_11_48_47", "kl5_z8_rel_p5prop_15consec_filter"),
    # ("local_latents_160m_zd_8_kl_5_tr_tr_130m_relio-inference-traditional-filtersdata", "kl5_z8_rel_filtercomp_nofilter"),
    # ("LL_160m_zd_8_kl_5_tr_tr_130m_relio_360k_smallrelseqsep_n_idx-inference-traditional-ckptAE-idx-relseqsep-2025_02_28_21_20_46", "LL_smallrel_idx_360kae"),
    # ("LL_160m_zd_8_kl_5_tr_tr_130m_relio_440k_smallrelseqsep_n_idx-inference-traditional-ckptAE-idx-relseqsep-2025_02_28_17_55_10", "LL_smallrel_idx_440kae"),
    # ("LL_160m_zd_8_kl_5_tr_tr_130m_relio_400k_smallrelseqsep_n_idx-inference-traditional-ckptAE-idx-relseqsep-2025_03_01_02_45_11", "LL_smallrel_idx_400kae"),
    # ("LL_160m_zd_8_kl_5_tr_tr_130m_relio_400k_resindex-inference-traditional-ckptAE-idx-relseqsep-2025_03_01_10_03_11", "LL_idx_400kae"),
    # ("LL_160m_zd_8_kl_5_tr_tr_130m_relio_400k_smallrelseqsep-inference-traditional-ckptAE-idx-relseqsep-2025_03_01_16_53_20", "LL_noidx_smallseqsep_400kae"),
    # ("local_latents_160m_zd_8_kl_5_tr_tr_130m_relio-inference-kl5-finegrained-a-2025_03_05_16_06_27", "kl5_a_2"),
    # ("local_latents_160m_zd_8_kl_5_tr_tr_130m_relio-inference-kl5-finegrained-b-2025_03_05_22_55_41", "kl5_b_2"),
    # ("local_latents_160m_zd_8_kl_5_tr_tr_130m_relio-inference-kl5-finegrained-c-2025_03_06_05_12_25", "kl5_c_2"),
    # ("LL_160m_zd_8_kl_5_tr_tr_130m_relio_440k_resindex-inference-2025_03_10_19_36_28", "LL_idx_440k"),
    # ("LL_160m_zd_8_kl_5_tr_tr_130m_relio_440k_no_resindex-inference-2025_03_10_22_35_20", "LL_no_idx_440k"),
    # ("LL_160m_zd_8_kl_5_tr_tr_130m_relio_440k_no_resindex-inference-noodeever-2025_03_14_14_30_00", "LL_no_idx_no_ode_440k"),
    # ("LL_160m_zd_8_kl_5_tr_tr_130m_relio_440k_no_resindex-inference-noodeever-otherckpts-2025_03_14_17_12_19", "LL_no_idx_no_ode_440k_oc"),  # (other good-ish from before)
    # ("LL_160m_zd_8_kl_5_tr_tr_130m_relio_440k_no_resindex_seedfixxx-inference-noodeever", "LL_no_idx_no_ode_440k_seedfix"),
    # ("LL_40m_zd_8_kl_5_tr_tr_130m_relio_440k_no_resindex-inference-noodeever-2025_03_18_15_27_15", "LL_40m_no_idx_no_ode_440k"),
    # ("LL_160m_zd_8_kl_5_tr_tr_130m_relio_440k_no_resindex_crop128_256-inference-noodeever-2025_03_19_06_47_22", "LL_crop_128_256"),
    # ("LL_160m_zd_8_kl_5_tr_tr_130m_relio_440k_no_resindex_crop128_256-inference-noodeeverrr-2025_03_21_19_29_59", "LL_crop_128_256_oc"),
    # ("LL_160m_zd_8_kl_5_tr_tr_130m_440k_beta0p01-inference-noodeever-2025_03_26_17_25_15", "LL_beta0p01"),  # More diverse at somewhat lower codes
    # ("LL_160m_zd_8_kl_5_tr_tr_130m_FFdec_7m-inference-noodeever-2025_03_26_20_48_43", "LL_FFdec_7m"),  # Pretty bad
    # ("LL_160m_zd_8_kl_5_tr_tr_130m_440k_abr3-inference-noodeever-2025_03_27_02_20_29", "LL_abr3"),
    # ("LL_160m_zd_8_kl_5_tr_tr_130m_invfeatsE-inference-noodeever-2025_03_27_21_06_34", "LL_invfeatsE_tg1"),
    # ("LL_160m_zd_8_kl_5_tr_tr_130m_AE500k_new-inference-noodeever-2025_04_02_05_22_16", "newAE_kl5_130"),
    # ("LL_160m_zd_8_kl_5_tr_tr_30m_AE500k_new-inference-noodeever-2025_04_02_14_05_11", "newAE_kl5_30"),
    # ("LL_160m_zd_8_kl_4_tr_tr_130m_AE500k_new-inference-noodeever-2025_04_02_23_36_27", "newAE_kl4_130"),  # SOTA for now
    # ("LL_160m_zd_8_kl_5_tr_tr_30m_AE500k_new-inference-llvfss-2025_04_03_06_20_00", "newAE_kl5_30_llvfss"),
    # ("LL_160m_zd_8_kl_4_tr_tr_30m_AE500k_new-inference-2025_4_4_18_48_35", "newAE_kl4_30"),
    # ("LL_160m_zd_8_kl_4_tr_tr_130m_AE500k_new__beta1p9ll_from120k-inference-moretemps-2025_04_06_02_35_23", "newAE_kl4_130_beta1p9ll_cf"),
    # ("LL_160m_zd_8_kl_4_tr_tr_130m_AE500k_new-inference-moretemps-2025_04_06_00_56_49", "newAE_kl4_130_moretemps_diffseed"),
    # ("LL_160m_tri_zd_8_kl_5_tr_tr_130m_440k_5nn-inference-2025_04_07_03_41_52", "LL_no_idx_no_ode_440k_tri_nocoilf"),
    # ("LL_160m_zd_8_kl_4_tr_tr_130m_AE500k_new__beta1p9ll_from120k_nocoilfilter-inference-2025_04_08_01_41_06", "AEkl4_130_b1p9ll"),
    # ("LL_160m_zd_8_kl_4_tr_tr_130m_AE500k_new__beta1p5ll_from120k_nocoilfilter-inference-2025_04_08_04_03_37", "AEkl4_130_b1p5ll"),
    # ("LL_160m_zd_8_kl_4_tr_tr_130m_AE500k_new-inference-kl4testlen-n-ft-100500-2025_04_09_02_03_28", "lp_kl4_130_nft_100500"),
    # ("LL_160m_zd_8_kl_4_tr_tr_130m_AE500k_new_finetune200kema_32_512_lr02_6nodess-inference-kl4testlen-y-ft-50-250-2025_04_09_03_32_27", "lp_kl4_130_yft_50250"),
    # ("LL_160m_zd_8_kl_4_tr_tr_130m_AE500k_new-inference-kl4testlen-n-ft-50250-2025_04_09_00_00_45", "lp_kl4_130_nft_50250"),
    # ("LL_160m_zd_8_kl_4_tr_tr_130m_AE500k_new_finetune200kema_32_512_lr02_6nodess-inference-kl4testlen-y-ft-100-500-2025_04_09_07_43_53", "lp_kl4_130_yft_100500"),
    # ("LL_len512scratch_6nds_160m_aef_kl4_z8_allfeats_ft500ke_32_512_140ke_b1p5ll-inference-ftae512-100-500-b1p5ll-2025_04_14_01_39_32", "lp_kl4_130_ftae_140ke_scrld_100500"),  # Here
    # ("LL_len512scratch_6nds_160m_aef_kl4_z8_allfeats_ft500ke_32_512_140ke_b1p5ll-inference-ftae512-50-250-b1p5ll-2025_04_14_13_19_45", "lp_kl4_130_ftae_140ke_scrld_50250"),
    # ("LL_160m_aef_kl4_z8_allfeats_500ke_b1p5ll_beta0p01-inference-kl4sota50250-50250-betamuch-2025_04_14_20_37_59", "lp_kl4_130_50250_betamuch"),
    # ("LL_160m_aef_kl4_z8_allfeats_ft500ke_32_512_60ke_b1p5ll-inference-AEkl4ft500-LD50250-2025_04_15_05_53_44", "expl_llftae_60ke_50250"),
    # ("LL_160m_aef_kl4_z8_allfeats_ft500ke_32_512_100ke_b1p5ll-inference-AEkl4ft500-LD50250-2025_04_15_04_08_02", "expl_llftae_100ke_50250"),
    # ("LL_160m_aef_kl4_z8_allfeats_ft500ke_32_512_140ke_b1p5ll-inference-AEkl4ft500-LD50250-2025_04_15_05_53_43", "expl_llftae_140ke_50250"),
    # ("LL_160m_zd_8_kl_4_sotaAE256_EDMM-inference-50250-2025_04_16_16_40_33", "edm_lp_kl4_130_50250_edmm"),
    # ("LL_160m_zd_8_kl_4_sotaAE256_EDMM-inference-2-50250-2025_04_17_16_01_38", "edm2_lp_kl4_130_2_50250"),
    # ("LL_160m_zd_8_kl_4_sotaAE256_EDMM-inference-3-50250-2025_04_18_01_24_57", "edm3_lp_kl4_130_3_50250"),
    # ("LL_160m_aeft_140ke_kl4_z8_allfeats_b1p5ll_betamuch_6n-inference-aeft140k_betamuch-2025_04_20_00_36_20", "LL_512_aeft_140ke_betamuch"),
    # ("LL_160m_aeft_140ke_kl4_z8_allfeats_b1p5ll_betamuch_6n-inference-aeft140k_betamuch-contd-2025_04_23_12_27_32", "LL_512_aeft_140ke_betamuch_contd"),  #  HERE BIS
    # ("LL_len512scratch_6nds_160m_aef_kl4_z8_allfeats_ft500ke_32_512_140ke_b1p5ll_ft_unif_len_270keorig-inference-aeft140k-2025_04_20_09_25_49", "LL_512_ft_140ke_uniflen_ft"),
    # ("LL_len512scratch_6nds_160m_aef_kl4_z8_allfeats_ft500ke_32_512_140ke_b1p5ll_ft_unif_len_270keorig-inference-aeft140k-contd-2025_04_23_06_43_50", "LL_512_ft_140ke_uniflen_ft_contd"),
    # ("LL_160m_sinembpair_aef_kl4_z8_allfeats_500ke_b1p5ll-inference-2025_04_23_11_25_25", "LL_512_sinembpair_500ke"),
    # ("LL_160m_sinemb_aef_kl4_z8_allfeats_500ke_b1p5ll-inference-AEkl4ft500-LD50250-2025_04_15_14_58_39", "LL_512_sinemb_500ke"),
    # ("LL_160m_aeft_140ke_kl4_z8_allfeats_b1p5ll_betamuch_6n-inference-50250-2025_05_01_02_57_09", "LL_512_aeft_140ke_betamuch_50250"),  # 50 - 250 beta much trained up to 512
    # ("LL_len512scratch_6nds_160m_aef_kl4_z8_allfeats_ft500ke_32_512_140ke_b1p5ll_ft_unif_len_270keorig-inference-50250-2025_05_01_05_46_46", "LL_512_ft_140ke_uniflen_50250"),  # 50 - 250 beta much ft uni length up to 512
    # ("LL_len512scratch_6nds_160m_sinemb_betamuch_aef_kl4_z8_allfeats_32_512_140ke_b1p5ll-inference-100500-2025_05_01_19_11_07", "LL_512_sinemb_betamuch_100500"),
    # ("L512_ft_betamuch_2_betamuchuniflen_aef_kl4_z8_allfeats_32_512_140ke_b1p5ll-inference-50250-2025_05_03_12_23_20", "LL_512_ft_betamuchunilen_50250"),  # 50 - 250 beta much trained up to 512 with unif length
    # ("L512_ft_betamuch_2_betamuchuniflen_aef_kl4_z8_allfeats_32_512_140ke_b1p5ll-inference-100500-2025_05_03_08_27_16", "LL_512_ft_betamuchunilen_100500"),
    # ("L512_alpha_ft_2_beta_kl4_z8_allfeats_ft500ke_32_512_140ke_b1p5ll-inference-50250-2025_05_04_09_52_20", "alpha_2_beta_512_50250"),  # a2b
    # ("L512_alpha_ft_2_beta_kl4_z8_allfeats_ft500ke_32_512_140ke_b1p5ll-inference-100500-2025_05_04_06_51_58", "alpha_2_beta_512_100500"),  # a2b
    # ("L512_scr_aeft_140ke_betamuch_coilfilters-inference-100500-2025_05_09_06_53_09", "LL_512_betamuch_coilfilters_100500"),  # Normal, beta, finetuned on coil filters. Cont of HERE BIS
    # ("L512_scr_aeft_140ke_betamuch_coilfilters-inference-50250-2025_05_10_01_39_46", "LL_512_betamuch_coilfilters_50250"),  # Normal, beta, finetuned on coil filters. Cont of HERE BIS
    # ("L512_scr_aeft_140ke_betamuch_coilfilters-inference-100500-test-fin1_base_c-600steps-2025_05_15_01_53_09", "fin1_base_c_600steps"),
    # ("L512_scr_aeft_140ke_betamuch_coilfilters-inference-100500-test-fin1_base_c-1otschll-2025_05_14_22_58_06", "fin1_base_c_1otschll"),
    # PAPER
    # ("L512_scr_tri_aeft_140ke_betamuch-inference-100500-2025_05_15_03_21_09", "L512_tri_aeft_140ke_betamuch_coilfilters"),
    # ("L512_scr_aeft_140ke_betamuch_coilfilters-inference-100500-test-fin1_base_noafdbnov-2025_05_14_11_58_54", "fin1_base"),  # Normal, beta, finetuned on coil filters. Cont of HERE BIS
    # Paper ablations
    # ("LL_160m_zd_8_kl_5_tr_tr_130m_FFdec_7m-inference-noodeever-2025_03_26_20_48_43", "ABL_FFdec_7m_kl5"),
    # Noise scales bbca [0.15 0.2  0.25 0.3  0.35]
    # Noise scales local [0.025 0.05  0.1   0.15 ]
    # ("LL_160m_zd_8_kl_5_tr_tr_130m__CAencoded-inference-noodeeverrr-2025_04_14_17_21_56", "ABL_CAenc_kl5"),
    # ("LL_160m_zd_8_kl_4_tr_tr_130m_AE500k_new-inference-noodeever-2025_04_02_23_36_27", "ABL_newAE_kl4_130"),  # KL4 comparison -- (SOTA for now)
    # Noise scales bbca [0.15 0.2  0.25 0.3  0.35]
    # Noise scales local [0.025 0.05  0.1   0.15 ]
    # ("ligand_binder_160M_ae_genie512iclr_pretrain_genie2_256_6n_bs18_betalatenttime_cacenter_oldseed_noseqidx-inference-2025_08_31_11_52_52", "512AE_256_noseqidx"),
    # ("ligand_binder_160M_ae_genie512iclr_pretrain_genie2_256_6n_bs18_betalatenttime_cacenter_oldseed-inference-2025_08_31_11_42_42", "512AE_256"),
    (
        "ligand_binder_160M_ae_genie512iclr_pretrain_genie2_512_6n_bs5_betalatenttime_cacenter_oldseed-inference-2025_08_31_20_40_15",
        "512AE_512",
    ),
    (
        "ligand_binder_160M_ae_genie512iclr_pretrain_genie2_512_6n_bs5_betalatenttime_cacenter_oldseed_noseqidx-inference-2025_08_31_22_22_12",
        "512AE_512_noseqidx",
    ),
]

SYMBOLS = {
    "512AE_512": "o",
    "512AE_512_noseqidx": "x",
}

# How to form different groups
group_by_cols = [
    "run_name",
    "ckpt_name",
    "generation_model_bb_ca_gt_mode",
    "generation_model_bb_ca_gt_p",
    "generation_model_local_latents_gt_mode",
    "generation_model_local_latents_gt_p",
    "generation_model_bb_ca_schedule_mode",
    "generation_model_bb_ca_schedule_p",
    "generation_model_local_latents_schedule_mode",
    "generation_model_local_latents_schedule_p",
    "generation_args_self_cond",
    # "generation_n_recycle",
]

# Filter certain parameters
filters = [
    # ("generation_model_bb_ca_schedule_mode", ["log"], None),
    # ("generation_model_local_latents_schedule_mode", ["power"], None),
    # ("generation_model_bb_ca_schedule_p", [2], None),
    # ("generation_model_local_latents_schedule_p", [2], None),
    # ("generation_model_bb_ca_gt_mode", ["1/t"], None),
    # ("generation_model_local_latents_gt_mode", ["tan"], None),
    # ("ckpt_name", ["chk_epoch=00000075_step=000000090000-EMA.ckpt"], None),
    # ("ckpt_name", ["chk_epoch=00000083_step=000000100000-EMA.ckpt"], None),
    # ("ckpt_name", ["chk_epoch=00000070_step=000000120000-EMA.ckpt"], None),
    # ("generation_n_recycle", [3], None),
    # ("generation_model_local_latents_simulation_step_params_sc_scale_noise", None, [0.05, 0.15]),
]

# Get all dataframes with results
ident = "_".join([v[1] for v in run_name])
all_dfs = []
for rn, rid in run_name:
    df = pd.read_csv(f"results_downloaded/{rn}/a_results_processed/res_all.csv")
    df["run_name"] = rid

    # Add if some entriesd missing
    for c in group_by_cols:
        if c not in df.columns:
            df[c] = -1

    # Remove rows with designability 0
    df = df[df["des_2A_ca_esmfold"] != 0]
    all_dfs.append(df)

df = pd.concat(all_dfs)
store_dir = f"results_downloaded/figures_plotting/{ident}"
os.makedirs(store_dir, exist_ok=True)

for f in filters:
    cf, fk, fr = f
    if fk is None:
        df = filter_data(df, cf, tuple_keep=None, tuple_remove=fr)
    else:
        df = filter_data(df, cf, tuple_keep=fk, tuple_remove=None)

# print("Noise scales bbca", df["generation_model_bb_ca_simulation_step_params_sc_scale_noise"].unique())
# print("Noise scales local", df["generation_model_local_latents_simulation_step_params_sc_scale_noise"].unique())

# For each diversity metric that's a tuple (div, nclus, nels), this adds two metrics, the actual fraction
# and number of clusters ("_nclus" suffix) and the actual float ("_frac" suffix).
df = fix_diversity(df)

# Which plots I want
# # (metric_x, metric_y, all_points, pareto_points)
# axes = [
#     (
#         "codes_2A_ca_esmfold",
#         "_res_diversity_foldseek_filtered_samples_codes_all_atom_frac",
#         True,
#         True,
#     ),
#     (
#         "codes_2A_all_atom_esmfold",
#         "_res_diversity_foldseek_filtered_samples_codes_all_atom_frac",
#         True,
#         True,
#     ),
#     (
#         "des_2A_ca_esmfold",
#         "_res_diversity_foldseek_filtered_samples_codes_all_atom_frac",
#         True,
#         True,
#     ),
#     (
#         "codes_2A_ca_esmfold",
#         "_res_diversity_foldseek_filtered_samples_codes_all_atom_nclus",
#         True,
#         True,
#     ),
#     (
#         "codes_2A_all_atom_esmfold",
#         "_res_diversity_foldseek_filtered_samples_codes_all_atom_nclus",
#         True,
#         True,
#     ),
#     (
#         "des_2A_ca_esmfold",
#         "_res_diversity_foldseek_filtered_samples_codes_all_atom_nclus",
#         True,
#         True,
#     ),
#     (
#         "codes_2A_all_atom_esmfold",
#         "_res_diversity_foldseek_joint_filtered_samples_codes_all_atom_frac",
#         True,
#         True,
#     ),
#     (
#         "codes_2A_all_atom_esmfold",
#         "_res_diversity_foldseek_joint_filtered_samples_codes_all_atom_nclus",
#         True,
#         True,
#     ),
#     (
#         "codes_2A_all_atom_esmfold",
#         "_res_diversity_mmseqs_filtered_samples_codes_all_atom_frac",
#         True,
#         True,
#     ),
#     (
#         "codes_2A_all_atom_esmfold",
#         "_res_diversity_mmseqs_filtered_samples_codes_all_atom_nclus",
#         True,
#         True,
#     ),
#     (
#         "codes_2A_all_atom_esmfold",
#         "_res_diversity_mmseqs_filtered_samples_codes_all_atom_nclus",
#         True,
#         True,
#     ),
#     (
#         "codes_2A_all_atom_esmfold",
#         "_res_diversity_pairwise_tm_by_len_filtered_samples_codes_all_atom",
#         True,
#         True,
#     ),
#     (
#         "codes_2A_all_atom_esmfold",
#         "_res_diversity_mmseqs_filtered_samples_codes_all_atom_nclus",
#         True,
#         True,
#     ),
#     (
#         "codes_2A_all_atom",
#         "_res_diversity_mmseqs_filtered_samples_codes_all_atom_nclus",
#         True,
#         True,
#     ),
#     (
#         "codes_2A_all_atom",
#         "_res_diversity_pairwise_tm_by_len_filtered_samples_codes_all_atom",
#         True,
#         True,
#     ),
#     ("codes_2A_ca", "codes_2A_all_atom", True, False),
#     ("codes_2A_ca", "des_2A_mpnn_1", True, False),
#     ("codes_2A_all_atom", "des_2A_mpnn_1", True, False),
#     (
#         "codes_2A_all_atom",
#         "_res_ss_biot_alpha_filtered_samples_codes_all_atom",
#         True,
#         False,
#     ),
#     (
#         "codes_2A_all_atom",
#         "_res_ss_biot_beta_filtered_samples_codes_all_atom",
#         True,
#         False,
#     ),
#     (
#         "codes_2A_all_atom",
#         "_res_ss_biot_coil_filtered_samples_codes_all_atom",
#         True,
#         False,
#     ),
#     ("codes_2A_ca", "_res_ss_biot_alpha_filtered_samples_codes_all_atom", True, False),
#     ("codes_2A_ca", "_res_ss_biot_beta_filtered_samples_codes_all_atom", True, False),
#     ("codes_2A_ca", "_res_ss_biot_coil_filtered_samples_codes_all_atom", True, False),
#     ("des_2A", "des_2A_mpnn_1", True, False),
# ]
axes = [
    (
        "codes_2A_all_atom_esmfold",
        "_res_diversity_foldseek_complex_filtered_samples_codes_nclus",
        True,
        True,
    ),
    ("codes_2A_all_atom_esmfold", "des_2A_ca_esmfold", True, True),
]
# Plot
for M1, M2, all_points, pareto_points in axes:
    for M in [M1, M2]:
        do_plot = True
        if M not in df.columns:
            print(f"{M} not in dataframe, skipping...")
            do_plot = False

    if do_plot:
        plot_data(
            df=df,
            M1=M1,
            M2=M2,
            group_by_cols=group_by_cols,
            plot_all_points=all_points,
            plot_pareto_points=pareto_points,
            store_dir=store_dir,
        )

# Get all dataframes with results by length
# ident = "_".join([v[1] for v in run_name])
# all_dfs = []
# for rn, rid in run_name:
#     df = pd.read_csv(
#         f"results_downloaded/{rn}/a_results_processed/res_codesignability_all_atom_len.csv"
#     )
#     df["run_name"] = rid

#     # Add if some entriesd missing
#     for c in group_by_cols:
#         if c not in df.columns:
#             df[c] = -1

#     all_dfs.append(df)

# df = pd.concat(all_dfs)

# plot_codes_by_len(
#     df=df,
#     group_by_cols=group_by_cols,
# )

# plot_codes_by_len_violin(
#     df=df,
#     group_by_cols=group_by_cols,
# )
