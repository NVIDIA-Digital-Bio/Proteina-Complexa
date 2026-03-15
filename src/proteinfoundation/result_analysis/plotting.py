import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
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
    "#000000",
]


def filter_data(df, column_name, tuple_keep=None, tuple_remove=None):
    assert tuple_keep is None or tuple_remove is None, "Only one of tuple_keep or tuple_remove should be provided."
    assert not (tuple_keep is None and tuple_remove is None), "One of tuple_keep or tuple_remove must be provided."

    if tuple_keep is not None:
        # Filter the DataFrame to only keep rows with column_name values in tuple_keep
        return df[df[column_name].isin(tuple_keep)]
    else:
        # Filter the DataFrame to remove rows with column_name values in tuple_remove
        return df[~df[column_name].isin(tuple_remove)]


def plot_data(
    run_name,
    df_orig,
    group_columns,
    x_column,
    y_column,
    log_scale_y=False,
    target_name=None,
    aggregate_by_target=False,
):
    assert "nsamples_all" not in x_column, "nsamples_all cannot be x axis"

    df = df_orig.copy()
    if x_column == "ckpt_name":
        x_column = "ckpt_steps (k)"
    df["ckpt_no"] = df["ckpt_name"].apply(lambda x: int(x.split("=")[-1].split(".")[0]) // 1000)
    df["ckpt_steps (k)"] = df["ckpt_no"].apply(lambda x: str(x) + "k")

    # Create unified x-axis based on all available ckpt_no values, sorted
    all_ckpt_nos = sorted(df["ckpt_no"].unique())
    [str(ckpt_no) + "k" for ckpt_no in all_ckpt_nos]

    # Potentially filter here what to group by if it fails
    df_grouped = df.groupby(group_columns, dropna=False)

    plt.figure(figsize=(10, 6))

    plot_straight_line = True
    for col in (x_column, y_column):
        if (
            ("_foldseek_" in col or "_maxcluster_" in col)
            and "_res_diversity_foldseek_des_cluster" not in col
            and "_res_diversity_foldseek_binder_filtered_samples_binder_success_cluster" not in col
            and "_res_diversity_foldseek_binder_filtered_samples_binder_success_self_cluster" not in col
            and "_res_diversity_foldseek_binder_filtered_samples_binder_success_mpnn_cluster" not in col
            and "_res_diversity_foldseek_binder_filtered_samples_binder_success_mpnn_fixed_cluster" not in col
            and "_res_diversity_foldseek_binder_filtered_samples_binder_success_self_total_success" not in col
        ):
            # for x in df[col]:
            #     print(x)
            df[col] = df[col].apply(lambda x: tuple(float(num) for num in x.strip("()").split(",")))
            df[col] = df[col].apply(lambda x: x[0])
        else:
            plot_straight_line = False

    if "nsamples_all" in y_column:
        df["nsamples_all"] = df["_res_diversity_foldseek_all_samples"].apply(
            lambda x: tuple(float(num) for num in x.strip("()").split(","))
        )
        df["nsamples_all"] = df["nsamples_all"].apply(lambda x: x[2])

    if "_res_diversity_foldseek_des_cluster" in y_column:
        df["_res_diversity_foldseek_des_cluster"] = df["_res_diversity_foldseek_filtered_samples_des"].apply(
            lambda x: tuple(float(num) for num in x.strip("()").split(","))
        )
        df["_res_diversity_foldseek_des_cluster"] = df["_res_diversity_foldseek_des_cluster"].apply(lambda x: x[1])

    if "_res_diversity_foldseek_binder_filtered_samples_binder_success_cluster" in y_column:
        df["_res_diversity_foldseek_binder_filtered_samples_binder_success_cluster"] = df[
            "_res_diversity_foldseek_binder_filtered_samples_binder_success"
        ].apply(lambda x: tuple(float(num) for num in x.strip("()").split(",")) if pd.notna(x) else (0, 0, 0))
        df["_res_diversity_foldseek_binder_filtered_samples_binder_success_cluster"] = df[
            "_res_diversity_foldseek_binder_filtered_samples_binder_success_cluster"
        ].apply(lambda x: x[1])

    if "_res_diversity_foldseek_binder_filtered_samples_binder_success_self_cluster" in y_column:
        df["_res_diversity_foldseek_binder_filtered_samples_binder_success_self_cluster"] = df[
            "_res_diversity_foldseek_binder_filtered_samples_binder_success_self"
        ].apply(lambda x: tuple(float(num) for num in x.strip("()").split(",")) if pd.notna(x) else (0, 0, 0))
        df["_res_diversity_foldseek_binder_filtered_samples_binder_success_self_cluster"] = df[
            "_res_diversity_foldseek_binder_filtered_samples_binder_success_self_cluster"
        ].apply(lambda x: x[1])

    if "_res_diversity_foldseek_binder_filtered_samples_binder_success_self_total_success" in y_column:
        df["_res_diversity_foldseek_binder_filtered_samples_binder_success_self_total_success"] = df[
            "_res_diversity_foldseek_binder_filtered_samples_binder_success_self"
        ].apply(lambda x: tuple(float(num) for num in x.strip("()").split(",")) if pd.notna(x) else (0, 0, 0))
        df["_res_diversity_foldseek_binder_filtered_samples_binder_success_self_total_success"] = df[
            "_res_diversity_foldseek_binder_filtered_samples_binder_success_self_total_success"
        ].apply(lambda x: x[2])

    if "_res_diversity_foldseek_binder_filtered_samples_binder_success_mpnn_cluster" in y_column:
        df["_res_diversity_foldseek_binder_filtered_samples_binder_success_mpnn_cluster"] = df[
            "_res_diversity_foldseek_binder_filtered_samples_binder_success_mpnn"
        ].apply(lambda x: tuple(float(num) for num in x.strip("()").split(",")) if pd.notna(x) else (0, 0, 0))
        df["_res_diversity_foldseek_binder_filtered_samples_binder_success_mpnn_cluster"] = df[
            "_res_diversity_foldseek_binder_filtered_samples_binder_success_mpnn_cluster"
        ].apply(lambda x: x[1])

    if "_res_diversity_foldseek_binder_filtered_samples_binder_success_mpnn_fixed_cluster" in y_column:
        df["_res_diversity_foldseek_binder_filtered_samples_binder_success_mpnn_fixed_cluster"] = df[
            "_res_diversity_foldseek_binder_filtered_samples_binder_success_mpnn_fixed"
        ].apply(lambda x: tuple(float(num) for num in x.strip("()").split(",")) if pd.notna(x) else (0, 0, 0))
        df["_res_diversity_foldseek_binder_filtered_samples_binder_success_mpnn_fixed_cluster"] = df[
            "_res_diversity_foldseek_binder_filtered_samples_binder_success_mpnn_fixed_cluster"
        ].apply(lambda x: x[1])

    count = 0
    for name, group in df_grouped:
        # Sort group by ckpt_no to ensure proper ordering
        group_sorted = group.sort_values(
            by=[
                "ckpt_no",
                "generation_model_bb_ca_simulation_step_params_sc_scale_noise",
                "_res_generation_time_hours",
            ]
        )

        if isinstance(name, tuple):  # If the group name is a tuple (multiple grouping columns)
            label = " (" + ") (".join(map(str, name)) + ")"
        else:  # If there's only one grouping column
            label = f"({name})"

        # Create x and y values for this group, only including valid data points
        plot_x_values = []
        plot_y_values = []

        if x_column == "ckpt_steps (k)":
            # Only collect points that have valid data (no NaN values)
            for ckpt_no in all_ckpt_nos:
                # Find data for this checkpoint in the current group
                group_data = group_sorted[group_sorted["ckpt_no"] == ckpt_no]
                if len(group_data) > 0:
                    # Use the mean if multiple exist for same checkpoint
                    # plot_x_values.append(str(ckpt_no) + "k")
                    plot_x_values.append(ckpt_no)
                    plot_y_values.append(group_data[y_column].mean())
        else:
            plot_x_values = list(group_sorted[x_column])
            plot_y_values = list(group_sorted[y_column])

        # Plot the line and scatter points (only for valid data points)
        if len(plot_x_values) > 0:
            markersize = np.linspace(10, 50, len(plot_x_values))
            plt.plot(
                plot_x_values,
                plot_y_values,
                linestyle="-",
                label=label,
                color=COLORS[count],
            )
            plt.scatter(
                plot_x_values,
                plot_y_values,
                marker="o",
                s=markersize,
                color=COLORS[count],
            )
        count += 1

    if plot_straight_line:
        plt.plot(
            np.linspace(0.1, 0.9, 10),
            np.linspace(0.1, 0.9, 10),
            color="black",
            alpha=0.4,
        )

    if log_scale_y:
        plt.yscale("log")

    # Setting the labels and title
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    # plt.title(f"{y_column} vs {x_column} grouped by {', '.join(group_columns)}")
    title = f"(Bigger marker = higher temp) Grouped by {', '.join(group_columns)}"
    if target_name:
        title = f"Target: {target_name} - " + title
    plt.title(title)

    # Enhancing the legend
    plt.legend(title="Groups", loc="best", fancybox=True, shadow=True, fontsize=6)
    plt.grid(True)  # Optional: Adds a grid for easier readability
    plt.savefig(f"figures_plotting/{run_name}/{x_column}_{y_column}.png")


filters = []


# Big Runs
run_name = [
    # ("big_run_genie2_16n8g_bs4_200M_tri-inference-2024_09_18_15_44_52", "tri-layers-orig"),  # Original tri layers from danny
    # ("big_run_genie2_16n8g_bs4_200M_tri-inference-gtabl1", "tri-layres-1000"),  # First attempr at a bunch of gt
    # ("big_run_genie2_16n8g_bs4_200M_tri-inference-abl_400s", "tri-layres-400"),  # Similar but 400 steps for promising ones - 1000 steps helps...
    # ("big_run_genie2_16n8g_bs4_200M_tri-inference-1000s-moretemp-2024_09_20_19_58_22", "tri-layers-100-gt-temps"),  # More temperatures - 1000 steps, also for promising ones
    # ("big_run_genie2_16n8g_bs4_200M_tri-inference-1000s-190k-somemissingrequeued", "190k")
    # ("big_run_genie2_16n8g_bs4_200M_tri-inference-1000s-sch-ckpt113k-2024_09_21_06_34_21", "sch-analysis"),
    # ("big_run_genie2_16n8g_bs4_200M_tri-inference-1000s-sch-ckpt113k-2024_09_21_06_34_21", "sch-analysis-thr"),
    # ("big_run_genie2_16n8g_bs4_200M_tri-inference-1000s-ckpt113kn200k-2024_09_21_13_26_04", "113k-200k"),  # 113k steps vs 200k, with and without EMA, with and without SC, two gt
    # ("big_run_genie2_16n8g_bs4_200M_tri-inference-1000s-sch-ckpt200k-21-9-19hs", "200k-sch"),  # 200k ckpt, no self cond, EMA and non-EMA, a subset of schedules from the long list, two gt (tan 1 none, tan 0.5 10)
    # subset is (unif, 1), (cos_sch_v_snr, 2), (edm 3), (log, 1.5)
    # ("big_run_genie2_16n8g_bs4_200M_tri-inference-1000s-schcos-ckpt200k-2024_09_21_23_47_35", "cossch-explore-200k"),  # 200k ckpt, no EMA, uniform and some cos sch around 2.0 (gt tan 1 none, tan 0.5 10)
    # ("big_run_genie2_200M_notri_tg_bis-inference-1000s-schcos-200Mnotri-ckpt360k-2024_09_22_11_07_43", "tgrun-360k-wnov-cos"),  # (gt tan 1 none, tan 0.5 10), pdb novelty in there
    # ("big_run_afdb_tri_128gpus_400m_bs4-inference-1000s-schcos-400Mtri-ckpt240k-2024_09_22_11_19_53", "zz240k-cos"),  # (gt tan 1 none, tan 0.5 10)
    # (
    #     "big_run_genie2_16n8g_bs4_200M_tri-inference-1000s-dr-cossch-tlimlowtemp0p99-2024_09_22_18_11_24",
    #     "cossch-explore-dr-240k",
    # ),  # 200k ckpt, no EMA, uniform and some cos sch around 2.0 (gt tan 1 none, tan 0.5 10)
    # ("big_run_genie2_16n8g_bs4_200M_tri-inference-1000s-dr-cossch-tlimlowtemp0p999-2024_09_22_18_53_01", "cossch-explore-dr-240k-999"),  # 200k ckpt, no EMA, uniform and some cos sch around 2.0 (gt tan 1 none, tan 0.5 10)
    # Above is broken very bad
    # ("test_refactor_tri-inference-test-zz-tri-uncond-des-2024_11_09_17_11_26", "w_tri"),
    # ("test_refactor_tg_1-inference-test-tg1-uncond-des-2024_11_09_07_28_13", "wo_tri"),
    # (
    #     "test_refactor_tg_2_noseq_idx-inference-test-tg2-noseq-idx-uncond-des-2024_11_08_23_51_04",
    #     "wo_tri_wo_seq_idx",
    # ),
    # (
    #     "LL_160m_zd_8_kl_5_tr_tr_130m_relio_440k_no_resindex-inference-2025_03_10_22_35_20_proteina_eval",
    #     "LL_440k_noresidx_des",
    # ),
    # ("ca_pdb_monomer_10m_fix-inference-2025_07_16_06_19_35", "ca_pdb_monomer"),
    # ("ca_pdb_monomer_10m_fix-inference-ckpt-sweep-2025_07_16_09_34_05", "ca_pdb_monomer_ckpt_sweep"),
    # ("ca_pdb_monomer_10m_fix-inference-novelty-2025_07_17_15_56_19", "ca_pdb_monomer_novelty"),
    # ("big_aa_ft_pdb_binder_160M_notri_synthetic_loop50_i10_lora-inference-binder-2025_08_13_17_54_17", "synthetic_lora"),
    # ("big_aa_ft_pdb_binder_160M_notri_synthetic_loop50_i10_v3-inference-binder-2025_08_12_23_15_44", "synthetic"),
    # ("big_aa_ft_pdb_binder_160M_notri_synthetic_loop50_i10_v3-inference-binder-20k40k-steps-2025_08_13_17_49_50", "synthetic"),
    # ("big_aa_ft_pdb_binder_160M_notri_filter", "pdb"),
    # ("big_aa_ft_pdb_binder_160M_notri_synthetic_cluster_try1-inference-binder-2025_08_18_20_27_43", "laproteina_notri_synthetic_cluster"),
    # ("big_aa_ft_pdb_binder_160M_notri_synthetic_loop50_b10_seq25_crop384_run1-inference-binder-2025_08_18_17_38_51", "laproteina_notri_synthetic_newcrop384"),
    # ("big_aa_ft_pdb_binder_160M_notri_synthetic_loop50_b10_seq35_crop500_run2-inference-binder-2025_08_18_17_39_52", "laproteina_notri_synthetic_newcrop500"),
    # ("ca_binder_ft_syndata_nocrop_160M_tri_bs1-inference-binder-ca-2025_08_18_15_27_01", "ca_tri_synthetic_nocrop"),
    # ("big_aa_ft_comb_ted_synthetic_160M_notri_64gpu-inference-binder-2025_08_18_16_24_38", "laproteina_notri_ted_synthetic"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-inference-binder-390k-2025_08_25_03_57_36", "ted_lenient_reps"),
    # ("big_aa_ft_comb_lenient_ted_synthetic_160M_notri_96gpu_try2-inference-binder-330k-2025_08_25_15_51_02", "ted_lenient_reps_comb_syn"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-inference-binder-scond-2025_08_22_21_31_33", "ted_lenient_filter"),
    # ("big_aa_ft_comb_strict_ted_synthetic_160M_notri_96gpu-inference-binder-2025_08_21_15_12_12", "ted_strict_reps_comb_syn"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-inference-binder-temp-2025_08_21_18_53_52", "ted_lenient_filter"),
    # ("big_aa_ft_comb_strict_ted_synthetic_160M_notri_96gpu-inference-binder-180k-2025_08_25_02_37_20", "ted_strict_reps_comb_syn"),
    # ("big_aa_ft_comb_strict_ted_synthetic_160M_notri_96gpu-inference-binder-300k-2025_08_25_04_14_26", "ted_strict_reps_comb_syn"),
    # ("big_aa_ft_comb_lenient_all_ted_synthetic_160M_notri_96gpu_run1-inference-binder-270k-2025_08_25_05_39_27", "ted_lenient_all_comb_syn"),
    # ("big_aa_ted_then_lora_ft_pdb_filter_binder_160M_notri", "ted_lenient_reps_then_lora_ft_pdb"),
    # ("big_aa_ted_then_ft_pdb_filter_binder_160M_notri", "ted_lenient_reps_then_ft_pdb"),
    # ("big_aa_ft_comb_strict_ted_binder_160M_notri", "ted_strict_reps_comb_pdb")
    # ("big_aa_ft_comb_lenient_ted_synthetic_160M_notri_96gpu_try2-inference-binder-new-2025_08_26_22_21_45", "ted_lenient_reps_comb_syn"),
    # ("big_aa_ft_comb_strict_ted_synthetic_160M_notri_96gpu-inference-binder-new-2025_08_26_22_22_06", "ted_strict_reps_comb_syn"),
    # ("big_aa_ft_comb_lenient_all_ted_synthetic_160M_notri_96gpu_run1-inference-binder-new-2025_08_26_22_21_17", "ted_lenient_all_comb_syn"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-2025_08_27_03_50_53", "ted_lenient_reps"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-strict-2025_08_28_15_03_52", "ted_lenient_reps"),
    # ("big_aa_ft_strict_ted_binder_160M_notri", "ted_strict_reps"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-target-explore-2025_09_01_19_50_37", "ted_lenient_reps_target_explore"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-all-target-2025_09_02_04_58_44", "best-of-n"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-all-target-8replicas-2025_09_02_04_59_10", "best-of-n"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-pdl1-2025_09_02_08_10_46", "test"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-repack-target-2025_09_02_23_00_15", "test"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-beam-search-2025_09_03_16_07_23", "a"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-best-of-n-time-2025_09_04_01_50_57", "best-of-n"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-beam-search-time-2025_09_04_04_13_31", "beam-search"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-fk-steering-time-2025_09_04_05_13_39", "fk-steering"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-best-of-n-hard-time-2025_09_04_16_20_28", "best-of-n"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-beam-search-hard-time-2025_09_04_16_01_58", "beam-search"),
    # ("colabfold", "colabfold"),
    # ("ptx", "ptx"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-best-of-n-multi-chain-2025_09_05_16_10_13", "best-of-n-multi-chain"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-beam-search-threshold-filter-2025_09_07_03_36_52", "beam-search-threshold-filter"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-beam-search-keep-intermediate-samples-2025_09_07_02_56_13", "beam-search-w-middle"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-spcas9-beam-search-nbranch4-2025_09_07_17_45_12", "beam-search"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-spcas9-beam-search-2025_09_07_22_35_48", "beam-search"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-spcas9-best-of-n-2025_09_07_17_20_03", "best-of-n"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-spcas9-fk-steering-nbranch4-2025_09_08_03_23_06", "fk-steering"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-spcas9-fk-steering-nbranch4-bw3-2025_09_08_05_07_57", "fk-steering"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-spcas9-beam-search-init-sample-2025_09_10_17_05_29", "beam-search"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-spcas9-beam-search-init-nbranch4-2025_09_10_17_05_03", "beam-search"),
    # ("big_aa_ft_comb_lenient_all_ted_synthetic_160M_notri_96gpu_run1-inference-binder-unified-2025_09_09_00_57_33", "ted_lenient_all"),
    # ("big_aa_ft_ted_160M_notri_96gpu_strict_filter-inference-binder-unified-2025_09_08_23_04_01", "ted_strict"),
    # ("big_aa_ft_pdb_binder_160M_notri_pdb_filter_try1-inference-binder-unified-2025_09_08_23_15_11", "ted_lenient_ft_pdb"),
    # ("big_aa_ft_pdb_binder_160M_notri_pdb_filter_lora_try1-inference-binder-unified-2025_09_09_01_00_49", "ted_lenient_ft_pdb_lora"),
    # ("big_aa_ft_pdb_binder_160M_notri_pdb_filter_lora_try1-inference-binder-unified-all-ckpts-2025_09_09_17_02_43", "ted_lenient_ft_pdb_lora"),
    # ("big_aa_ft_pdb_binder_160M_notri_pdb_filter_lora_try1-inference-binder-unified-all-ckpts-early-2025_09_10_00_56_46", "ted_lenient_ft_pdb_lora"),
    # ("tg_test_tri_attn_ted_lenient-inference-binder-2025_09_10_21_22_43", "ted_lenient_tri_attn"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-spcas9-mcts-2025_09_13_03_27_48", "mcts"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-spcas9-mcts-checkpoints400-2025_09_13_15_20_46", "mcts"),
    # Best setup
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-spcas9-best-of-n-2025_09_07_17_20_03", "best-of-n"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-spcas9-best-of-n-init-sample-2025_09_14_21_32_36", "best-of-n"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-spcas9-fk-steering-init-sample-2025_09_14_15_04_03", "fk-steering"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-spcas9-beam-search-short-checkpoint-2025_09_14_18_45_27", "beam-search"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-spcas9-mcts-init-sample-2025_09_14_02_39_12", "mcts"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-spcas9-mcts-init-sample-shorter-2025_09_14_01_36_29", "mcts"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-spcas9-mcts-init-sample-more-2025_09_14_06_52_13", "mcts"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-spcas9-mcts-exploration-constant-init-sample-2025_09_15_15_57_46", "mcts"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-spcas9-mcts-exploration-constant-2025_09_15_03_23_22", "mcts"),
    # ("new_bfmd_results", "new_bfmd"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-inference-binder-unified-all-targets-ptx-2025_09_11_06_31_55", "ted_lenient"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-spcas9-beam-search-tmol-2025_09_15_15_58_10", "beam-search"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-spcas9-beam-search-tmol-only-2025_09_15_15_58_42", "beam-search"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-spcas9-beam-search-tmol-more-weight-2025_09_15_18_06_27", "beam-search"),
    # base run
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-inference-binder-unified-ted-lenient-filter-2025_09_08_21_40_46", "ted_lenient"),
    (
        "big_aa_ft_ted_160M_notri_96gpu_lenient_filter-inference-binder-unified-all-ckpts-2025_09_09_20_35_10",
        "ted_lenient",
    ),
    # ("combined_binder_datasets_pdb1_ted3_bfmd2_96gpu-inference-binder-unified-all-ckpts-2025_09_16_07_09_32", "pdb_ted_lenient_bfmd"),
    # ("combined_two_binder_datasets_ted4_bfmd2_96gpu-inference-binder-unified-all-ckpts-2025_09_16_16_04_50", "ted_lenient_bfmd"),
    (
        "big_aa_ft_ted_160M_notri_96gpu_comb_extra_lenient_pdb_try1-inference-binder-unified-all-ckpts-2025_09_16_04_31_44",
        "ted_extra_lenient_pdb",
    ),
    (
        "big_aa_ft_ted_160M_notri_96gpu_comb_extra_lenient_pdb_try1-inference-binder-unified-380k-2025_09_18_16_35_43",
        "ted_extra_lenient_pdb",
    ),
    (
        "big_aa_ft_ted_160M_notri_96gpu_extra_lenient_cat_try2-inference-binder-unified-all-ckpts-2025_09_16_06_20_38",
        "ted_extra_lenient_cat",
    ),
    (
        "big_aa_ft_ted_160M_notri_96gpu_extra_lenient_cat_try2-inference-binder-unified-380k-2025_09_18_16_36_35",
        "ted_extra_lenient_cat",
    ),
    (
        "lora_rank4_ft_binder_160M_notri_pdb_filter_after_TEDcat_try1-inference-binder-unified-2025_09_18_06_17_52",
        "pdb_lora4",
    ),
    (
        "lora_rank8_ft_binder_160M_notri_pdb_filter_after_TEDcat_try1-inference-binder-unified-2025_09_18_06_17_17",
        "pdb_lora8",
    ),
    (
        "lora_rank16_ft_binder_160M_notri_pdb_filter_after_TEDcat_try1-inference-binder-unified-2025_09_18_06_16_36",
        "pdb_lora16",
    ),
    (
        "lora_rank32_ft_binder_160M_notri_pdb_filter_after_TEDcat_try1-inference-binder-unified-2025_09_18_06_16_03",
        "pdb_lora32",
    ),
    # new search runs
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-best-of-n-four-targets-2025_09_16_15_47_37", "best-of-n"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-best-of-n-four-targets-more-2025_09_16_20_06_41", "best-of-n"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-beam-search-four-targets-2025_09_16_16_13_57", "beam-search"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-beam-search-four-targets-400samples-2025_09_16_16_15_01", "beam-search"),
    # tmol search
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-beam-search-tmol-2025_09_17_04_31_09", "beam-search-tmol"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-beam-search-tmol-less-2025_09_17_17_10_13", "beam-search-tmol"),
    # refinement search
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-bc-refinement-2025_09_17_17_10_52", "bindcraft-refinement"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-bc-refinement-no-soft-2025_09_17_17_11_49", "bindcraft-refinement"),
    # ("big_aa_ft_ted_160M_notri_96gpu_lenient_filter-search-binder-bc-refinement-no-soft-no-greedy-2025_09_17_17_11_22", "bindcraft-refinement"),
]
group_by = [
    # "generation_model_bb_ca_schedule_mode",
    # "generation_model_bb_ca_schedule_p",
    # "generation_model_bb_ca_gt_mode",
    # "generation_model_bb_ca_gt_p",
    # "generation_model_bb_ca_gt_clamp_val",
    # "run_name",
    # "ckpt_name",
    # "generation_model_bb_ca_simulation_step_params_sc_scale_noise",
    # "generation_model_local_latents_simulation_step_params_sc_scale_noise",
    # "generation_dataset_target_task_name"
    "run_name",
    "generation_dataset_target_task_name",
    # "generation_search_beam_search_step_checkpoints",
    # "generation_search_beam_search_n_branch",
    # "generation_search_fk_steering_n_branch",
    # "generation_search_beam_search_beam_width",
    # "generation_search_fk_steering_step_checkpoints",
    # "generation_search_fk_steering_beam_width",
    # "generation_args_self_cond",
    # "generation_search_mcts_step_checkpoints",
    # "generation_search_mcts_exploration_prob",
    # "generation_search_mcts_exploration_constant",
    # "generation_search_mcts_n_simulations",
    # "generation_search_tmol_reward_weight",
    # "generation_search_folding_reward_weight",
    # "generation_refinement_enable_soft_optimization",
    # "generation_refinement_enable_greedy_optimization",
    # "generation_refinement_n_temp_iters",
    # "generation_refinement_n_recycles",
    # "generation_refinement_greedy_percentage",
]
filters = [
    # ("generation_search_beam_search_step_checkpoints", ["[0, 200, 225, 250, 275, 300, 325, 350, 375, 400]"], None),
    # ("generation_dataset_target_task_name", [
    #     "32_PDL1_ALPHA",
    #     "01_PD1",
    #     "34_Insulin",
    #     "29_BHRF1",
    #     "30_SC2RBD",
    #     "25_CbAgo",
    #     "26_CbAgo",
    #     "04_IFNAR2",
    # ], None),  # Now handled in target loop
    # ("generation_dataset_target_task_name", [
    #     "13_BBF14",
    #     "36_VEGFA", # 2 chains
    #     "31_IL7RA",
    #     "15_DerF7",
    #     "18_DerF21",
    #     "35_H1", # 2 chains
    #     "05_CD45",
    # ], None),     # Now handled in target loop
    # ("generation_dataset_target_task_name", [
    #     "37_IL17A", # 2 chains
    #     "38_TNFalpha", # 3 chains
    #     "33_TrkA",
    #     "12_Claudin1",
    #     "14_CrSAS6",
    #     "24_SpCas9",
    #     "23_BetV1",
    #     "28_HER2_AAV"
    # ], None),       # Now handled in target loop
    # ("generation_dataset_target_task_name", [
    #     "35_H1",
    #     "37_IL17A", # 2 chains
    #     # "38_TNFalpha",
    #     # "38_TNFalpha_REPACK", # 3 chains
    # ], None),       # Now handled in target loop
    # ("generation_dataset_target_task_name", ["33_TrkA"], None),        # Now handled in target loop
    # ("generation_search_beam_search_beam_width", None, [2, 1]),
    # ("generation_search_fk_steering_beam_width", None, [2, 1]),
    # ("generation_search_beam_search_step_checkpoints", None, ["[0, 200, 250, 300, 350, 400]"]),
    # ("generation_search_fk_steering_step_checkpoints", None, ["[0, 200, 225, 250, 275, 300, 325, 350, 375, 400]"]),
    # ("generation_search_mcts_step_checkpoints", None, ["[0, 50, 100, 150, 200, 250, 300, 350, 400]"]),
    # ("generation_search_mcts_n_simulations", None, [40, 20]),
    # ("generation_search_mcts_exploration_constant", None, [0.3, 0.5, 0.7]),
    # ("generation_search_mcts_exploration_prob", None, [0.3]),
    # ("ckpt_name", None, ["chk_epoch=00000009_step=000000040000-EMA.ckpt","chk_epoch=00000012_step=000000050000-EMA.ckpt", "chk_epoch=00000014_step=000000060000-EMA.ckpt"]),
    # ("schedule_schedule_mode", ["power", "uniform"], None),
    # ("schedule_schedule_p", [0.33], None),
    # ("dt", [0.01], None),
    # ("sampling_caflow_gt_mode", ["tan"], None),
    # ("sampling_caflow_gt_clamp_val", None, [1000]),
    # ("sampling_caflow_gt_clamp_val", None, [10]),
    # ("self_cond", [False], None),
    # ("sampling_caflow_gt_p", [0.5], None),
    # ("schedule_schedule_mode", ["cos_sch_v_snr", "uniform", "edm"], None),
    # ("schedule_schedule_p", [1.5, 2, 7, 2, -1, 1], None),
    # ("ckpt_name", ["chk_epoch=00000113_step=000000130000.ckpt", "chk_epoch=00000174_step=000000200000.ckpt"], None),
    # ("ckpt_name", ["chk_epoch=00000174_step=000000200000-EMA.ckpt", "chk_epoch=00000174_step=000000200000.ckpt"], None),
    # ("ckpt_name", ["chk_epoch=00000113_step=000000130000-EMA.ckpt", "chk_epoch=00000174_step=000000200000-EMA.ckpt"], None),
    # ("ckpt_name", ["chk_epoch=00000174_step=000000200000.ckpt"], None),
]


if type(run_name) == list:
    ident = "_".join([v[1] for v in run_name])
    all_dfs = []
    all_cols = set()
    for rn, rid in run_name:
        df = pd.read_csv(f"results_downloaded/{rn}/a_results_processed/res_all.csv")
        df["run_name"] = rid

        # Remove rows with designability 0
        if "des_2A_ca_esmfold" in df.columns:
            df = df[df["des_2A_ca_esmfold"] != 0]

        # Remove rows with very low designability as well
        # df = df[df["des_2A"] > 0.5]

        all_dfs.append(df)
        all_cols.update(df.columns)

    for i in range(len(all_dfs)):
        missing_cols = set(all_cols) - set(all_dfs[i].columns)
        for col in missing_cols:
            all_dfs[i][col] = -1

    df = pd.concat(all_dfs)
    os.makedirs(f"figures_plotting/{ident}", exist_ok=True)
    run_name = ident

else:
    df = pd.read_csv(f"results_downloaded/{run_name}/a_results_processed/res_all.csv")
    os.makedirs(f"figures_plotting/{run_name}", exist_ok=True)

# df = filter_data(
#     df, "generation_model_bb_ca_simulation_step_params_sampling_mode", tuple_keep=["sc"], tuple_remove=None
# )

# =============================================================================
# TARGET FILTERING CONFIGURATION
# =============================================================================
#
# Option 1: Filter by specific targets (set filter_by_target = True)
# - Define targets_to_plot list with target names
# - Script will create separate plots for each target
#
# Option 2: Use all data without target filtering (set filter_by_target = False)
# - Set targets_to_plot to [] or None
# - Script will use all available data
#
# =============================================================================

# All potential things to plot in each axis
# _res_PDB_FID,_res_PDB_fJSD_C,_res_PDB_fJSD_A,_res_PDB_fJSD_T,_res_AFDB_FID,_res_AFDB_fJSD_C,_res_AFDB_fJSD_A,_res_AFDB_fJSD_T,_res_IS_C,_res_IS_A,_res_IS_T
# _res_ss_biot_alpha_all_samples,_res_ss_biot_beta_all_samples,_res_ss_biot_coil_all_samples,_res_ss_biot_alpha_des_samples,_res_ss_biot_beta_des_samples,
# _res_ss_biot_coil_des_samples,_res_diversity_foldseek_all_samples,_res_diversity_foldseek_des_samples,_res_diversity_pairwise_tm_by_len_all_samples
# _res_diversity_pairwise_tm_by_len_des_samples,_res_novelty_pdb_tm_all_samples,_res_novelty_pdb_tm_des_samples,des_2A

axes = [
    # ("generation_model_bb_ca_simulation_step_params_sc_scale_noise", "des_2A_ca_esmfold"),
    # ("generation_model_bb_ca_simulation_step_params_sc_scale_noise", "_res_PDB_FID"),
    # ("generation_model_bb_ca_simulation_step_params_sc_scale_noise", "_res_AFDB_FID"),
    # (
    #     "generation_model_bb_ca_simulation_step_params_sc_scale_noise",
    #     "_res_novelty_pdb_tm_des_samples",
    # ),
    # ("_res_PDB_FID", "_res_AFDB_FID"),
    # (
    #     "generation_model_bb_ca_simulation_step_params_sc_scale_noise",
    #     "_res_diversity_foldseek_filtered_samples_des",
    # ),
    # (
    #     "generation_model_bb_ca_simulation_step_params_sc_scale_noise",
    #     "_res_diversity_foldseek_des_cluster",
    # ),
    # (
    #     "generation_model_bb_ca_simulation_step_params_sc_scale_noise",
    #     "des_2A_ca_esmfold",
    # ),
    # (
    #     "generation_model_bb_ca_simulation_step_params_sc_scale_noise",
    #     "_res_ss_biot_coil_filtered_samples_des",
    # ),
    # (
    #     "generation_model_bb_ca_simulation_step_params_sc_scale_noise",
    #     "_res_ss_biot_beta_filtered_samples_des",
    # ),
    # (
    #     "generation_model_bb_ca_simulation_step_params_sc_scale_noise",
    #     "_res_ss_biot_alpha_filtered_samples_des",
    # ),
    # ("ckpt_name", "_res_diversity_foldseek_binder_filtered_samples_binder_success_cluster"),
    # ("ckpt_name", "_res_mpnn_best_of_n_success_all_samples"),
    # ("ckpt_name", "_res_mpnn_fixed_best_of_n_success_all_samples"),
    # ("ckpt_name", "_res_self_best_of_n_success_all_samples"),
    (
        "ckpt_name",
        "_res_diversity_foldseek_binder_filtered_samples_binder_success_self_cluster",
    ),
    (
        "ckpt_name",
        "_res_diversity_foldseek_binder_filtered_samples_binder_success_mpnn_cluster",
    ),
    (
        "ckpt_name",
        "_res_diversity_foldseek_binder_filtered_samples_binder_success_mpnn_fixed_cluster",
    ),
    # ("ckpt_name", "_res_mpnn_best_of_n_success_rfdiffusion_all_samples"),
    # ("ckpt_name", "_res_mpnn_fixed_best_of_n_success_rfdiffusion_all_samples"),
    # ("ckpt_name", "_res_self_best_of_n_success_rfdiffusion_all_samples"),
    ("ckpt_name", "_res_mpnn_best_of_n_success_alphaproteo_all_samples"),
    ("ckpt_name", "_res_mpnn_fixed_best_of_n_success_alphaproteo_all_samples"),
    ("ckpt_name", "_res_self_best_of_n_success_alphaproteo_all_samples"),
    # ("ckpt_name", "_res_total_interface_hbond_energy_tmol_filtered_samples_binder_success_self"),
    # ("generation_search_best_of_n_replicas", "_res_diversity_foldseek_binder_filtered_samples_binder_success_self_cluster"),
    # ("generation_search_best_of_n_replicas", "_res_diversity_foldseek_binder_filtered_samples_binder_success_mpnn_cluster"),
    # ("generation_search_best_of_n_replicas", "_res_diversity_foldseek_binder_filtered_samples_binder_success_mpnn_fixed_cluster"),
    # ("generation_search_best_of_n_replicas", "_res_self_best_of_n_success_alphaproteo_all_samples"),
    # ("generation_search_best_of_n_replicas", "_res_mpnn_best_of_n_success_alphaproteo_all_samples"),
    # ("generation_search_best_of_n_replicas", "_res_mpnn_fixed_best_of_n_success_alphaproteo_all_samples"),
    # ("generation_search_best_of_n_replicas", "_res_mpnn_best_of_n_success_rfdiffusion_all_samples"),
    # ("generation_search_best_of_n_replicas", "_res_mpnn_fixed_best_of_n_success_rfdiffusion_all_samples"),
    # ("generation_search_best_of_n_replicas", "_res_self_best_of_n_success_rfdiffusion_all_samples"),
    # ("generation_search_best_of_n_replicas", "_res_mpnn_best_of_n_success_min_ipae_all_samples"),
    # ("generation_search_best_of_n_replicas", "_res_mpnn_fixed_best_of_n_success_min_ipae_all_samples"),
    # ("generation_search_best_of_n_replicas", "_res_self_best_of_n_success_min_ipae_all_samples"),
    # ("generation_search_best_of_n_replicas", "_res_mpnn_best_of_n_success_min_ipsae_all_samples"),
    # ("generation_search_best_of_n_replicas", "_res_mpnn_fixed_best_of_n_success_min_ipsae_all_samples"),
    # ("generation_search_best_of_n_replicas", "_res_self_best_of_n_success_min_ipsae_all_samples"),
    # ("generation_search_beam_search_beam_width", "_res_diversity_foldseek_binder_filtered_samples_binder_success_self_cluster"),
    # ("generation_search_beam_search_beam_width", "_res_diversity_foldseek_binder_filtered_samples_binder_success_mpnn_cluster"),
    # ("generation_search_beam_search_beam_width", "_res_diversity_foldseek_binder_filtered_samples_binder_success_mpnn_fixed_cluster"),
    # ("generation_search_beam_search_beam_width", "_res_self_best_of_n_success_alphaproteo_all_samples"),
    # ("generation_search_beam_search_beam_width", "_res_mpnn_best_of_n_success_alphaproteo_all_samples"),
    # ("generation_search_beam_search_beam_width", "_res_mpnn_fixed_best_of_n_success_alphaproteo_all_samples"),
    # ("generation_search_beam_search_beam_width", "_res_self_best_of_n_success_min_ipae_all_samples"),
    # ("generation_search_beam_search_beam_width", "_res_mpnn_best_of_n_success_min_ipae_all_samples"),
    # ("generation_search_beam_search_beam_width", "_res_mpnn_fixed_best_of_n_success_min_ipae_all_samples"),
    # ("_res_total_time_hours", "_res_diversity_foldseek_binder_filtered_samples_binder_success_self_total_success"),
    # ("_res_total_time_hours", "_res_diversity_foldseek_binder_filtered_samples_binder_success_self_cluster"),
    # ("_res_generation_time_seconds", "_res_diversity_foldseek_binder_filtered_samples_binder_success_self_cluster"),
    # ("_res_total_time_seconds", "_res_diversity_foldseek_binder_filtered_samples_binder_success_self_total_success"),
    # ("_res_total_time_seconds", "_res_diversity_foldseek_binder_filtered_samples_binder_success_self_cluster"),
    # ("_res_total_time_seconds", "_res_diversity_foldseek_binder_filtered_samples_binder_success_mpnn_cluster"),
    # ("_res_total_time_seconds", "_res_diversity_foldseek_binder_filtered_samples_binder_success_mpnn_fixed_cluster"),
    # ("_res_total_time_hours", "_res_self_best_of_n_success_alphaproteo_all_samples"),
    # ("_res_total_time_seconds", "_res_mpnn_best_of_n_success_alphaproteo_all_samples"),
    # ("_res_total_time_seconds", "_res_mpnn_fixed_best_of_n_success_alphaproteo_all_samples"),
    #### search metrics #####################################
    # ("_res_generation_time_hours", "_res_diversity_foldseek_binder_filtered_samples_binder_success_self_total_success"),
    # ("_res_generation_time_hours", "_res_diversity_foldseek_binder_filtered_samples_binder_success_self_cluster"),
    # ("_res_generation_time_hours", "_res_self_best_of_n_success_alphaproteo_all_samples"),
    # ("_res_generation_time_hours", "_res_refolded_self_n_interface_hbonds_tmol_filtered_samples_binder_success_self"),
    # ("_res_generation_time_hours", "_res_refolded_self_total_interface_hbond_energy_tmol_filtered_samples_binder_success_self"),
    #########################################################
    # ("_res_total_time_seconds", "_res_diversity_foldseek_binder_filtered_samples_binder_success_self_cluster"),
    # ("_res_total_samples", "_res_diversity_foldseek_binder_filtered_samples_binder_success_mpnn_cluster"),
    # ("_res_total_samples", "_res_diversity_foldseek_binder_filtered_samples_binder_success_mpnn_fixed_cluster"),
    # ("_res_total_samples", "_res_self_best_of_n_success_alphaproteo_all_samples"),
    # ("_res_total_samples", "_res_mpnn_best_of_n_success_alphaproteo_all_samples"),
    # ("_res_total_samples", "_res_mpnn_fixed_best_of_n_success_alphaproteo_all_samples"),
    # ("_res_total_time_seconds", "_res_refolded_self_n_interface_hbonds_tmol_filtered_samples_binder_success_self"),
    # ("_res_total_time_seconds", "_res_refolded_self_total_interface_hbond_energy_tmol_filtered_samples_binder_success_self"),
    # ("ckpt_name", "des_2A_ca_esmfold"),
    # ("ckpt_name", "_res_diversity_foldseek_filtered_samples_des"),
    # ("ckpt_name", "_res_diversity_foldseek_des_cluster"),
    # ("ckpt_name", "_res_ss_biot_coil_filtered_samples_des"),
    # ("ckpt_name", "_res_ss_biot_beta_filtered_samples_des"),
    # ("ckpt_name", "_res_ss_biot_alpha_filtered_samples_des"),
    # ("ckpt_name", "_res_novelty_pdb_tm_all_samples"),
    # (
    #     "ckpt_name",
    #     "des_2A_ca_esmfold",
    # ),
    # ("generation_model_bb_ca_simulation_step_params_sc_scale_noise", "_res_diversity_pairwise_tm_by_len_des_samples"),
    # ("des_2A", "_res_diversity_pairwise_tm_by_len_des_samples"),  #
    # ("des_2A_ca_esmfold", "_res_diversity_foldseek_filtered_samples_des"),
    # ("des_2A_ca_esmfold", "_res_diversity_foldseek_des_cluster"),
    # ("des_2A_ca_esmfold", "_res_ss_biot_coil_filtered_samples_des"),  #
    # ("des_2A_ca_esmfold", "_res_ss_biot_beta_filtered_samples_des"),  #
    # ("des_2A_ca_esmfold", "_res_ss_biot_alpha_filtered_samples_des"),  #
    # ("des_2A", "_res_diversity_maxcluster_des_samples"),
    # ("_res_diversity_foldseek_des_samples", "_res_diversity_maxcluster_des_samples"),
    # ("generation_model_bb_ca_simulation_step_params_sc_scale_noise", "nsamples_all"),  #
    # ("des_2A", "_res_AFDB_FID"),
    # ("des_2A", "_res_novelty_pdb_tm_des_samples"),
    # ("_res_diversity_foldseek_des_samples", "_res_novelty_pdb_tm_des_samples"),
    # ("_res_AFDB_FID", "_res_IS_C"),
    # ("_res_AFDB_FID", "_res_IS_A"),
    # ("_res_AFDB_FID", "_res_IS_T"),
    # ("_res_AFDB_FID", "_res_AFDB_fJSD_C"),
    # ("_res_AFDB_FID", "_res_AFDB_fJSD_A"),
    # ("_res_AFDB_FID", "_res_AFDB_fJSD_T"),
]

# Define the list of targets to plot by difficulty level
# You can modify this list to include any targets you want to analyze
# Set to None or empty list if you don't want to filter by target_task_name

# Target difficulty categorization
targets_by_difficulty = {
    "easy": [
        "32_PDL1_ALPHA_REPACK",
        "01_PD1",
        # "34_Insulin",
        "29_BHRF1",
        # "30_SC2RBD",
        # "25_CbAgo",
        # "26_CbAgo",
        "04_IFNAR2",
    ],
    "medium": [
        "13_BBF14",
        # "36_VEGFA", # 2 chains
        # "31_IL7RA_REPACK",
        "15_DerF7",
        "18_DerF21",
        # "35_H1", # 2 chains
        # "05_CD45",
    ],
    "hard": [
        # "37_IL17A", # 2 chains
        # "38_TNFalpha_REPACK",
        # "33_TrkA",
        # "12_Claudin1",
        # "14_CrSAS6",
        # "24_SpCas9",
        # "23_BetV1",
        # "28_HER2_AAV"
    ],
}

# Flatten all targets for individual plotting
targets_to_plot = []
for difficulty, targets in targets_by_difficulty.items():
    targets_to_plot.extend(targets)

# Set to True if you want to filter by target_task_name, False if you want to use all data
filter_by_target = True

if filter_by_target and targets_to_plot:
    # Loop through each target and create plots
    for target in targets_to_plot:
        print(f"Processing target: {target}")

        # Create a copy of the original dataframe for this target
        df_target = df.copy()

        # Apply filters for this specific target
        df_target = filter_data(
            df_target,
            "generation_dataset_target_task_name",
            tuple_keep=[target],
            tuple_remove=None,
        )
        for f in filters:
            cf, fk, fr = f
            # Apply other filters as usual
            if fk is None:
                df_target = filter_data(df_target, cf, tuple_keep=None, tuple_remove=fr)
            else:
                df_target = filter_data(df_target, cf, tuple_keep=fk, tuple_remove=None)

        # Create target-specific directory
        target_dir = f"figures_plotting/{run_name}/{target}"
        os.makedirs(target_dir, exist_ok=True)

        # Use target name for plot titles
        target_name = target

        # Plot for this target
        for x, y in axes:
            plot_data(
                run_name=f"{run_name}/{target}",  # Use target-specific path
                df_orig=df_target,
                group_columns=group_by,
                x_column=x,
                y_column=y,
                log_scale_y=False,
                target_name=target_name,
            )

    # Create aggregated plots by difficulty level
    print("Creating aggregated plots by difficulty level...")
    for difficulty, targets in targets_by_difficulty.items():
        if not targets:  # Skip if no targets in this difficulty level
            continue

        print(f"Processing difficulty level: {difficulty}")

        # Create a copy of the original dataframe for this difficulty level
        df_difficulty = df.copy()

        # Apply filters for this specific difficulty level
        df_difficulty = filter_data(
            df_difficulty,
            "generation_dataset_target_task_name",
            tuple_keep=targets,
            tuple_remove=None,
        )
        for f in filters:
            cf, fk, fr = f
            # Apply other filters as usual
            if fk is None:
                df_difficulty = filter_data(df_difficulty, cf, tuple_keep=None, tuple_remove=fr)
            else:
                df_difficulty = filter_data(df_difficulty, cf, tuple_keep=fk, tuple_remove=None)

        # Create difficulty-specific directory
        difficulty_dir = f"figures_plotting/{run_name}/difficulty_{difficulty}"
        os.makedirs(difficulty_dir, exist_ok=True)

        # Use difficulty name for plot titles
        difficulty_name = f"{difficulty.capitalize()} Targets (n={len(targets)}) - Mean"

        # Plot for this difficulty level using aggregated data
        for x, y in axes:
            plot_data(
                run_name=f"{run_name}/difficulty_{difficulty}",  # Use difficulty-specific path
                df_orig=df_difficulty,
                group_columns=group_by,
                x_column=x,
                y_column=y,
                log_scale_y=False,
                target_name=difficulty_name,
                aggregate_by_target=True,  # Enable aggregation by target
            )
else:
    # Don't filter by target - use all data
    print("Processing all data without target filtering")

    # Create a copy of the original dataframe
    df_target = df.copy()

    # Apply filters (excluding target_task_name filtering)
    for f in filters:
        cf, fk, fr = f
        # if cf != "generation_dataset_target_task_name":  # Skip target filtering
        if fk is None:
            df_target = filter_data(df_target, cf, tuple_keep=None, tuple_remove=fr)
        else:
            df_target = filter_data(df_target, cf, tuple_keep=fk, tuple_remove=None)

    # Create directory for all data
    target_dir = f"figures_plotting/{run_name}"
    os.makedirs(target_dir, exist_ok=True)

    # Plot for all data
    for x, y in axes:
        plot_data(
            run_name=f"{run_name}",  # Use all-data path
            df_orig=df_target,
            group_columns=group_by,
            x_column=x,
            y_column=y,
            log_scale_y=False,
        )
