import argparse
import os
import sys

import hydra
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from loguru import logger
from sklearn.decomposition import PCA

from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder

COLORS_RT = [
    "#FF0000",  # Red
    "#008000",  # Green
    "#0000FF",  # Blue
    "#FFFF00",  # Yellow
    "#FFA500",  # Orange
    "#800080",  # Purple
    "#00FFFF",  # Cyan
    "#FF00FF",  # Magenta
    "#00FF00",  # Lime
    "#FFC0CB",  # Pink
    "#008080",  # Teal
    "#E6E6FA",  # Lavender
    "#A52A2A",  # Brown
    "#F5F5DC",  # Beige
    "#800000",  # Maroon
    "#808000",  # Olive
    "#FF7F50",  # Coral
    "#000080",  # Navy
    "#AAF0D1",  # Mint
    "#FFDB58",  # Mustard
]


def parse_args_and_cfg() -> tuple[dict, dict, str]:
    """
    Parses command line arguments and loads the corresponding config file.

    Returns:
        Command line arguments (dict)
        Config file (dict)
        config_name (string)
    """
    parser = argparse.ArgumentParser(description="Job info")
    parser.add_argument(
        "--config_name",
        type=str,
        default="inference_ae",
        help="Name of the config yaml file.",
    )
    parser.add_argument("--config_number", type=int, default=-1, help="Number of the config yaml file.")
    parser.add_argument(
        "--config_subdir",
        type=str,
        help="(Optional) Name of directory with config files, if not included uses base inference config.\
            Likely only used when submitting to the cluster with script.",
    )
    args = parser.parse_args()

    # Inference config
    # If config_subdir is None then use base inference config
    # Otherwise use config_subdir/some_config
    if args.config_subdir is None:
        config_path = "../../configs"
    else:
        config_path = f"../../configs/{args.config_subdir}"

    with hydra.initialize(config_path, version_base=hydra.__version__):
        # If number provided use it, otherwise name
        if args.config_number != -1:
            config_name = f"inf_{args.config_number}"
        else:
            config_name = args.config_name
        cfg = hydra.compose(config_name=config_name)
        logger.info(f"Inference config {cfg}")

    return args, cfg, config_name


def extract_ckpt_info(ckpt_file_path):
    ae_name = ckpt_file_path.split("/")[-3]
    ckpt_name = ckpt_file_path.split("/")[-1]
    return ae_name, ckpt_name


def setup(
    cfg: dict,
    config_name: str,
    create_root: bool = True,
) -> str:
    """
    Checks if metrics being computed are compatible, sets the right seed, and creates the root directory
    where the run will store things.

    Returns:
        Path of the root directory (string)
    """
    logger.info(" ".join(sys.argv))

    assert torch.cuda.is_available(), "CUDA not available"  # Needed for ESMfold and designability
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
    )  # Send to stdout

    # Set root path for this inference run
    root_path = f"./inference/{config_name}"
    if create_root:
        os.makedirs(root_path, exist_ok=True)
    else:
        if not os.path.exists(root_path):
            raise ValueError("Results path %s does not exist" % root_path)

    # Set seed
    logger.info(f"Seeding everything to seed {cfg.seed}")
    L.seed_everything(cfg.seed)

    return root_path


def load_dataloader(cfg):
    """
    Loads data config file and returns dataloader.
    """
    if cfg.dataset == "genie2":
        config_path = "../../configs/dataset/afdb_fromraw"
        config_name = "genie2"
    elif cfg.dataset == "pdb":
        config_path = "../../configs/dataset/pdb"
        config_name = "pdb_multimer_chain"
    elif cfg.dataset == "pdb_multimer":
        config_path = "../../configs/dataset/pdb"
        config_name = "pdb_multimer"
    else:
        raise ValueError(f"Dataset {cfg.dataset} not implemented")

    with hydra.initialize(config_path, version_base=hydra.__version__):
        cfg_data = hydra.compose(config_name=config_name)
        cfg_data["datamodule"]["batch_size"] = cfg.bs

    datamodule = hydra.utils.instantiate(cfg_data.datamodule)
    datamodule.prepare_data()
    datamodule.setup("fit")
    dataloader = datamodule.val_dataloader()
    print(
        f"Number of batches in dataloader: {len(dataloader)}, batch size: {cfg.bs}, total number of structures: {len(dataloader) * cfg.bs}"
    )
    return dataloader


def extract_pdb_ids(predictions):
    logger.info("Extracting PDBs we test on")
    vals = []
    for x_in, _ in predictions:
        v = x_in["id"]
        vals += v
    return vals


def compute_all_atom_rmsd(predictions, model):
    logger.info("Computing all-atom RMSD")
    vals = []
    for x_in, x_out in predictions:
        v = model.compute_struct_rec_loss(
            output_dec=x_out,
            batch=x_in,
        )["rmsd_no_align_a37_ang_justlog"]
        vals += v.tolist()
    return vals


def compute_sec_rec_rate(predictions, model):
    logger.info("Computing sequence recovery rate")
    vals = []
    for x_in, x_out in predictions:
        v = model.compute_seq_rec_loss(
            output_dec=x_out,
            batch=x_in,
        )["seq_rec_rate_justlog"]
        vals += v.tolist()
    return vals


def compute_kl_latent(predictions, model):
    logger.info("Computing sequence recovery rate")
    vals = []
    for _, x_out in predictions:
        v = model.compute_kl_penalty(
            mean=x_out["mean"],
            log_scale=x_out["log_scale"],
            mask=x_out["residue_mask"],
            w=1.0,
        )["kl_now_justlog"]
        vals += v.tolist()
    return vals


# Need to fix
# def compute_std_latent_dist(predictions, model):
#     logger.info(f"Computing std latent distribution")
#     vals = []
#     for (_, x_out) in predictions:
#         log_scale=x_out["log_scale"]  # [b, n, d]
#         mask=x_out["residue_mask"]  # [b, n, d]
#         vals_aux = log_scale.clone()[mask]  # [num of unmasked residues, d]
#         vals += torch.flatten(vals_aux).tolist()
#     return vals


def plot_latent_pca_per_residue_type(predictions, root_path, config_name, n_components=4):
    dir_store = os.path.join(root_path, f"pca_per_residue_type_{config_name}")
    os.makedirs(dir_store, exist_ok=False)
    x_in, x_out = predictions[0]  # Only for the first batch

    res_ty = x_in["residue_type"]  # [b, n]
    v = x_out["z_latent"]  # [b, n, d]
    mask = x_out["residue_mask"]  # [b, n]

    for i in range(n_components):
        for j in range(n_components):
            if j <= i:
                continue
            fig, ax = plt.subplots()
            ax.set_xlabel(f"PC {i}")
            ax.set_ylabel(f"PC {j}")
            ax.set_title(f"PC {i} - PC {j}")

            for rt in range(20):
                mask_rt = res_ty == rt  # [b, n]
                mask_rt = mask_rt * mask  # [b, n]
                vals = v.clone()[mask_rt].cpu().detach().float().numpy()  # [num of unmasked residues, d]
                vals_pca = PCA(n_components=n_components).fit_transform(
                    vals
                )  # [num of unmasked residues, n_components]
                ax.scatter(
                    vals_pca[:, i],
                    vals_pca[:, j],
                    color=COLORS_RT[rt],
                    alpha=0.4,
                    label=f"{rt}",
                )
            ax.legend(loc="best", fontsize="x-small")
            fig.savefig(os.path.join(dir_store, f"{i}_{j}.png"))
            plt.close(fig)
    return dir_store


def compute_metric(metric, predictions, model):
    if metric == "all_atom_rmsd":
        return compute_all_atom_rmsd(predictions, model)  # List of floats
    elif metric == "seq_rec_rate":
        return compute_sec_rec_rate(predictions, model)  # List of floats
    elif metric == "kl_latent_dist":
        return compute_kl_latent(predictions, model)  # List of floats
    # elif metric == "std_latent_dist":
    #     return compute_std_latent_dist(predictions, model)  # List of floats
    else:
        raise OSError(f"Metric {metric} not implemented")


def get_df_stats(df):
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    means = df[numeric_cols].mean()
    stds = df[numeric_cols].std()

    stats_data = {"stat_type": ["mean", "std"]}
    for col in numeric_cols:
        stats_data[col] = [means[col], stds[col]]

    return pd.DataFrame(stats_data)


def main() -> None:
    load_dotenv()

    # Load config
    args, cfg, config_name = parse_args_and_cfg()
    ae_name, ckpt_name = extract_ckpt_info(cfg.ckpt_file)

    # Some setup
    root_path = setup(cfg, create_root=True, config_name=config_name)
    df_file_store = os.path.join(root_path, f"../results_{config_name}.csv")
    df_file_store_summary = os.path.join(root_path, f"../results_{config_name}_summary.csv")

    # Get dataloader
    dataloader = load_dataloader(cfg)

    # Model
    model = AutoEncoder.load_from_checkpoint(cfg.ckpt_file)

    # Make predictions, store them together with inputs
    trainer = L.Trainer(accelerator="gpu", devices=1, limit_predict_batches=int(cfg.n_structs / cfg.bs))
    predictions = trainer.predict(model, dataloader)
    # List of tuples, each tuple is (data_batch, predicted_batch)
    # and the predicted batch has all outputs from the endocer and decoder

    # Compute requested metrics
    metrics = {}
    metrics_to_compute = [k for k in cfg.metrics if cfg.metrics[k]]
    for metric in metrics_to_compute:
        metrics[metric] = compute_metric(metric=metric, predictions=predictions, model=model)

    # Extract PDB ids
    pdb_id = extract_pdb_ids(predictions)  # List of strs

    # Plot requested stuff
    dir_storages = {}
    if cfg.plot.pca_latents_per_residue_type:
        dir_storages["pca_per_aatype"] = plot_latent_pca_per_residue_type(
            predictions, root_path, config_name
        )  # Saves figures

    # Create dataframe with results
    info_df = {"pdb_id": pdb_id}
    info_df.update(metrics)
    df = pd.DataFrame(info_df)

    # Save summary results
    col_names = ["ae_name", "ckpt_name", "dataset"]
    values = [ae_name, ckpt_name, cfg.dataset]
    for m in metrics:
        col_names += [f"{m}_mean", f"{m}_std", f"{m}_max", f"{m}_min"]
        vals_aux = np.array(metrics[m])
        values += [vals_aux.mean(), vals_aux.std(), vals_aux.max(), vals_aux.min()]
    col_names += [k for k in dir_storages]
    values += [dir_storages[k] for k in dir_storages]
    df_summary = pd.DataFrame({col_names[i]: [values[i]] for i in range(len(col_names))})

    # Save dataframes
    df.to_csv(df_file_store, index=False)
    df_summary.to_csv(df_file_store_summary, index=False)

    # Save df
    df.to_csv(df_file_store, index=False)
    df_summary.to_csv(df_file_store_summary, index=False)
    print("Done saving dataframes")


if __name__ == "__main__":
    main()
