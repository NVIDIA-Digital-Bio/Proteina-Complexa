import os
from typing import Literal

import lightning as L
import matplotlib.pyplot as plt
import torch
import wandb
from jaxtyping import Bool, Float, Int
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from loguru import logger
from sklearn.decomposition import PCA
from torch import Tensor

from proteinfoundation.partial_autoencoder.decoder import DecoderTransformer
from proteinfoundation.partial_autoencoder.decoder_ff import DecoderFFLocal
from proteinfoundation.partial_autoencoder.encoder import EncoderTransformer
from proteinfoundation.utils.coors_utils import nm_to_ang
from proteinfoundation.utils.pdb_utils import write_prot_to_pdb

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


@rank_zero_only
def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)


class AutoEncoder(L.LightningModule):
    def __init__(self, cfg_ae, store_dir=None):
        super().__init__()
        self.save_hyperparameters()
        self.cfg_ae = cfg_ae

        self.store_dir = store_dir if store_dir is not None else "./tmp_ae"
        self.val_path_tmp = os.path.join(self.store_dir, "val_stuff")
        # create_dir(self.val_path_tmp)

        self.encoder = EncoderTransformer(**self.cfg_ae.nn_ae)

        decoder_type = self.cfg_ae.nn_ae.decoder.get("type", "transformer")
        if decoder_type == "transformer":
            self.decoder = DecoderTransformer(**self.cfg_ae.nn_ae)
        elif decoder_type == "ff_local":
            self.decoder = DecoderFFLocal(**self.cfg_ae.nn_ae)
        else:
            raise OSError(f"Invalid decoder_type {decoder_type}")

        self.nsamples_processed = 0
        self.nparams_enc = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        self.nparams_dec = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)

        self.validation_output = []
        self.validation_data_samples = []
        self.validation_rec_samples = []

        self.latent_dim = self.cfg_ae.nn_ae["latent_z_dim"]

    def log_histogram(self, id_log: str, v: Float[torch.Tensor, "r"]):
        """
        Logs histogram, v must be a flat tensor.
        """
        assert v.ndim == 1, f"Tensor v has shape {v.shape}, cannot log histogram"
        try:
            self.logger.experiment.log({id_log: wandb.Histogram(v.cpu().detach().numpy())})
        except:
            # If you're not seeing histograms come back here and check why
            pass

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(
        #     [p for p in self.parameters() if p.requires_grad], lr=self.cfg_ae.opt.lr,
        # )
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.cfg_ae.opt.lr,
            amsgrad=True,
            weight_decay=1e-2,
        )
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        """Adds additional variables to checkpoint."""
        checkpoint["nsamples_processed"] = self.nsamples_processed

    def on_load_checkpoint(self, checkpoint):
        """Loads additional variables from checkpoint."""
        try:
            self.nsamples_processed = checkpoint["nsamples_processed"]
        except:
            logger.info("Failed to load nsamples_processed from checkpoint")
            self.nsamples_processed = 0

    def encode(self, batch: dict) -> Float[torch.Tensor, "b n d"]:
        """
        Runs the encoder and returns only the latent variables.
        """
        if "mask" not in batch:
            mask = batch["mask_dict"]["coords"][..., 0, 0]  # [b, n] boolean
            batch["mask"] = mask
        # return self.encoder(batch)["z_latent"]
        return self.encoder(batch)  # z_latent, mean, log_scale

    def decode(
        self,
        z_latent: Float[torch.Tensor, "b n d"],
        ca_coors_nm: Float[torch.Tensor, "b n 3"],
        mask: Bool[torch.Tensor, "b n"],
    ) -> dict:
        """
        Runs the decoder and returns a dictionary with all necessary decoding information.
        """
        input_decoder = {
            "z_latent": z_latent,
            "ca_coors_nm": ca_coors_nm,
            "residue_mask": mask,
            "mask": mask,
        }
        output_dec = self.decoder(input_decoder)
        mask = output_dec["residue_mask"]  # [b, n]
        atom_mask = output_dec["atom_mask"]  # [b, n, 37]
        coors_nm = output_dec["coors_nm"] * mask[..., None, None] * atom_mask[..., None]  # [b, n, 37, 3]
        return {
            "coors_nm": coors_nm,
            "seq_logits": output_dec["seq_logits"] * mask[..., None],
            "residue_type": output_dec["aatype_max"] * mask,
            "residue_mask": mask,
            "atom_mask": atom_mask,
        }

    def training_step(self, batch: dict, batch_idx: int):
        """
        Computes training loss for batch of samples.

        Args:
            batch: Data batch.

        Returns:
            Training loss averaged over batch dimension.
        """
        val_step = batch_idx == -1  # validation step is indicated with batch_idx -1
        log_prefix = "validation_loss" if val_step else "train"
        histogram_every_n = 5000
        pca_every_n = 5000
        per_aatype_kl = True

        mask = batch["mask_dict"]["coords"][..., 0, 0]  # [b, n] boolean
        batch["mask"] = mask
        ca_coors_nm = batch["coords_nm"][..., 1, :]  # [b, n, 3]
        ca_coors_nm = ca_coors_nm * mask[..., None]  # [b, n, 3]
        bs, n = mask.shape[0], mask.shape[1]

        output_enc = self.encoder(batch)
        # {
        #   "z_latent": latent_sample, shape [b, n, d]
        #   "mean": mean of latent (diag) Gaussian dist, shape [b, n, d]
        #   "log_scale": log standard deviation of latent (diag) Gaussian dist, shape [b, n, d]
        # }
        log_prefix_stats = "train_stats_latent" if "train" in log_prefix else "val_stats_latent"
        self.log_tensor_statistics(
            bs=bs,
            v=output_enc["mean"],
            log_prefix=log_prefix_stats + "_mean_LS",
            mask=mask,
            histogram_every_n=histogram_every_n,
        )
        self.log_tensor_statistics(
            bs=bs,
            v=torch.exp(output_enc["log_scale"]),
            log_prefix=log_prefix_stats + "_scale_LS",
            mask=mask,
            histogram_every_n=histogram_every_n,
        )
        self.log_tensor_statistics(
            bs=bs,
            v=output_enc["z_latent"],
            log_prefix=log_prefix_stats + "_z_LS",
            mask=mask,
            histogram_every_n=histogram_every_n,
        )
        self.log_pca(
            v=output_enc["z_latent"],
            log_prefix=log_prefix_stats + "_z_PCA",
            mask=mask,
            every_n=pca_every_n,
        )
        self.log_pca_per_residue_type(
            v=output_enc["z_latent"],
            log_prefix=log_prefix_stats + "_z_PCA_pre_aatype",
            mask=mask,
            every_n=pca_every_n,
            res_ty=batch["residue_type"],
        )

        input_decoder = {
            "z_latent": output_enc["z_latent"],
            "ca_coors_nm": ca_coors_nm,
            "residue_mask": mask,
            "mask": mask,
        }
        output_dec = self.decoder(input_decoder)
        # {
        #   "coors_nm": all atom coordinates, shape [b, n, 37, 3], in nm
        #   "seq_logits": logits for the residue types, shape [b, n, 20]
        #   "residue_mask": boolean [b, n]
        #   "aatype_max": residue type by taking the most likely logit, shape [b, n], with integer values {0, ..., 19}
        #   "atom_mask": boolean [b, n, 37, 3], atom37 mask corresponding to aatype_max
        # }

        # If we are working with side chain angles, or anything else, could output those too
        # Maybe for simplicity define the reconstruction loss in the decoder? So that the
        # decoder knows what loss to use and no need to pass it around here

        losses = {}  # Will be a Dict[str, tensor[b]]. If "_justlog" in name just for logging

        # KL loss with annealing weight
        f = min(1.0, self.global_step / self.cfg_ae.loss.kl.patience) if self.cfg_ae.loss.kl.anneal else 1.0
        self.log(
            "kl_weight",
            self.cfg_ae.loss.kl.weight * f,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        losses.update(
            self.compute_kl_penalty(
                mean=output_enc["mean"],
                log_scale=output_enc["log_scale"],
                mask=mask,
                w=self.cfg_ae.loss.kl.weight * f,
            )
        )

        # Structure loss
        losses.update(
            self.compute_struct_rec_loss(
                output_dec=output_dec,
                batch=batch,
                reduce_mode="sum",
                loss_ty=self.cfg_ae.loss.struct.type,
                weight=self.cfg_ae.loss.struct.weight,
            )
        )

        # Sequence loss
        losses.update(
            self.compute_seq_rec_loss(
                output_dec=output_dec,
                batch=batch,
                weight=self.cfg_ae.loss.seq.weight,
            )
        )

        # Log losses and training loss, losses with "_justlog" just for logging purposes
        self.log_losses(bs, losses, log_prefix)
        train_loss = sum([torch.mean(losses[k]) for k in losses if "_justlog" not in k])

        self.log(
            f"{log_prefix}/loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=bs,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        if not val_step:  # Don't log these for val step
            # Log Kl statistics
            componentwise_kl = self._per_component_kl(
                mean=output_enc["mean"],
                log_scale=output_enc["log_scale"],
                mask=mask,
            )  # [b, n, d]
            self.log_tensor_statistics(
                bs=bs,
                v=componentwise_kl,
                log_prefix=log_prefix + "_kl_latent",
                mask=mask,
                histogram_every_n=histogram_every_n,
            )
            self.log_tensor_statistics(
                bs=bs,
                v=(componentwise_kl > 0.1) * 1.0,
                log_prefix=log_prefix + "_kl_latent_active_thresh_0p1",
                mask=mask,
                do_hist=False,
                stats_to_log=["mean"],
            )
            # KL per aa type
            if per_aatype_kl and self.global_step % histogram_every_n == 2000:
                for i in range(20):
                    self.log_tensor_statistics(
                        bs=bs,
                        v=(componentwise_kl > 0.1) * 1.0,
                        log_prefix=log_prefix_stats + "_kl_latent_active_thresh_0p1_per_aatype",
                        mask=mask * (batch["residue_type"] == i),
                        do_hist=False,
                        suffix=f"_aa_{i}",
                        stats_to_log=["mean"],
                    )

            self.log_train_loss_n_prog_bar(bs, train_loss)
            self.update_n_log_nsamples_processed(bs)
            self.log_nparams()

        if val_step:
            return train_loss, output_dec
        return train_loss

    def compute_struct_rec_loss(
        self,
        output_dec: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        reduce_mode: Literal["sum", "mean"] = "sum",
        # loss_ty: Literal["l1", "l2", "l12", "l2cut"]="l2",
        loss_ty: str = "l2",
        weight: float = 1.0,
    ) -> dict[str, Float[Tensor, "b"]]:
        """
        Computes structural loss.

        Args:
            output_dec:
                {
                  "coors_nm": all atom coordinates, shape [b, n, 37, 3], in nm
                  "seq_logits": logits for the residue types, shape [b, n, 20]
                  "residue_mask": boolean [b, n]
                  "aatype_max": residue type by taking the most likely logit, shape [b, n], with integer values {0, ..., 19}
                  "atom_mask": boolean [b, n, 37, 3], atom37 mask corresponding to aatype_max
                }
            batch: data batch from dataloader
            reduce_mode: whether to apply mean over atoms or just sum, when reducing accross 37 atoms types
            loss_ty: whether to apply l1 loss, l2 loss, or both, l2cut (ignore atoms close to each other, to expand)

        Returns:
            Dictionary [str, tensor[b]] with multiple per batch element losses. If the key has "_justlog" then this loss
            will not be used to compute the total loss, but will just be logged.
        """

        def reduce_37(
            err: Float[torch.Tensor, "b n 37 3"],
            mask: Bool[torch.Tensor, "b n"],
            atom_mask: Bool[torch.Tensor, "b n 37"],
            mode: Literal["sum", "mean"] = "sum",
        ) -> Float[torch.Tensor, "b"]:
            nres = mask.sum(dim=-1)  # [b]
            nat = atom_mask.sum(dim=-1) * mask  # [b, n]
            err = torch.sum(err, dim=(-1, -2))  # [b, n]
            if mode == "mean":
                err = err / nat  # Take mean over existing atoms if mode == "mean"
            err = err.sum(dim=-1) / nres  # [b]
            return err

        coors_nm_pred = output_dec["coors_nm"]  # [b, n, 37, 3]
        coors_nm_true = batch["coords_nm"]  # [b, n, 37, 3]
        mask = output_dec["residue_mask"]  # [b, n] boolean
        atom_mask_true = batch["coord_mask"] * mask[..., None]  # [b, n, 37] boolean

        err = coors_nm_true - coors_nm_pred  # [b, n, 37, 3]
        err = err * mask[..., None, None] * atom_mask_true[..., None]  # [b, n, 37, 3]
        # Should we do something about residues with more atoms than others?
        # If we just sum they contribute more to the loss

        losses = {}

        # Compute RMSD in Å (without alignment)
        err_ang = nm_to_ang(err)  # [b, n, 37, 3]
        err_ang = torch.linalg.norm(err_ang, dim=-1) ** 2  # [b, n, 37]
        err_ang = err_ang * mask[..., None] * atom_mask_true  # [b, n, 37]
        nat = atom_mask_true.sum((-1, -2))  # [b]
        rmsd = torch.sqrt(torch.sum(err_ang, dim=(-1, -2)) / nat)  # [b]
        losses["rmsd_no_align_a37_ang_justlog"] = rmsd

        err_l1 = reduce_37(torch.abs(err), mask, atom_mask_true, mode=reduce_mode)  # L1 loss
        err_l2 = reduce_37(err**2, mask, atom_mask_true, mode=reduce_mode)  # L2 loss

        if "l2cut" in loss_ty:  # l2cut_0.5 or something like this
            lower_lim = float(loss_ty.split("_")[-1])
            err_sq_at = (err**2).sum(dim=-1)  # [b, n, 37]
            mask_lim = err_sq_at > lower_lim  # Which elements should be used, shape [b, n, 37]
            err_sq_at = err_sq_at * mask[..., None] * atom_mask_true * mask_lim
            err_l2cut = reduce_37(err_sq_at[..., None], mask, atom_mask_true * mask_lim, mode=reduce_mode)  # L2 loss

        if loss_ty == "l1":
            losses["struct_l1"] = err_l1 * weight
            losses["struct_l2_justlog"] = err_l2 * weight
        elif loss_ty == "l2":
            losses["struct_l1_justlog"] = err_l1 * weight
            losses["struct_l2"] = err_l2 * weight
        elif loss_ty == "l12":
            losses["struct_l1"] = err_l1 * weight
            losses["struct_l2"] = err_l2 * weight
        elif "l2cut" in loss_ty:
            losses["struct_l1_justlog"] = err_l1 * weight
            losses["struct_l2_justlog"] = err_l2 * weight
            losses["struct_l2cut"] = err_l2cut * weight
        else:
            raise OSError(f"Loss type {loss_ty} not recognized")
        return losses

    def compute_seq_rec_loss(
        self,
        output_dec: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        weight: float = 1.0,
    ) -> dict[str, Float[Tensor, "b"]]:
        """
        Computes cross entropy loss on predicted sequence.

        Args:
            output_dec:
                {
                  "coors_nm": all atom coordinates, shape [b, n, 37, 3], in nm
                  "seq_logits": logits for the residue types, shape [b, n, 20]
                  "residue_mask": boolean [b, n]
                  "aatype_max": residue type by taking the most likely logit, shape [b, n], with integer values {0, ..., 19}
                  "atom_mask": boolean [b, n, 37, 3], atom37 mask corresponding to aatype_max
                }
            batch: data batch from dataloader

        Returns:
            Dictionary [str, tensor[b]] with multiple per batch element losses. If the key has "_justlog" then this loss
            will not be used to compute the total loss, but will just be logged.
        """
        assert "residue_type" in batch, "`residue_type` not in batch, failed in compute_seq_rec_loss"
        mask = output_dec["residue_mask"]  # [b, n]
        nres = mask.sum(dim=-1)  # [b]
        logits_pred = output_dec["seq_logits"]  # [b, n, 20]
        target_aa = batch["residue_type"]  # [b, n]
        target_aa = target_aa * mask  # [b, n] gets rid of -1 for padding (issue with cross entropy loss below)

        assert logits_pred.shape[-1] == 20, "Wrong number of logits"

        # Compute cross entropy
        b, n = mask.shape[0], mask.shape[1]
        logits_pred_flat = logits_pred.view(b * n, 20)  # [b * n, 20]
        target_aa_flat = target_aa.view(b * n)  # [b * n]
        # seq_loss_flat = torch.nn.functional.cross_entropy(
        #     input=logits_pred_flat, target=target_aa_flat, reduction="none", label_smoothing=0.001,  # # # SMOOTHING!!!
        # )  # [b * n]
        seq_loss_flat = torch.nn.functional.cross_entropy(
            input=logits_pred_flat,
            target=target_aa_flat,
            reduction="none",
        )  # [b * n]
        seq_loss = seq_loss_flat.view(b, n)  # [b, n]
        seq_loss = seq_loss * mask  # [b, n]
        seq_loss = torch.sum(seq_loss, dim=-1) / nres  # [b]

        # Compute seq recovery rate
        pred_aa = output_dec["aatype_max"]  # [b, n]
        seq_rec = pred_aa == target_aa  # [b, n]
        seq_rec = seq_rec * mask  # [b, n]
        seq_rec_rate = seq_rec.sum(dim=-1) / nres  # [b]
        return {
            "ce_seq": seq_loss * weight,
            "ce_seq_now": seq_loss,
            "seq_rec_rate_justlog": seq_rec_rate,
        }

    def _per_component_kl(
        self,
        mean: Float[Tensor, "b n d"],
        log_scale: Float[Tensor, "b n d"],
        mask: Bool[Tensor, "b n"],
    ) -> Float[Tensor, "b n d"]:
        """
        Computes KL penalty on the latent Gaussian distribution, per residue.

        Returns KL, per residue (masked) and latent dimension, shape [b n d].
        """
        scale = torch.exp(log_scale)  # [b, n, 3]
        kl_div = (scale**2 + mean**2 - 1.0 - 2.0 * log_scale) * 0.5  # [b, n, d]
        return kl_div * mask[..., None]  # [b, n, d]

    def compute_kl_penalty(
        self,
        mean: Float[Tensor, "b n d"],
        log_scale: Float[Tensor, "b n d"],
        mask: Bool[Tensor, "b n"],
        w: float,
    ) -> dict[str, Float[Tensor, "b"]]:
        """
        Computes KL penalty on the latent Gaussian distribution.
        """
        nres = torch.sum(mask, dim=-1)  # [b]
        kl_div = self._per_component_kl(
            mean=mean,
            log_scale=log_scale,
            mask=mask,
        )  # [b, n, d]
        kl_div = torch.sum(kl_div, dim=(-1, -2))  # [b]
        kl_div = kl_div / nres  # [b]
        loss = {
            "kl_w": kl_div * w,
            "kl_now_justlog": kl_div,
        }
        return loss

    def log_tensor_statistics(
        self,
        bs: int,
        v: Float[torch.Tensor, "b n d"],
        log_prefix: str,
        mask: Bool[torch.Tensor, "b n"],
        do_stats: bool = True,
        do_hist: bool = True,
        suffix: str = "",
        histogram_every_n: int = 3000,
        stats_to_log: list[str] = ["max", "min", "mean", "median", "std"],
    ) -> None:
        """
        Logs tensor statistics: mean, median, max, min, std, ...

        If res_type is not None, then it also logs histograms per residue type
        """
        vals = v.clone()[mask]  # [num of unmasked residues, d]
        vals = torch.flatten(vals)  # 1D vector
        if do_stats:
            funs = {
                "max": torch.max,
                "min": torch.min,
                "mean": torch.mean,
                "median": torch.median,
                "std": torch.std,
            }
            for k in stats_to_log:
                self.log(
                    f"{log_prefix}/{k}{suffix}",
                    # funs[k](vals).cpu().detach(),
                    funs[k](vals),
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=bs,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

        if self.global_step % histogram_every_n == 0 and do_hist:
            self.log_histogram(id_log=f"{log_prefix}/histogram{suffix}", v=vals)

    def log_pca(
        self,
        v: Float[torch.Tensor, "b n d"],
        log_prefix: str,
        mask: Bool[torch.Tensor, "b n"],
        every_n: int = 3000,
    ) -> None:
        """
        Logs PCA components plot of latent variable z.
        """

        def _log_scatter(vals_x, vals_y, xlabel, ylabel, log_id):
            fig, ax = plt.subplots()
            ax.scatter(vals_x, vals_y)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{xlabel} - {ylabel} - Step: {self.trainer.global_step}")
            try:
                self.logger.experiment.log({log_id: wandb.Image(fig)})
            except:
                # If you don't see the PCA plots come back here
                pass
            plt.close("all")

        if self.global_step % every_n != 0:
            return

        n_components = 4
        vals = v.clone()[mask].cpu().detach().float().numpy()  # [num of unmasked residues, d]
        vals_pca = PCA(n_components=n_components).fit_transform(vals)  # [num of unmasked residues, n_components]

        for i in range(n_components):
            for j in range(n_components):
                if j <= i:
                    continue
                _log_scatter(
                    vals_x=vals_pca[:, i],
                    vals_y=vals_pca[:, j],
                    xlabel=f"PC {i}",
                    ylabel=f"PC {j}",
                    log_id=f"{log_prefix}/{i}_{j}",
                )

    def log_pca_per_residue_type(
        self,
        v: Float[torch.Tensor, "b n d"],
        res_ty: Int[torch.Tensor, "b n"],
        log_prefix: str,
        mask: Bool[torch.Tensor, "b n"],
        every_n: int = 3000,
    ) -> None:
        """
        Logs PCA components plot of latent variable z.
        """
        return None
        # if self.global_step % every_n != 0:
        #     return

        # n_components = 4
        # for i in range(n_components):
        #     for j in range(n_components):
        #         if j <= i:
        #             continue
        #         fig, ax = plt.subplots()
        #         ax.set_xlabel(f"PC {i}")
        #         ax.set_ylabel(f"PC {j}")
        #         ax.set_title(f"PC {i} - PC {j} - Step: {self.trainer.global_step}")

        #         for rt in range(20):
        #             mask_rt = res_ty == rt  # [b, n]
        #             mask_rt = mask_rt * mask  # [b, n]
        #             vals = v.clone()[mask_rt].cpu().detach().float().numpy()  # [num of unmasked residues, d]
        #             vals_pca = PCA(n_components=n_components).fit_transform(vals)  # [num of unmasked residues, n_components]
        #             ax.scatter(vals_pca[:, i], vals_pca[:,j], color=COLORS_RT[rt], alpha=0.4, label=f"{rt}")
        #         ax.legend(loc="best", fontsize="x-small")

        #         log_id = f"{log_prefix}/{i}_{j}"
        #         try:
        #             self.logger.experiment.log({log_id: wandb.Image(fig)})
        #         except:
        #             # If you don't see the PCA plots come back here
        #             pass

        #         plt.close("all")

    def log_losses(self, bs: int, losses: dict[str, Float[torch.Tensor, "b"]], log_prefix: str) -> None:
        for k in losses:
            log_name = k[: -len("_justlog")] if k.endswith("_justlog") else k
            self.log(
                f"{log_prefix}/loss_{log_name}",
                torch.mean(losses[k]),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=bs,
                sync_dist=True,
                add_dataloader_idx=False,
            )

    def log_train_loss_n_prog_bar(self, b: int, train_loss: torch.Tensor) -> None:
        self.log(
            "train_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=b,
            sync_dist=True,
            add_dataloader_idx=False,
        )

    def log_nparams(self):
        self.log(
            "scaling/nparams_enc",
            self.nparams_enc * 1.0,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )  # constant line but ok, easy to compare # params

        self.log(
            "scaling/nparams_dec",
            self.nparams_dec * 1.0,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )  # constant line but ok, easy to compare # params

    def update_n_log_nsamples_processed(self, b: int):
        self.nsamples_processed = self.nsamples_processed + b * self.trainer.world_size
        self.log(
            "scaling/nsamples_processed",
            self.nsamples_processed * 1.0,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )

    def validation_step(self, batch: dict, batch_idx: int):
        """
        Evalaute validation loss.

        Args:
            batch: batch from dataset (see last argument)
            batch_idx: batch index (unused)
        """
        with torch.no_grad():
            bs = batch["coords_nm"].shape[0]
            loss, rec_sample = self.training_step(batch, batch_idx=-1)
            self.validation_output.append(loss.item())
            for i in range(bs):
                self.validation_data_samples.append(
                    {
                        "coors_nm": batch["coords_nm"][i, ...],  # [n, 37, 3]
                        "aatype": batch["residue_type"][i, ...],  # [n]
                        "atom_mask": batch["coord_mask"][i, ...],  # [n, 37]
                        "mask": batch["mask_dict"]["coords"][i, :, 0, 0],  # [n]
                    }
                )
                self.validation_rec_samples.append(
                    {
                        "coors_nm": rec_sample["coors_nm"][i, ...],  # [n, 37, 3]
                        "aatype": rec_sample["aatype_max"][i, ...],  # [n]
                        "atom_mask": rec_sample["atom_mask"][i, ...],  # [n, 37]
                        "mask": rec_sample["residue_mask"][i, ...],  # [n], should be the same as mask from orig sample
                    }
                )

    def on_validation_epoch_end(self):
        """
        Stores samples as PDBs and cleans validation results.
        """
        count = 0
        for sample_data, sample_rec in zip(self.validation_data_samples, self.validation_rec_samples, strict=False):
            count += 1
            if count > 10:
                break
            coors_data = nm_to_ang(sample_data["coors_nm"]).float().detach().cpu().numpy()  # [n, 37, 3]
            aatype_data = sample_data["aatype"].detach().cpu().numpy()  # [n]
            mask_data = sample_data["mask"].detach().cpu().numpy()  # [n]
            atom_mask_data = sample_data["atom_mask"].detach().cpu().numpy()  # [n, 37]

            coors_rec = nm_to_ang(sample_rec["coors_nm"]).float().detach().cpu().numpy()
            aatype_rec = sample_rec["aatype"].detach().cpu().numpy()
            mask_rec = sample_rec["mask"].detach().cpu().numpy()
            atom_mask_rec = sample_rec["atom_mask"].detach().cpu().numpy()

            f_base = f"epoch_{self.current_epoch}_id_{count}_rank_{self.global_rank}"

            # Save original sample
            fname = f_base + "_data_sample.pdb"
            if not os.path.exists(self.val_path_tmp):
                create_dir(self.val_path_tmp)
            fpath = os.path.join(self.val_path_tmp, fname)
            write_prot_to_pdb(
                prot_pos=coors_data * mask_data[..., None, None] * atom_mask_data[..., None],
                file_path=fpath,
                aatype=aatype_data * mask_data,
                overwrite=True,
                no_indexing=True,
            )

            # Save recovered sample
            fname = f_base + "_rec_sample.pdb"
            fpath = os.path.join(self.val_path_tmp, fname)
            write_prot_to_pdb(
                prot_pos=coors_rec * mask_rec[..., None, None] * atom_mask_rec[..., None],
                file_path=fpath,
                aatype=aatype_rec * mask_rec,
                overwrite=True,
                no_indexing=True,
            )

            # Save recovered sample with true sequence
            fname = f_base + "_rec_sample_w_true_seq.pdb"
            fpath = os.path.join(self.val_path_tmp, fname)
            write_prot_to_pdb(
                prot_pos=coors_rec * mask_data[..., None, None] * atom_mask_data[..., None],
                file_path=fpath,
                aatype=aatype_data * mask_data,
                overwrite=True,
                no_indexing=True,
            )

        self.validation_data_samples = []
        self.validation_rec_samples = []
        self.validation_output_data = []
        # Should log here?

    def predict_step(self, batch: dict, batch_idx: int) -> dict:
        """
        Makes predictions. Given a data batch, encodes, and returns decoded batch.

        Args:
            batch: data batch.

        Returns:
            Dict representing the decoded batch, with all info from the encoder output.
        """
        mask = batch["mask_dict"]["coords"][..., 0, 0]  # [b, n] boolean
        batch["mask"] = mask
        ca_coors_nm = batch["coords_nm"][..., 1, :]  # [b, n, 3]
        ca_coors_nm = ca_coors_nm * mask[..., None]  # [b, n, 3]

        output_enc = self.encoder(batch)
        # {
        #   "z_latent": latent_sample, shape [b, n, d]
        #   "mean": mean of latent (diag) Gaussian dist, shape [b, n, d]
        #   "log_scale": log standard deviation of latent (diag) Gaussian dist, shape [b, n, d]
        # }

        input_decoder = {
            "z_latent": output_enc["z_latent"],
            "ca_coors_nm": ca_coors_nm,
            "residue_mask": mask,
            "mask": mask,
        }
        output = self.decoder(input_decoder)
        # {
        #   "coors_nm": all atom coordinates, shape [b, n, 37, 3], in nm
        #   "seq_logits": logits for the residue types, shape [b, n, 20]
        #   "residue_mask": boolean [b, n]
        #   "aatype_max": residue type by taking the most likely logit, shape [b, n], with integer values {0, ..., 19}
        #   "atom_mask": boolean [b, n, 37, 3], atom37 mask corresponding to aatype_max
        # }

        output.update(output_enc)
        return (batch, output)
