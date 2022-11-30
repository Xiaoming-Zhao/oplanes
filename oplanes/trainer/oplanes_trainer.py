import numpy as np

import torch
import pytorch_lightning as pl

from oplanes.models import *
import oplanes.utils.comm as comm
from oplanes.utils.crit_utils import dice_loss
from oplanes.utils.registry import registry


class OPlanes_Trainer(pl.LightningModule):
    def __init__(
        self,
        *,
        model_name,
        backbone_name,
        pos_enc_kwargs={},
        feat_channels=256,
        trainset=None,
        learning_rate=1e-3,
        coeff_ce=1.0,
        coeff_ce_pos=1.0,
        coeff_ce_neg=1.0,
        coeff_dice=1.0,
        intermediate_supervise=True,
    ):
        super().__init__()
        self.model_name = model_name
        model_cls = registry.get_model(model_name)
        assert model_cls is not None, f"Cannot find model {model_name}."
        self.model = model_cls(
            backbone_name,
            pos_enc_kwargs=pos_enc_kwargs,
            feat_channels=feat_channels,
        )

        self.crit_ce = torch.nn.BCEWithLogitsLoss(reduction="none")

        self.intermediate_supervise = intermediate_supervise

        self.debug = True
        self.trainset = trainset

        self.coeff_ce = coeff_ce
        self.coeff_ce_pos = coeff_ce_pos
        self.coeff_ce_neg = coeff_ce_neg
        self.coeff_dice = coeff_dice

        self.learning_rate = learning_rate

        self.vis_threshold = 0.1
        self.gt_pos_threshold = 0.5

        self.avoid_nan_eps = 1e-8

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        print("\nLearning rate: ", self.learning_rate, "\n")
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer  # [optimizer], [scheduler]

    def _compute_loss(self, out, gt, select_mask, is_val=False):
        masked_gt = select_mask * gt
        masked_out = select_mask * out

        if is_val or ((not is_val) and (self.coeff_ce > 0)):
            loss_ce_per_pix = self.crit_ce(out, gt)  # [B, C, H, W]
            gt_pos_mask = (gt > self.gt_pos_threshold).float()
            loss_ce_pos_per_pix = loss_ce_per_pix * select_mask * gt_pos_mask  # CE loss for positive
            loss_ce_neg_per_pix = loss_ce_per_pix * select_mask * (1 - gt_pos_mask)  # CE loss for negatve
            loss_ce_pos = torch.sum(loss_ce_pos_per_pix) / (torch.sum(select_mask * gt_pos_mask) + self.avoid_nan_eps)
            loss_ce_neg = torch.sum(loss_ce_neg_per_pix) / (
                torch.sum(select_mask * (1 - gt_pos_mask)) + self.avoid_nan_eps
            )
            loss_ce = self.coeff_ce_pos * loss_ce_pos + self.coeff_ce_neg * loss_ce_neg
        else:
            loss_ce_pos = torch.zeros(1, device=gt.device)
            loss_ce_neg = torch.zeros(1, device=gt.device)
            loss_ce = torch.zeros(1, device=gt.device)

        if is_val or ((not is_val) and (self.coeff_dice > 0)):
            b, n_planes, h, w = out.shape
            masked_gt_flat = masked_gt.reshape((b * n_planes, h, w))
            masked_out_flat = masked_out.reshape((b * n_planes, h, w))

            # we use some threshold. Since 'out' is raw logits, discriminatory boundary is at 0.
            masked_prob_flat = select_mask.reshape((b * n_planes, h, w)) * masked_out_flat.sigmoid()

            # [B x #planes, ]
            loss_dice_per_plane = dice_loss(masked_prob_flat, masked_gt_flat)
            loss_dice = loss_dice_per_plane.mean()
        else:
            loss_dice_per_plane = torch.zeros(1, device=gt.device)
            loss_dice = torch.zeros(1, device=gt.device)

        return loss_ce, loss_ce_pos, loss_ce_neg, loss_dice, loss_dice_per_plane

    def _compute_acc(self, out, gt, select_mask):

        numel = torch.sum(select_mask).item()

        # alternative way to compute accuracy.
        gt_pos_mask = (gt >= self.gt_pos_threshold).float()
        out_pos_mask = (out > 0).float()

        # [B, #planes]
        numel_per_plane = torch.sum(select_mask, dim=[2, 3])
        masked_tp_per_plane = torch.sum(select_mask * gt_pos_mask * out_pos_mask, dim=[2, 3])  # true positive
        masked_fp_per_plane = torch.sum(select_mask * (1 - gt_pos_mask) * out_pos_mask, dim=[2, 3])  # false positive
        masked_tn_per_plane = torch.sum(
            select_mask * (1 - gt_pos_mask) * (1 - out_pos_mask), dim=[2, 3]
        )  # true negative
        masked_fn_per_plane = torch.sum(select_mask * gt_pos_mask * (1 - out_pos_mask), dim=[2, 3])  # false negative
        acc_per_plane = (masked_tp_per_plane + masked_tn_per_plane) / (numel_per_plane + self.avoid_nan_eps)
        # [B, 1, #planes]
        acc_per_plane = acc_per_plane.detach().cpu().unsqueeze(1)

        masked_tp = torch.sum(masked_tp_per_plane).item()
        masked_fp = torch.sum(masked_fp_per_plane).item()
        masked_tn = torch.sum(masked_tn_per_plane).item()
        masked_fn = torch.sum(masked_fn_per_plane).item()
        acc = (masked_tp + masked_tn) / (numel + self.avoid_nan_eps)

        return acc, acc_per_plane

    def training_step(self, batch, batch_idx):
        # batch:
        # - im:[B, H, W, 5]
        # - annot: [B, H, W, num_depth]
        # - vis: [B, H, W]
        # - d_im: [B, H, W]
        # - p_d: [B, 1, num_depth]

        batch_out = self.model(batch)  # [B, C, feat_h, feat_w]

        if not isinstance(batch_out, list):
            batch_out = [batch_out]

        if not self.intermediate_supervise:
            batch_out = batch_out[-1:]

        # [B, #planes, H, W]
        gt_orig = torch.permute(batch["annot"], (0, 3, 1, 2)).float()
        vis_orig = batch["vis"].unsqueeze(1)

        loss = 0
        loss_ce = 0
        loss_ce_pos = 0
        loss_ce_neg = 0
        loss_dice = 0
        loss_dice_per_plane = 0

        for i, out_list in enumerate(batch_out):

            out, flag_after_mesh = out_list

            out_shape = out.shape

            gt = torch.nn.functional.interpolate(gt_orig, size=out_shape[-2:])
            # [B, 1, H, W]
            vis = torch.nn.functional.interpolate(vis_orig, size=out_shape[-2:])
            select = (
                vis.expand((-1, out_shape[1], -1, -1)) > self.vis_threshold
            )  # only compute loss on the visible parts
            select_mask = select.float()

            select_mask = select_mask * flag_after_mesh

            (
                loss_ce_i,
                loss_ce_pos_i,
                loss_ce_neg_i,
                loss_dice_i,
                loss_dice_per_plane_i,
            ) = self._compute_loss(out, gt, select_mask, is_val=False)

            loss_i = self.coeff_ce * loss_ce_i + self.coeff_dice * loss_dice_i

            loss += loss_i
            loss_ce += loss_ce_i
            loss_ce_pos += loss_ce_pos_i
            loss_ce_neg += loss_ce_neg_i
            loss_dice += loss_dice_i
            loss_dice_per_plane += loss_dice_per_plane_i

            if i == len(batch_out) - 1:
                tmp_suffix = ""
            else:
                tmp_suffix = f"_{i}"

            self.log(
                f"train_loss{tmp_suffix}/ce",
                loss_ce_i.item(),
                sync_dist=True,
                reduce_fx=torch.mean,
            )
            self.log(
                f"train_loss{tmp_suffix}/ce_pos",
                loss_ce_pos_i.item(),
                sync_dist=True,
                reduce_fx=torch.mean,
            )
            self.log(
                f"train_loss{tmp_suffix}/ce_neg",
                loss_ce_neg_i.item(),
                sync_dist=True,
                reduce_fx=torch.mean,
            )
            self.log(
                f"train_loss{tmp_suffix}/dice",
                loss_dice_i.item(),
                sync_dist=True,
                reduce_fx=torch.mean,
            )
            self.log(
                f"train_loss{tmp_suffix}/sum",
                loss_i.item(),
                sync_dist=True,
                reduce_fx=torch.mean,
            )

        self.log(
            f"train_loss_all_sum/ce",
            loss_ce.item(),
            sync_dist=True,
            reduce_fx=torch.mean,
        )
        self.log(
            f"train_loss_all_sum/ce_pos",
            loss_ce_pos.item(),
            sync_dist=True,
            reduce_fx=torch.mean,
        )
        self.log(
            f"train_loss_all_sum/ce_neg",
            loss_ce_neg.item(),
            sync_dist=True,
            reduce_fx=torch.mean,
        )
        self.log(
            f"train_loss_all_sum/dice",
            loss_dice.item(),
            sync_dist=True,
            reduce_fx=torch.mean,
        )
        self.log(f"train_loss_all_sum/sum", loss.item(), sync_dist=True, reduce_fx=torch.mean)

        return {"loss": loss, "idx": batch["idx"]}

    def validation_step(self, batch, batch_idx):
        # assert len(batch["fn"]) == 1, f"{len(batch['fn'])}"

        batch_out = self.model(batch)  # [B, C, feat_h, feat_w]

        if not isinstance(batch_out, list):
            batch_out = [batch_out]

        # [B, #planes, H, W]
        gt_orig = torch.permute(batch["annot"], (0, 3, 1, 2)).float()
        vis_orig = batch["vis"].unsqueeze(1)

        loss = 0
        loss_ce = 0
        loss_ce_pos = 0
        loss_ce_neg = 0
        loss_dice = 0
        loss_dice_per_plane = 0

        ret_dict = {}

        for i, out_list in enumerate(batch_out):

            out, flag_after_mesh = out_list

            out_shape = out.shape

            b, n_planes, h, w = out_shape

            # [B, H, W, #planes] -> [B, #planes, H, W]
            gt = torch.nn.functional.interpolate(gt_orig, size=out_shape[-2:])
            vis = torch.nn.functional.interpolate(vis_orig, size=out_shape[-2:])
            # Select pixels in visibiity mask
            select = vis.expand((-1, out_shape[1], -1, -1)) > self.vis_threshold
            select_mask = select.float()

            select_mask = select_mask * flag_after_mesh

            (
                loss_ce_i,
                loss_ce_pos_i,
                loss_ce_neg_i,
                loss_dice_i,
                loss_dice_per_plane_i,
            ) = self._compute_loss(out, gt, select_mask, is_val=True)

            loss_dice_per_plane_i = loss_dice_per_plane_i.detach().cpu().reshape((b, 1, n_planes))

            loss_i = self.coeff_ce * loss_ce_i + self.coeff_dice * loss_dice_i

            loss += loss_i
            loss_ce += loss_ce_i
            loss_ce_pos += loss_ce_pos_i
            loss_ce_neg += loss_ce_neg_i
            loss_dice += loss_dice_i
            loss_dice_per_plane += loss_dice_per_plane_i

            acc_i, acc_per_plane_i = self._compute_acc(out, gt, select_mask)

            # - batch["plane_diff_per_plane"]: [B, 1, #planes];
            # - acc_z_per_plane: [B, 3, #planes], 1st elem for z, 2nd for acc, 3rd for dice
            zdiff_metric_per_plane_i = torch.cat(
                (
                    batch["plane_diff_per_plane"].cpu(),
                    acc_per_plane_i,
                    loss_dice_per_plane_i,
                ),
                dim=1,
            )

            ret_dict[i] = {
                "loss_ce": loss_ce_i,
                "loss_ce_pos": loss_ce_pos_i,
                "loss_ce_neg": loss_ce_neg_i,
                "loss_dice": loss_dice_i,
                "loss": loss_i,
                "acc": acc_i,
                "zdiff_metric_per_plane": zdiff_metric_per_plane_i,
            }

        ret_dict["all_sum"] = {
            "loss_ce": loss_ce,
            "loss_ce_pos": loss_ce_pos,
            "loss_ce_neg": loss_ce_neg,
            "loss_dice": loss_dice,
            "loss": loss,
        }

        return {
            "out_list_len": len(batch_out),
            "fn": batch["fn"][0],
            "idx": batch["idx"],
            **ret_dict,
        }

    def validation_epoch_end(self, validation_step_outputs):

        l = len(validation_step_outputs)

        out_list_len = validation_step_outputs[0]["out_list_len"]

        for i in range(out_list_len):
            sl_ce = sum(d[i]["loss_ce"] for d in validation_step_outputs) / l
            sl_ce_pos = sum(d[i]["loss_ce_pos"] for d in validation_step_outputs) / l
            sl_ce_neg = sum(d[i]["loss_ce_neg"] for d in validation_step_outputs) / l
            sl_dice = sum(d[i]["loss_dice"] for d in validation_step_outputs) / l
            sl = sum(d[i]["loss"] for d in validation_step_outputs) / l
            sa = sum(d[i]["acc"] for d in validation_step_outputs) / l

            if i == out_list_len - 1:
                tmp_suffix = ""
            else:
                tmp_suffix = f"_{i}"

            # Ref: https://github.com/PyTorchLightning/pytorch-lightning/discussions/6501
            self.log(f"val{tmp_suffix}/loss_ce", sl_ce, sync_dist=True, reduce_fx=torch.mean)
            self.log(
                f"val{tmp_suffix}/loss_ce_pos",
                sl_ce_pos,
                sync_dist=True,
                reduce_fx=torch.mean,
            )
            self.log(
                f"val{tmp_suffix}/loss_ce_neg",
                sl_ce_neg,
                sync_dist=True,
                reduce_fx=torch.mean,
            )
            self.log(
                f"val{tmp_suffix}/loss_dice",
                sl_dice,
                sync_dist=True,
                reduce_fx=torch.mean,
            )
            self.log(f"val{tmp_suffix}/loss", sl, sync_dist=True, reduce_fx=torch.mean)
            self.log(f"val{tmp_suffix}/acc", sa, sync_dist=True, reduce_fx=torch.mean)

        sl_ce = sum(d["all_sum"]["loss_ce"] for d in validation_step_outputs) / l
        sl_ce_pos = sum(d["all_sum"]["loss_ce_pos"] for d in validation_step_outputs) / l
        sl_ce_neg = sum(d["all_sum"]["loss_ce_neg"] for d in validation_step_outputs) / l
        sl_dice = sum(d["all_sum"]["loss_dice"] for d in validation_step_outputs) / l
        sl = sum(d["all_sum"]["loss"] for d in validation_step_outputs) / l

        self.log(f"val_all_sum/loss_ce", sl_ce, sync_dist=True, reduce_fx=torch.mean)
        self.log(f"val_all_sum/loss_ce_pos", sl_ce_pos, sync_dist=True, reduce_fx=torch.mean)
        self.log(f"val_all_sum/loss_ce_neg", sl_ce_neg, sync_dist=True, reduce_fx=torch.mean)
        self.log(f"val_all_sum/loss_dice", sl_dice, sync_dist=True, reduce_fx=torch.mean)
        self.log(f"val_all_sum/loss", sl, sync_dist=True, reduce_fx=torch.mean)
