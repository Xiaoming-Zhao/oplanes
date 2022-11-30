import torch

from .positionencoding import PositionEmbeddingSimple
from .backbone import Resnet50WithFPN, Conv2d
from oplanes.utils.registry import registry
from oplanes.utils.img_utils import depth2normal, Sobel_X, Sobel_Y


FLOATING_EPS = 1e-8


@registry.register_model(name="v_coarse2fine_conv")
class ModelCoarseToFineConv(torch.nn.Module):
    def __init__(self, backbone_name, pos_enc_kwargs={}, feat_channels=256, **kwargs):
        super().__init__()

        backbone_cls = registry.get_model(backbone_name)
        assert backbone_cls is not None, f"Cannot find backbone {backbone_cls}."
        self.backbone = backbone_cls(withPixelDecoder=True)

        self.PE = PositionEmbeddingSimple(**pos_enc_kwargs)

        self.feat_channels = feat_channels

        self.z_diff_branch = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 1, padding=0, bias=False),
            torch.nn.GroupNorm(4, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, self.feat_channels, 1, padding=0, bias=False),
            torch.nn.GroupNorm(4, self.feat_channels),
            torch.nn.ReLU(inplace=True),
        )

        self.img_branch = torch.nn.Sequential(
            torch.nn.Conv2d(256, self.feat_channels, 3, padding=1, bias=False),
            torch.nn.GroupNorm(4, self.feat_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(self.feat_channels, self.feat_channels, 3, padding=1, bias=False),
            torch.nn.GroupNorm(4, self.feat_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(self.feat_channels, self.feat_channels, 1, padding=0, bias=False),
        )

        self.spatial_branch = torch.nn.Sequential(
            torch.nn.Conv2d(2 * self.feat_channels, self.feat_channels, 3, padding=1, bias=False),
            torch.nn.GroupNorm(4, self.feat_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(self.feat_channels, self.feat_channels, 3, padding=1, bias=False),
            torch.nn.GroupNorm(4, self.feat_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(self.feat_channels, 1, 1, padding=0, bias=False),
        )

        self.fine_down_factor = 2
        self.coarse_down_factor = 4
        self.fine2coarse_factor = self.coarse_down_factor / self.fine_down_factor

    def forward(self, batched_inputs):
        # batched_inputs:
        # - im:[B, H, W, 5]
        # - annot: [B, H, W, num_depth]
        # - vis: [B, H, W]
        # - d_im: [B, H, W]
        # - p_d: [B, 1, num_depth]

        H = batched_inputs["d_im"].shape[1]
        W = batched_inputs["d_im"].shape[2]

        fine_h = H // self.fine_down_factor
        fine_w = W // self.fine_down_factor
        coarse_h = H // self.coarse_down_factor
        coarse_w = W // self.coarse_down_factor

        # -----------------------------------------------------------------------------------------------------------------
        # process depth

        # [B, H, W, #planes]
        z_diff_raw = batched_inputs["d_im"].unsqueeze(-1) - batched_inputs["p_d"].unsqueeze(1)

        z_diff_raw = z_diff_raw.permute(0, 3, 1, 2)  # [B, #planes, H, W]
        if fine_h != H or fine_w != W:
            z_diff_fine = torch.nn.functional.interpolate(z_diff_raw, (fine_h, fine_w), mode="nearest")

        z_diff_coarse = torch.nn.functional.interpolate(z_diff_raw, (coarse_h, coarse_w), mode="nearest")

        # directly set pixels before depth to zero.
        # [B, #planes, coarse_h, coarse_w]
        flag_after_mesh_coarse = (z_diff_coarse > 0).float()
        flag_after_mesh_fine = (z_diff_fine > 0).float()

        z_diff_fine = z_diff_fine.permute(0, 2, 3, 1)  # [B, H, W, #planes]

        z_diff_pe_fine = self.PE(z_diff_fine)  # [B, fine_H, fine_W, num_depth, num_pos_feats]
        z_diff_pe_fine = torch.permute(z_diff_pe_fine, (0, 3, 4, 1, 2))  # [B, num_depth, num_pos_feats, H, W]

        B = z_diff_pe_fine.shape[0]
        P = z_diff_pe_fine.shape[1]
        F = z_diff_pe_fine.shape[2]

        z_diff_pe_fine = z_diff_pe_fine.reshape((B * P, F, fine_h, fine_w))

        z_diff_feat_fine = self.z_diff_branch(z_diff_pe_fine)
        z_diff_feat_coarse = torch.nn.functional.interpolate(z_diff_feat_fine, (coarse_h, coarse_w), mode="nearest")

        z_diff_feat_fine = z_diff_feat_fine.view((B, P, self.feat_channels, fine_h, fine_w))
        z_diff_feat_coarse = z_diff_feat_coarse.view((B, P, self.feat_channels, coarse_h, coarse_w))

        z_diff_feat_fine_normalized = z_diff_feat_fine
        z_diff_feat_coarse_normalized = z_diff_feat_coarse

        # ------------------------------------------------------------------------------------------------------
        # Coarse reconstruction

        images = torch.permute(batched_inputs["im"], (0, 3, 1, 2))  # batched_inputs['im']: [b, h, w, 5]

        mask_features, multi_scale_features = self.backbone(images)  # mask_features: [B, C, new_h, new_w]

        # [B, C, coarse_h, coarse_w]
        img_feat_coarse = self.img_branch(mask_features)

        img_feat_coarse_expand = img_feat_coarse.unsqueeze(1)
        img_feat_coarse_expand = img_feat_coarse_expand.expand((-1, P, -1, -1, -1))

        # [B, #planes, H, W]
        pred_coarse = torch.sum(img_feat_coarse_expand * z_diff_feat_coarse_normalized, dim=2)

        # --------------------------------------------------------------------------------------------
        # Fine reconstruction

        img_feat_fine = torch.nn.functional.interpolate(
            img_feat_coarse, (fine_h, fine_w), mode="bilinear", align_corners=False
        )

        # [B, 1, C, H, W]
        img_feat_fine_expand = img_feat_fine.unsqueeze(1).expand(-1, P, -1, -1, -1)

        # # [B, #planes, C, H, W]
        # hadamard_prod_feat = img_feat_fine_expand * z_diff_feat_fine_normalized

        # [B, #planes, 3C, H, W]
        img_zdiff_feat = torch.cat((img_feat_fine_expand, z_diff_feat_fine_normalized), dim=2)
        # [B x #planes, 3C, H, W]
        img_zdiff_feat = img_zdiff_feat.view((B * P, 2 * self.feat_channels, fine_h, fine_w))

        # [B x #planes, 1, H, W]
        pred_fine = self.spatial_branch(img_zdiff_feat)
        pred_fine = pred_fine.view((B, P, fine_h, fine_w))

        pred_list = [
            [pred_coarse, flag_after_mesh_coarse],
            [pred_fine, flag_after_mesh_fine],
        ]

        return pred_list


@registry.register_model(name="v_coarse2fine_1x1conv")
class ModelCoarseToFine1x1Conv(torch.nn.Module):
    def __init__(self, backbone_name, pos_enc_kwargs={}, feat_channels=256, **kwargs):
        super().__init__()

        backbone_cls = registry.get_model(backbone_name)
        assert backbone_cls is not None, f"Cannot find backbone {backbone_cls}."
        self.backbone = backbone_cls(withPixelDecoder=True)

        self.PE = PositionEmbeddingSimple(**pos_enc_kwargs)

        self.feat_channels = feat_channels

        self.z_diff_branch = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 1, padding=0, bias=False),
            torch.nn.GroupNorm(4, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, self.feat_channels, 1, padding=0, bias=False),
            torch.nn.GroupNorm(4, self.feat_channels),
            torch.nn.ReLU(inplace=True),
        )

        self.img_branch = torch.nn.Sequential(
            torch.nn.Conv2d(256, self.feat_channels, 3, padding=1, bias=False),
            torch.nn.GroupNorm(4, self.feat_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(self.feat_channels, self.feat_channels, 3, padding=1, bias=False),
            torch.nn.GroupNorm(4, self.feat_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(self.feat_channels, self.feat_channels, 1, padding=0, bias=False),
        )

        self.spatial_branch = torch.nn.Sequential(
            torch.nn.Conv2d(2 * self.feat_channels, self.feat_channels, 1, padding=0, bias=False),
            torch.nn.GroupNorm(4, self.feat_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(self.feat_channels, self.feat_channels, 1, padding=0, bias=False),
            torch.nn.GroupNorm(4, self.feat_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(self.feat_channels, 1, 1, padding=0, bias=False),
        )

        self.fine_down_factor = 2
        self.coarse_down_factor = 4
        self.fine2coarse_factor = self.coarse_down_factor / self.fine_down_factor

    def forward(self, batched_inputs):
        # batched_inputs:
        # - im:[B, H, W, 5]
        # - annot: [B, H, W, num_depth]
        # - vis: [B, H, W]
        # - d_im: [B, H, W]
        # - p_d: [B, 1, num_depth]

        H = batched_inputs["d_im"].shape[1]
        W = batched_inputs["d_im"].shape[2]

        fine_h = H // self.fine_down_factor
        fine_w = W // self.fine_down_factor
        coarse_h = H // self.coarse_down_factor
        coarse_w = W // self.coarse_down_factor

        # -----------------------------------------------------------------------------------------------------------------
        # process depth

        z_diff_raw = batched_inputs["d_im"].unsqueeze(-1) - batched_inputs["p_d"].unsqueeze(1)  # [B, H, W, #planes]

        z_diff_raw = z_diff_raw.permute(0, 3, 1, 2)  # [B, #planes, H, W]
        if fine_h != H or fine_w != W:
            z_diff_fine = torch.nn.functional.interpolate(z_diff_raw, (fine_h, fine_w), mode="nearest")

        z_diff_coarse = torch.nn.functional.interpolate(z_diff_raw, (coarse_h, coarse_w), mode="nearest")

        # directly set pixels before depth to zero.
        # [B, #planes, coarse_h, coarse_w]
        flag_after_mesh_coarse = (z_diff_coarse > 0).float()
        flag_after_mesh_fine = (z_diff_fine > 0).float()

        z_diff_fine = z_diff_fine.permute(0, 2, 3, 1)  # [B, H, W, #planes]

        z_diff_pe_fine = self.PE(z_diff_fine)  # [B, fine_H, fine_W, num_depth, num_pos_feats]
        z_diff_pe_fine = torch.permute(z_diff_pe_fine, (0, 3, 4, 1, 2))  # [B, num_depth, num_pos_feats, H, W]

        B = z_diff_pe_fine.shape[0]
        P = z_diff_pe_fine.shape[1]
        F = z_diff_pe_fine.shape[2]

        z_diff_pe_fine = z_diff_pe_fine.reshape((B * P, F, fine_h, fine_w))

        z_diff_feat_fine = self.z_diff_branch(z_diff_pe_fine)
        z_diff_feat_coarse = torch.nn.functional.interpolate(z_diff_feat_fine, (coarse_h, coarse_w), mode="nearest")

        z_diff_feat_fine = z_diff_feat_fine.view((B, P, self.feat_channels, fine_h, fine_w))
        z_diff_feat_coarse = z_diff_feat_coarse.view((B, P, self.feat_channels, coarse_h, coarse_w))

        z_diff_feat_fine_normalized = z_diff_feat_fine
        z_diff_feat_coarse_normalized = z_diff_feat_coarse

        # ------------------------------------------------------------------------------------------------------
        # Coarse reconstruction

        images = torch.permute(batched_inputs["im"], (0, 3, 1, 2))  # batched_inputs['im']: [b, h, w, 5]

        mask_features, multi_scale_features = self.backbone(images)  # mask_features: [B, C, new_h, new_w]

        # [B, C, coarse_h, coarse_w]
        img_feat_coarse = self.img_branch(mask_features)

        img_feat_coarse_expand = img_feat_coarse.unsqueeze(1)
        img_feat_coarse_expand = img_feat_coarse_expand.expand((-1, P, -1, -1, -1))

        # [B, #planes, H, W]
        pred_coarse = torch.sum(img_feat_coarse_expand * z_diff_feat_coarse_normalized, dim=2)

        # --------------------------------------------------------------------------------------------
        # Fine reconstruction

        img_feat_fine = torch.nn.functional.interpolate(
            img_feat_coarse, (fine_h, fine_w), mode="bilinear", align_corners=False
        )

        # [B, 1, C, H, W]
        img_feat_fine_expand = img_feat_fine.unsqueeze(1).expand(-1, P, -1, -1, -1)

        # [B, #planes, 3C, H, W]
        img_zdiff_feat = torch.cat((img_feat_fine_expand, z_diff_feat_fine_normalized), dim=2)
        # [B x #planes, 3C, H, W]
        img_zdiff_feat = img_zdiff_feat.view((B * P, 2 * self.feat_channels, fine_h, fine_w))

        # [B x #planes, 1, H, W]
        pred_fine = self.spatial_branch(img_zdiff_feat)
        pred_fine = pred_fine.view((B, P, fine_h, fine_w))

        pred_list = [
            [pred_coarse, flag_after_mesh_coarse],
            [pred_fine, flag_after_mesh_fine],
        ]

        return pred_list
