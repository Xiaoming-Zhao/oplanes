import os
import sys
import traceback
import skimage
import numpy as np
import matplotlib.pyplot as plt
import meshplot as mp
from tqdm import tqdm
from PIL import Image
from scipy.ndimage.morphology import binary_erosion, distance_transform_edt
from scipy.ndimage import maximum_filter, gaussian_filter
from scipy import ndimage
from skimage import filters, transform

import torch
from torch.utils.data import Dataset

from oplanes.utils.img_utils import crop_vis_map


class DataObjNonS3D:
    def __init__(self, im_f, depth_f, mask_f, cam_mat_f):

        im = np.array(Image.open(im_f))
        depth = np.load(depth_f)["depth"]
        mask = np.array(Image.open(mask_f)) > 0
        cam_mat_dict = np.load(cam_mat_f)

        self.im = im
        self.depth = depth
        self.mask = mask
        self.intri_K = cam_mat_dict["intri"]  # from camera to image
        self.extri_mat = cam_mat_dict["extri"]  # from world to camera
        self.nons3d_to_s3d_coord_sys = cam_mat_dict["nons3d_to_s3d"]
        self.inv_K = np.linalg.inv(self.intri_K)
        self.inv_extri_mat = np.linalg.inv(self.extri_mat)  # from camera to world

        self.float_type = np.float32
        self.avoid_nan_eps = 1e-8

    def detph2pcl(self):
        # X for horizontal/cols, Y for vertical/rows
        # tmp_mask = binary_erosion(self.mask, np.ones((10, 10)))
        tmp_mask = maximum_filter(self.mask, footprint=np.ones((5, 5)))
        tmp_py_orig, tmp_px_orig = np.nonzero(tmp_mask)
        tmp_z = self.depth[tmp_py_orig, tmp_px_orig]

        tmp_homo_coord = np.stack([tmp_px_orig, tmp_py_orig, np.ones_like(tmp_py_orig)], axis=1)  # [#points, 3]
        tmp_camera_coord = np.matmul(self.inv_K[:3, :3], tmp_homo_coord.T)  # [3, #points]
        tmp_camera_coord = tmp_camera_coord.T  # [#points, 3]
        tmp_camera_coord = tmp_camera_coord[:, 0:3] / tmp_camera_coord[:, -1:]  # [#pixels, 3]
        tmp_camera_coord = tmp_camera_coord * tmp_z[..., None]

        tmp_camera_coord_in_s3d = np.matmul(self.nons3d_to_s3d_coord_sys, tmp_camera_coord.T).T

        return tmp_camera_coord

    def pcl2img(self, pcl):
        homo_pix_coord = np.matmul(self.intri_K[:3, :3], pcl[:, :3].T).T  # [#points, 3]
        pix_coord = homo_pix_coord[:, :2] / homo_pix_coord[:, 2:]

        pix_x = pix_coord[:, 0]
        pix_y = pix_coord[:, 1]

        return pix_x, pix_y

    def ndcs_to_pixels(self, x, y, size):
        s_y, s_x = size
        s_x -= 1
        s_y -= 1
        xx = np.float32(x + 1) * np.float32(s_x / 2)
        yy = np.float32(1 - y) * np.float32(s_y / 2)
        return xx, yy

    def pixels_to_ndcs(self, xx, yy, size):
        s_y, s_x = size
        s_x -= 1  # so 1 is being mapped into (n-1)th pixel
        s_y -= 1  # so 1 is being mapped into (n-1)th pixel
        x = np.float32(2 / s_x) * np.float32(xx) - 1
        y = np.float32(-2 / s_y) * np.float32(yy) + 1
        return x, y

    def vis_to_cam_coord(self, vis, depth):
        py_orig, px_orig = np.nonzero(vis)
        vis_z = depth[py_orig, px_orig]

        homo_coord = np.stack([px_orig, py_orig, np.ones_like(py_orig)], axis=1)  # [#points, 3]
        camera_coord = np.matmul(self.inv_K[:3, :3], homo_coord.T)  # [3, #points]
        camera_coord = camera_coord.T  # [#points, 3]
        camera_coord = camera_coord[:, 0:3] / camera_coord[:, -1:]  # [#pixels, 3]
        camera_coord = camera_coord * vis_z[..., None]

        camera_coord_in_s3d = np.matmul(self.nons3d_to_s3d_coord_sys, camera_coord.T).T

        return camera_coord, camera_coord_in_s3d, px_orig, py_orig

    def load_data(
        self,
        *,
        n_planes,
        is_val,
        pixel_mean,
        pixel_std,
        target_h=512,
        target_w=512,
        crop_expand_ratio=0.1,
        given_depth_range=2.0,
        for_debug=False,
        use_masked_out_img=False,
        given_crop_info=None,
    ):

        assert is_val

        im = np.copy(self.im)
        depth = np.copy(self.depth)
        vis_orig = np.copy(self.mask)

        cur_depth_range = given_depth_range

        # NOTE: during inference, we heavily rely on mask to give correct closest_depth value.
        # Therefore, we need to be somehow conservative.
        vis_for_closest_Z = binary_erosion(vis_orig, np.ones((10, 10)))

        tmp_camera_coord, tmp_camera_coord_in_s3d, _, _ = self.vis_to_cam_coord(vis_for_closest_Z, depth)

        cur_closest_vis_depth = np.max(tmp_camera_coord_in_s3d[:, 2])

        # NOTE: mask may not be sharp enough, we shrink it a little bit.
        vis = maximum_filter(vis_orig, footprint=np.ones((5, 5)))

        try:
            (
                start_row,
                end_row,
                start_col,
                end_col,
                min_row,
                max_row,
                min_col,
                max_col,
            ) = crop_vis_map(torch.from_numpy(vis_orig), expand_ratio=crop_expand_ratio)
        except:
            print("\n", self.fn, "\n")
            traceback.print_exc()
            err = sys.exc_info()[0]
            print(err)
            sys.exit(1)

        crop_info = given_crop_info
        assert (
            end_row - start_row == end_col - start_col
        ), f"\nCurrently only support squared crop. However, we get {start_row}, {end_row}, {start_col}, {end_col}.\n"
        crop_info = {
            "start_row": start_row,
            "end_row": end_row,
            "start_col": start_col,
            "end_col": end_col,
        }

        resize_info = {
            "row_resize_factor": target_h / (end_row - start_row),
            "col_resize_factor": target_w / (end_col - start_col),
            "rows_before_resize": end_row - start_row,
            "rows_after_resize": target_h,
            "cols_before_resize": end_col - start_col,
            "cols_after_resize": target_w,
        }

        # nearest-neighbour
        tgt_vis_orig = transform.resize(
            vis_orig[start_row:end_row, start_col:end_col],
            (target_h, target_w),
            order=0,
        ).astype(self.float_type)
        tgt_vis = transform.resize(vis[start_row:end_row, start_col:end_col], (target_h, target_w), order=0).astype(
            self.float_type
        )
        # bicubic
        tgt_depth = transform.resize(depth[start_row:end_row, start_col:end_col], (target_h, target_w), order=3)
        # Lanczos
        tgt_im = np.array(
            Image.fromarray(im[start_row:end_row, start_col:end_col, :]).resize(
                (target_w, target_h), resample=Image.LANCZOS
            )
        )

        dist = distance_transform_edt(tgt_vis).astype(self.float_type) / tgt_vis.shape[1] * 10.0 - 5  # range [-5, 5]
        dedge = filters.farid(tgt_depth).astype(self.float_type)
        for k in range(tgt_im.shape[2]):
            dedge += filters.farid(tgt_im[:, :, k]).astype(self.float_type)  # Add edges detected from RGB
        dedge /= 1 + tgt_im.shape[2]

        normalized_tgt_im = (tgt_im.astype(self.float_type) - pixel_mean.astype(self.float_type)) / (
            pixel_std.astype(self.float_type) + self.avoid_nan_eps
        )
        im_out = np.concatenate(
            (normalized_tgt_im, dist[..., np.newaxis], dedge[..., np.newaxis]), axis=2
        )  # [new_h, new_w, 5]

        maxZ = cur_closest_vis_depth

        aug = np.reshape(np.linspace(maxZ, maxZ - cur_depth_range, n_planes), (1, n_planes)).astype(self.float_type)

        aug = np.sort(aug)[:, ::-1]  # [1, num_depth], 1st plane is for the plane closest to the camera

        # [1, #planes]
        plane_diff_per_plane = maxZ - aug

        camera_coord, camera_coord_in_s3d, px_orig, py_orig = self.vis_to_cam_coord(vis, depth)

        # [H, W]
        # d_im = (maxZ - cur_depth_range) * np.ones(vis.shape, dtype=self.float_type)   # init with furthest depth
        d_im = -9999 * np.ones(vis.shape, dtype=self.float_type)  # init with depth really far
        d_im[py_orig, px_orig] = camera_coord_in_s3d[:, 2]

        tgt_d_im = transform.resize(d_im[start_row:end_row, start_col:end_col], (target_h, target_w), order=0).astype(
            self.float_type
        )

        if use_masked_out_img:
            im_out = im_out * tgt_vis[..., np.newaxis]
            tgt_im = tgt_im * tgt_vis[..., np.newaxis]
            tgt_depth = tgt_depth * tgt_vis

        ret_dict = {
            # after crop and resize
            "raw_im": torch.from_numpy(tgt_im),
            "im": torch.from_numpy(im_out),
            "vis_orig": torch.from_numpy(tgt_vis_orig),
            "vis": torch.from_numpy(tgt_vis),
            "d_im": torch.from_numpy(tgt_d_im),
            "depth": torch.from_numpy(tgt_depth),
            # plane information
            "p_d": torch.from_numpy(aug.copy()),
            "plane_diff_per_plane": torch.from_numpy(plane_diff_per_plane),
            "crop_info": crop_info,
            "resize_info": resize_info,
        }
        if for_debug:
            ret_dict.update(
                {
                    # before crop and resize
                    "raw_vis_orig": torch.from_numpy(vis_orig),
                    "raw_vis": torch.from_numpy(vis),
                    "raw_d_im": torch.from_numpy(d_im),
                    "raw_depth": torch.from_numpy(depth),
                }
            )

        return ret_dict
