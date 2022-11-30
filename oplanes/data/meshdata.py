import os
import sys
import gc
import glob
import igl
import multiprocessing
import trimesh
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

from oplanes.data import binvox_rw
from oplanes.utils.img_utils import crop_vis_map


class MeshObj:
    def __init__(
        self,
        fn,
        fnroot="",
        binvox_path_prefix="",
        pure_infer=False,
    ):
        self.float_type = np.float32
        arr = fn.split("/")
        self.csName = arr[-4]
        self.objID = int(arr[-3].split("_")[0])
        self.frameID = arr[-2].split("_")[0]
        self.IMAGE_SIZE = (800, 1280)
        self.fnroot = fnroot
        self.fn = self.fnroot + fn[1:]

        self.imfn = self.fnroot + "/{}/images/{}.bmp".format(self.csName, self.frameID)

        self.depthfn = self.fnroot + "/{}/depth/{}.npy".format(self.csName, self.frameID)
        self.visfn = self.fnroot + "/{}/visible/{}.npy".format(self.csName, self.frameID)

        self.pure_infer = pure_infer
        if not self.pure_infer:
            self.binvox = binvox_path_prefix + self.get_obj_fn()[1:-8] + "voxel_256.binvox2"
            assert os.path.exists(self.binvox), f"{self.binvox}"

        self.load_rot_mats()

        self.avoid_nan_eps = 1e-8

    def compute_visible_mesh_depth_range(self):
        xx, yy, xxi, yyi, select, obj = self.proj_obj_to_img()
        vis_verts = np.array(obj.vertices)[select, :]
        min_coords = np.min(vis_verts, axis=0)
        max_coords = np.max(vis_verts, axis=0)
        depth_range = max_coords[2] - min_coords[2]
        return depth_range, max_coords[2]

    def load_rot_mats(self):

        rmatrices_file = sorted(glob.glob(self.fn[:-8] + "draw_*/rage_matrices_bin.csv"))[0]

        rage_matrices = np.fromfile(rmatrices_file, dtype=np.float32).astype(self.float_type)
        rage_matrices = rage_matrices.reshape((4, 4, 4))
        self.VP = np.dot(np.linalg.inv(rage_matrices[0, :, :]), rage_matrices[2, :, :])
        self.VP_inverse = np.linalg.inv(self.VP)  # multiply this matrix to convert from NDC to world coordinate
        self.P = np.dot(np.linalg.inv(rage_matrices[1, :, :]), rage_matrices[2, :, :])
        self.P_inverse = np.linalg.inv(self.P)  # multiply this matrix to convert from NDC to camera coordinate

    def get_obj_fn(self):
        return "." + self.fn[len(self.fnroot) :]

    def get_mesh(self):
        return trimesh.load(self.fn)

    def load_binvox(self):
        with open(self.binvox, "rb") as f:
            return binvox_rw.read_as_3d_array(f)

    def ndcs_to_pixels(self, x, y):
        s_y, s_x = self.IMAGE_SIZE
        s_x -= 1
        s_y -= 1
        xx = self.float_type(x + 1) * self.float_type(s_x / 2)
        yy = self.float_type(1 - y) * self.float_type(s_y / 2)
        return xx, yy

    def pixels_to_ndcs(self, xx, yy):
        s_y, s_x = self.IMAGE_SIZE
        s_x -= 1  # so 1 is being mapped into (n-1)th pixel
        s_y -= 1  # so 1 is being mapped into (n-1)th pixel
        x = self.float_type(2 / s_x) * self.float_type(xx) - 1
        y = self.float_type(-2 / s_y) * self.float_type(yy) + 1
        return x, y

    def proj_obj_to_img(self):
        obj = trimesh.load(self.fn)
        ndcpts = np.concatenate([obj.vertices, np.ones((obj.vertices.shape[0], 1))], axis=1) @ self.P
        ndcpts = ndcpts[:, 0:2] / ndcpts[:, -1:]
        xx, yy = self.ndcs_to_pixels(ndcpts[:, 0], ndcpts[:, 1])
        xxi = np.rint(xx).astype(int)
        yyi = np.rint(yy).astype(int)
        select = np.logical_and(
            np.logical_and(xxi > 0, xxi < self.IMAGE_SIZE[1]),
            np.logical_and(yyi > 0, yyi < self.IMAGE_SIZE[0]),
        )
        return xx, yy, xxi, yyi, select, obj

    def compute_visibility(self):
        _, _, xxi, yyi, select, _ = self.proj_obj_to_img()
        vis = np.load(self.visfn) == self.objID
        vis = binary_erosion(vis, np.ones((10, 10)))
        xxi = xxi[select]
        yyi = yyi[select]
        select2 = vis[yyi, xxi] == 1
        return select2.sum() / select.shape[0]

    def __repr__(self):
        return "MeshObj(\n  {}\n  {};  {};  {})".format(self.fn, self.csName, self.objID, self.frameID)

    def __str__(self):
        return self.__repr__()

    def get_vis_coords(self):
        depth = np.load(self.depthfn) / 6.0 - 4e-5
        vis_orig = np.load(self.visfn) == self.objID
        # vis = binary_erosion(vis_orig, np.ones((3,3)))
        vis = vis_orig
        resize_factor = 2
        new_size_rc = (
            self.IMAGE_SIZE[0] // resize_factor,
            self.IMAGE_SIZE[1] // resize_factor,
        )
        vis_resize = transform.resize(vis, new_size_rc, order=0).astype(self.float_type)
        py_orig, px_orig = np.nonzero(vis_resize)
        px = px_orig * resize_factor
        py = py_orig * resize_factor
        ndcx, ndcy = self.pixels_to_ndcs(px, py)
        ndcz = depth[py, px]

        ndc_coord = np.stack([ndcx, ndcy, ndcz, np.ones_like(ndcz)], axis=1)  # NDC
        camera_coord = ndc_coord @ self.P_inverse  # convert to camera coordinate
        camera_coord = camera_coord[:, 0:3] / camera_coord[:, -1:]  # divide

        px_orig = (px_orig / (vis_resize.shape[1] / 2) - 1).astype(self.float_type)
        py_orig = (py_orig / (vis_resize.shape[0] / 2) - 1).astype(self.float_type)
        return (camera_coord, px_orig, py_orig)

    def load_data(
        self,
        *,
        num_depth,
        is_val,
        pixel_mean,
        pixel_std,
        target_h=512,
        target_w=512,
        crop_expand_ratio=0.1,
        given_depth_range=None,
        for_debug=False,
        depth_range_expand_ratio=0.1,
        use_masked_out_img=False,
        given_crop_info=None,
        pure_infer=False,
    ):

        im = np.array(Image.open(self.imfn))
        depth = np.load(self.depthfn).astype(self.float_type) / 6.0 - 4e-5
        vis_orig = np.load(self.visfn) == self.objID

        if given_depth_range is not None:
            cur_depth_range = given_depth_range
            if pure_infer:
                # NOTE: during inference, we heavily rely on mask to give correct closest_depth value.
                # Therefore, we need to be somehow conservative.
                vis_for_max_Z = binary_erosion(vis_orig, np.ones((10, 10)))
                tmp_py_orig, tmp_px_orig = np.nonzero(vis_for_max_Z)
                tmp_px = tmp_px_orig
                tmp_py = tmp_py_orig
                tmp_ndcx, tmp_ndcy = self.pixels_to_ndcs(tmp_px, tmp_py)
                tmp_ndcz = depth[tmp_py, tmp_px]

                tmp_ndc_coord = np.stack([tmp_ndcx, tmp_ndcy, tmp_ndcz, np.ones_like(tmp_ndcz)], axis=1)  # NDC
                tmp_camera_coord = tmp_ndc_coord @ self.P_inverse  # convert to camera coordinate, [#pixels, 3]
                tmp_camera_coord = tmp_camera_coord[:, 0:3] / tmp_camera_coord[:, -1:]  # divide, [#pixels, 3]
                cur_closest_vis_depth = np.max(tmp_camera_coord[:, 2])  # depth in of negative value
            else:
                raise NotImplementedError
        else:
            (
                cur_depth_range,
                cur_closest_vis_depth,
            ) = self.compute_visible_mesh_depth_range()
            cur_depth_range = cur_depth_range * (1 + depth_range_expand_ratio)

        if pure_infer:
            # NOTE: during inference, we use the raw mask to keep object sharp.
            vis = vis_orig
        else:
            # During training, we use some out-of-boundary pixels to give some negative examples
            vis = maximum_filter(vis_orig, footprint=np.ones((5, 5)))

        if given_crop_info is None:
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
        else:
            start_row = given_crop_info["start_row"]
            end_row = given_crop_info["end_row"]
            start_col = given_crop_info["start_col"]
            end_col = given_crop_info["end_col"]

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
        # nearest-neighbour
        tgt_depth = transform.resize(depth[start_row:end_row, start_col:end_col], (target_h, target_w), order=0)
        # Lanczos
        tgt_im = np.array(
            Image.fromarray(im[start_row:end_row, start_col:end_col, :]).resize(
                (target_w, target_h), resample=Image.LANCZOS
            )
        )

        # edge detector
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

        py_orig, px_orig = np.nonzero(vis)
        px = px_orig
        py = py_orig
        ndcx, ndcy = self.pixels_to_ndcs(px, py)
        ndcz = depth[py, px]

        ndc_coord = np.stack([ndcx, ndcy, ndcz, np.ones_like(ndcz)], axis=1)  # NDC
        camera_coord = ndc_coord @ self.P_inverse  # convert to camera coordinate, [#pixels, 3]
        camera_coord = camera_coord[:, 0:3] / camera_coord[:, -1:]  # divide, [#pixels, 3]

        if cur_closest_vis_depth is None:
            # neg-Z is for forward. Therefore, maxZ is the closest depth.
            maxZ = np.max(camera_coord[:, 2])
        else:
            maxZ = cur_closest_vis_depth

        if is_val:
            aug = np.reshape(np.linspace(maxZ, maxZ - cur_depth_range, num_depth), (1, num_depth)).astype(
                self.float_type
            )
        else:
            aug = maxZ - cur_depth_range * np.random.rand(1, num_depth).astype(self.float_type)

        aug = np.sort(aug)[:, ::-1]  # [1, num_depth], 1st plane is for the plane closest to the camera

        # [1, #planes]
        plane_diff_per_plane = maxZ - aug

        # [H, W]
        # d_im = (maxZ - cur_depth_range) * np.ones(vis.shape, dtype=self.float_type)   # init with furthest depth
        d_im = -9999 * np.ones(vis.shape, dtype=self.float_type)  # init with depth that is really far
        d_im[py_orig, px_orig] = camera_coord[:, 2]

        tgt_d_im = transform.resize(d_im[start_row:end_row, start_col:end_col], (target_h, target_w), order=0).astype(
            self.float_type
        )

        if (not pure_infer) or (pure_infer and for_debug):
            with open(self.binvox, "rb") as f:
                m1 = binvox_rw.read_as_3d_array(f)

            coords = np.tile(camera_coord[..., np.newaxis], (1, 1, num_depth))  # [#pixels, 3, num_depth]

            # Path along the ray
            coords = (
                coords * aug[:, np.newaxis, :] / (camera_coord[:, 2:, np.newaxis] + self.avoid_nan_eps)
            )  # [#p, 3, #d], aug: [1, 1, #d], [#p, 1, 1], [#p, 1, #d]

            coords = np.swapaxes(coords, 1, 2)  # [#pixels, num_depth, 3]
            coords = np.reshape(coords, (-1, 3))  # [#pixels x num_depth, 3]

            grid_coords = np.round((coords - m1.translate) / m1.scale * m1.dims - 0.5).astype(int)
            label = np.zeros((coords.shape[0],), dtype=bool)
            select = np.logical_and(np.all(grid_coords >= 0, axis=1), np.all(grid_coords < m1.dims, axis=1))
            label[select] = m1.data[grid_coords[select, 0], grid_coords[select, 1], grid_coords[select, 2]]

            gt_im = np.zeros(vis.shape + (num_depth,), dtype=bool)  # [new_h, new_w, num_depth]
            for ii in range(num_depth):
                gt_im[py_orig, px_orig, ii] = label[ii::num_depth]

            # TODOs: nearest-neighbour?
            tgt_gt_im = transform.resize(
                gt_im[start_row:end_row, start_col:end_col, :],
                (target_h, target_w),
                order=0,
            ).astype(self.float_type)

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
        if not self.pure_infer:
            ret_dict["annot"] = torch.from_numpy(tgt_gt_im)
        if for_debug:
            ret_dict.update(
                {
                    # before crop and resize
                    "raw_annot": torch.from_numpy(gt_im),
                    "raw_vis_orig": torch.from_numpy(vis_orig),
                    "raw_vis": torch.from_numpy(vis),
                    "raw_d_im": torch.from_numpy(d_im),
                    "raw_depth": torch.from_numpy(depth),
                }
            )

        return ret_dict


class MyMeshObj(MeshObj):
    def __init__(self, fn, fnroot="", binvox=""):
        self.float_type = np.float32
        arr = fn.split("/")
        self.csName = arr[-4]
        self.objID = int(arr[-3].split("_")[0])
        self.frameID = arr[-2].split("_")[0]
        self.IMAGE_SIZE = (800, 1280)
        self.fnroot = fnroot
        self.fn = self.fnroot + fn[1:]
        self.imfn = self.fnroot + "/{}/images/{}.bmp".format(self.csName, self.frameID)
        self.depthfn = self.fnroot + "/{}/depth/{}.npy".format(self.csName, self.frameID)
        self.visfn = self.fnroot + "/{}/visible/{}.npy".format(self.csName, self.frameID)
        # self.binvox = binvox_path_prefix + self.get_obj_fn()[1:-8] + 'voxel.binvox'
        self.binvox = binvox
        if not os.path.exists(self.binvox):
            self.binvox = f"{self.binvox}2"
        assert os.path.exists(self.binvox), f"{self.binvox}"
        # self.depth_range = depth_range
        self.load_rot_mats()


class MyData(Dataset):
    def __init__(
        self,
        *,
        fn="",
        threshold=0.0,
        is_val=False,
        pixel_mean=[123.675, 116.280, 103.530],
        pixel_std=[58.395, 57.120, 57.375],
        max_depth_range=2.0,
        n_planes_for_train=5,
        n_planes_for_val=20,
        mesh_data_root="",
        binvox_path_prefix="",
        selector=None,
        data_h=512,
        data_w=512,
        crop_expand_ratio=0.1,
        depth_range_expand_ratio=0.1,
        use_masked_out_img=False,
    ):
        self.mesh_data_root = mesh_data_root
        self.binvox_path_prefix = binvox_path_prefix
        self.depth_range_expand_ratio = depth_range_expand_ratio
        self.max_depth_range = max_depth_range * (1 + depth_range_expand_ratio)
        with open(fn) as f:
            lines = f.read().splitlines()
        np.random.shuffle(lines)
        if selector is not None:
            lines = lines[selector]
        if lines[0].find(",") != -1:
            lines = [parts[0] for line in lines if float((parts := line.split(","))[1]) > threshold]
        procnum = min(multiprocessing.cpu_count() // 2, len(lines))
        pool = multiprocessing.Pool(processes=procnum)
        self.objIdentifiers = pool.map(self.init_helper, lines, chunksize=100)
        pool.close()
        pool.join()
        self.is_val = is_val
        if self.is_val:
            self.n_sample_planes = n_planes_for_val
        else:
            self.n_sample_planes = n_planes_for_train

        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std

        self.data_h = data_h
        self.data_w = data_w
        self.crop_expand_ratio = crop_expand_ratio

        self.use_masked_out_img = use_masked_out_img

    def init_helper(self, line):
        return MeshObj(
            line,
            self.mesh_data_root,
            self.binvox_path_prefix,
            # self.extra_mesh_data_root,
        )

    def __len__(self):
        return len(self.objIdentifiers)

    def __getitem__(self, idx):
        loaded_data = self.objIdentifiers[idx].load_data(
            num_depth=self.n_sample_planes,
            is_val=self.is_val,
            pixel_mean=self.pixel_mean,
            pixel_std=self.pixel_std,
            target_h=self.data_h,
            target_w=self.data_w,
            crop_expand_ratio=self.crop_expand_ratio,
            depth_range_expand_ratio=self.depth_range_expand_ratio,
            use_masked_out_img=self.use_masked_out_img,
        )
        return {**loaded_data, **{"idx": str(idx), "fn": self.objIdentifiers[idx].fn}}
