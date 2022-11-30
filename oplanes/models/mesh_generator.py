import os
import glob
import mcubes
import trimesh
import itertools
import numpy as np
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage import maximum_filter, gaussian_filter
from skimage import transform

import torch

from oplanes.data import binvox_rw


class MeshGenerator:
    def __init__(self):
        self.float_type = np.float32

    def ndcs_to_pixels(self, x, y, w, h):
        # s_y, s_x = self.IMAGE_SIZE
        # s_x -= 1
        # s_y -= 1
        w -= 1
        h -= 1
        xx = self.float_type(x + 1) * self.float_type(w / 2)
        yy = self.float_type(1 - y) * self.float_type(h / 2)
        return xx, yy

    def pixels_to_ndcs(self, xx, yy, w, h):
        # s_y, s_x = self.IMAGE_SIZE
        # s_x -= 1  # so 1 is being mapped into (n-1)th pixel
        # s_y -= 1  # so 1 is being mapped into (n-1)th pixel
        w -= 1  # so 1 is being mapped into (n-1)th pixel
        h -= 1  #
        x = self.float_type(2 / w) * self.float_type(xx) - 1
        y = self.float_type(-2 / h) * self.float_type(yy) + 1
        return x, y

    def get_orig_pix_coords_from_transformed_img(self, rows, cols, orig_to_transformed_order, crop_info, resize_info):
        # NOTE: we always assume when raw -> transformed image:
        # first crop then resize.
        # Therefore, to transform back, we need to first resize, then crop back.

        if orig_to_transformed_order == "none":
            return rows, cols
        elif orig_to_transformed_order == "crop_resize":

            if resize_info is not None:
                # resize back
                unresize_rows = rows / resize_info["row_resize_factor"]
                unresize_cols = cols / resize_info["col_resize_factor"]
                # unresize_rows = resize_info["rows_before_resize"]
                # unresize_cols = resize_info["cols_before_resize"]
            else:
                unresize_rows = rows
                unresize_cols = cols

            if crop_info is not None:
                # un-crop
                orig_rows = unresize_rows + crop_info["start_row"]
                orig_cols = unresize_cols + crop_info["start_col"]
            else:
                orig_rows = unresize_rows
                orig_cols = unresize_cols

            # print("transform->raw: ", rows[:5], unresize_rows[:5], orig_rows[:5])

            return orig_rows, orig_cols
        else:
            raise ValueError(f"We always assume when orig -> transformed image: first crop then resize.\n")

    def get_transformed_pix_coords_from_orig_img(
        self, orig_rows, orig_cols, orig_to_transformed_order, crop_info, resize_info
    ):
        # NOTE: we always assume when raw -> transformed image:
        # first crop then resize.
        # Therefore, to transform back, we need to first resize, then crop back.

        if orig_to_transformed_order == "none":
            return orig_rows, orig_cols
        elif orig_to_transformed_order == "crop_resize":

            if crop_info is not None:
                # crop
                cropped_rows = orig_rows - crop_info["start_row"]
                cropped_cols = orig_cols - crop_info["start_col"]
            else:
                cropped_rows = orig_rows
                cropped_cols = orig_cols

            if resize_info is not None:
                # resize
                rows = cropped_rows * resize_info["row_resize_factor"]
                cols = cropped_cols * resize_info["col_resize_factor"]
                # rows = resize_info["rows_after_resize"]
                # cols = resize_info["cols_after_resize"]
            else:
                rows = cropped_rows
                cols = cropped_cols

            # print("raw->transform: ", orig_rows[:5], cropped_rows[:5], rows[:5])

            return rows, cols
        else:
            raise ValueError(f"We always assume when orig -> transformed image: first crop then resize.\n")

    def get_visible_pix_cam_coords_s3d(
        self,
        *,
        orig_h,
        orig_w,
        vis,
        depth,
        orig_to_transformed_order,
        crop_info,
        resize_info,
        cam_matrices,
    ):
        # Row for py, cols for px
        py, px = np.nonzero(vis)

        rows, cols = py, px

        py_orig, px_orig = self.get_orig_pix_coords_from_transformed_img(
            rows, cols, orig_to_transformed_order, crop_info, resize_info
        )

        ndcx, ndcy = self.pixels_to_ndcs(px_orig, py_orig, orig_w, orig_h)
        ndcz = depth[py, px]

        ndc_coord = np.stack([ndcx, ndcy, ndcz, np.ones_like(ndcz)], axis=1)  # NDC
        camera_coords = ndc_coord @ cam_matrices["ndc_to_cam"]  # convert to camera coordinate
        camera_coords = camera_coords[:, 0:3] / camera_coords[:, -1:]  # divide

        px_normalized = (px / (vis.shape[1] / 2) - 1).astype(self.float_type)
        py_normalized = (py / (vis.shape[0] / 2) - 1).astype(self.float_type)
        return (camera_coords, px, py, px_normalized, py_normalized)

    def get_visible_pix_cam_coords_nons3d(
        self,
        *,
        orig_h,
        orig_w,
        vis,
        depth,
        orig_to_transformed_order,
        crop_info,
        resize_info,
        cam_matrices,
    ):
        # Row for py, cols for px
        py, px = np.nonzero(vis)

        rows, cols = py, px

        py_orig, px_orig = self.get_orig_pix_coords_from_transformed_img(
            rows, cols, orig_to_transformed_order, crop_info, resize_info
        )

        vis_z = depth[py, px]

        homo_coord = np.stack([px_orig, py_orig, np.ones_like(py_orig)], axis=1)  # [#points, 3]
        camera_coord = np.matmul(cam_matrices["inv_K"][:3, :3], homo_coord.T).T  # [#points, 3]
        camera_coord = camera_coord[:, 0:3] / camera_coord[:, -1:]  # [#pixels, 3]
        camera_coord = camera_coord * vis_z[..., None]

        camera_coord_in_s3d = np.matmul(cam_matrices["nons3d_to_s3d"], camera_coord.T).T

        px_normalized = (px / (vis.shape[1] / 2) - 1).astype(self.float_type)
        py_normalized = (py / (vis.shape[0] / 2) - 1).astype(self.float_type)
        return (camera_coord_in_s3d, px, py, px_normalized, py_normalized)

    def compute_binvox_to_cam_coords(self, n_bins, m_dims, m_scale, m_translate):

        n_voxs = n_bins**3

        # +X right, +Y down, +Z forward (1st plane to last plane)
        # NOTE: for Y (vertical/rows):
        # - when querying in numpy, rows start from top to bottom;
        # - when considering coordinates, Y starts from bottom to top

        # NOTE: in binvox's generation, the grid data's dimensional indexing change speed from slowest to fastest is: XYZ.
        # The following makes X-axis changes slowest, Y-axis changes 2nd fast, and Z-axis changes fastest.
        X, Y, Z = np.meshgrid(np.arange(n_bins), np.arange(n_bins), np.arange(n_bins), indexing="ij")

        # [#bins x #bins x #planes, 3]
        grid_coords = np.stack([np.reshape(X, -1), np.reshape(Y, -1), np.reshape(Z, -1)], axis=1).astype(np.float32)

        # [#bins x #bins x #planes, 3]
        cam_coords = (grid_coords + 0.5) * m_scale / m_dims + m_translate

        # [#bins x #bins x #planes, 4]
        cam_coords = np.concatenate((cam_coords, np.ones((n_voxs, 1))), axis=1)

        return cam_coords

    def compute_binvox_to_2d_pix_coords_s3d(
        self,
        *,
        orig_h,
        orig_w,
        n_bins,
        m_dims,
        m_scale,
        m_translate,
        orig_to_transformed_order,
        crop_info,
        resize_info,
        cam_matrices,
    ):

        tmp_eps = 1e-8

        cam_coords = self.compute_binvox_to_cam_coords(n_bins, m_dims, m_scale, m_translate)

        # [#bins x #bins x #planes, 4]
        ndc_coords = cam_coords @ cam_matrices["cam_to_ndc"]  # convert to ndc coordinate
        ndc_xyz = ndc_coords[:, :3] / (ndc_coords[:, 3:] + tmp_eps)

        # [#bins x #bins x #planes, ]
        orig_pix_coords_x, orig_pix_coords_y = self.ndcs_to_pixels(ndc_xyz[:, 0], ndc_xyz[:, 1], orig_w, orig_h)

        pix_coords_y, pix_coords_x = self.get_transformed_pix_coords_from_orig_img(
            orig_pix_coords_y,
            orig_pix_coords_x,
            orig_to_transformed_order,
            crop_info,
            resize_info,
        )

        return pix_coords_x, pix_coords_y, cam_coords

    def compute_binvox_to_2d_pix_coords_nons3d(
        self,
        *,
        orig_h,
        orig_w,
        n_bins,
        m_dims,
        m_scale,
        m_translate,
        orig_to_transformed_order,
        crop_info,
        resize_info,
        cam_matrices,
    ):

        tmp_eps = 1e-8

        cam_coords_in_s3d = self.compute_binvox_to_cam_coords(n_bins, m_dims, m_scale, m_translate)[
            :, :3
        ]  # [#points, 3]

        cam_coords_in_nons3d = np.matmul(cam_matrices["s3d_to_nons3d"][:3, :3], cam_coords_in_s3d.T).T

        homo_pix_coord = np.matmul(cam_matrices["intri_K"][:3, :3], cam_coords_in_nons3d.T).T  # [#points, 3]
        pix_coord = homo_pix_coord[:, :2] / (homo_pix_coord[:, 2:] + tmp_eps)

        orig_pix_coords_x = pix_coord[:, 0]
        orig_pix_coords_y = pix_coord[:, 1]

        pix_coords_y, pix_coords_x = self.get_transformed_pix_coords_from_orig_img(
            orig_pix_coords_y,
            orig_pix_coords_x,
            orig_to_transformed_order,
            crop_info,
            resize_info,
        )

        return pix_coords_x, pix_coords_y, cam_coords_in_s3d

    def project_binvox_to_2d(
        self,
        *,
        orig_h,
        orig_w,
        binvox_f,
        vis,
        orig_to_transformed_order,
        crop_info,
        resize_info,
        cam_matrices,
    ):

        img_size = vis.shape

        with open(binvox_f, "rb") as f:
            m = binvox_rw.read_as_3d_array(f)

        m_dims = np.array(m.dims).reshape((1, 3))
        assert np.all(m_dims == m_dims[0, 0]), f"{m_dims}"
        n_bins = m_dims[0, 0]

        m_data = np.array(m.data).astype(np.float32)

        # grid_coords = (rep_camera_coords - translate) / scale - 0.5
        m_scale = np.array(m.scale).reshape((1, 3))
        m_translate = np.array(m.translate).reshape((1, 3))

        # print("m_data: ", np.min(m_data), np.max(m_data))

        (pix_coords_x, pix_coords_y, cam_coords,) = self.compute_binvox_to_2d_pix_coords_s3d(
            orig_h=orig_h,
            orig_w=orig_w,
            n_bins=n_bins,
            m_dims=m_dims,
            m_scale=m_scale,
            m_translate=m_translate,
            orig_to_transformed_order=orig_to_transformed_order,
            crop_info=crop_info,
            resize_info=resize_info,
            cam_matrices=cam_matrices,
        )

        if False:
            # # NOTE: DEBUG
            # # save PCL of binvox's camera coords into disk
            # pcl_flag = m_data.reshape(-1) > 0
            # debug_pcl = trimesh.Trimesh(vertices=cam_coords[pcl_flag, :3])
            # pcl_color = (m_data.reshape((-1, 1)) * 255).astype(np.uint8)[pcl_flag, :]
            # print("pcl_color: ", pcl_color.shape)
            # debug_pcl.visual.vertex_colors = np.tile(pcl_color, (1, 3))
            # _ = debug_pcl.export("./binvox_pcl.ply")
            pass

        # NOTE: rows correspond to Y
        # [#bins, #bins, #bins]
        pix_rows = np.round(pix_coords_y).astype(np.int32).reshape(m_dims[0, :])
        pix_cols = np.round(pix_coords_x).astype(np.int32).reshape(m_dims[0, :])

        # [#bins, H, W]
        occ_imgs = np.zeros((n_bins, *img_size))
        for k in range(n_bins):

            # NOTE: when we use meshgrid in self.compute_binvox_to_2d_pix_coords,
            # the axis order for changing speed from slowest to fastest is XYZ.
            # Therefore, to slice Z-axis, we need to index the 3rd dim.
            cur_pix_rows = pix_rows[..., k].reshape(-1)
            cur_pix_cols = pix_cols[..., k].reshape(-1)

            flag_valid_row = (cur_pix_rows >= 0) & (cur_pix_rows < img_size[0])
            flag_valid_col = (cur_pix_cols >= 0) & (cur_pix_cols < img_size[1])
            flag_valid = flag_valid_row & flag_valid_col
            # print("flag_valid: ", np.sum(flag_valid), np.sum(flag_valid_row), np.sum(flag_valid_col), flag_valid.shape)

            valid_pix_rows = cur_pix_rows[flag_valid]
            valid_pix_cols = cur_pix_cols[flag_valid]

            m_data_slice = m_data[..., k].reshape(-1)
            occ_imgs[k, valid_pix_rows, valid_pix_cols] = m_data_slice[flag_valid]

        # NOTE: in binvox's data, Z_index=0 corresponds to the plane that has smallest value of Z.
        # Namely, Z_index=0 means furtheest plane.
        # We need to reverse the Z-axis order to have planes from closest to furthest.
        occ_imgs = np.ascontiguousarray(occ_imgs[::-1, ...])

        return occ_imgs

    def gen_mesh_reverse_mapping(
        self,
        *,
        orig_h,
        orig_w,
        occ_planes,
        plane_depths,
        n_bins,
        vis,
        depth,
        orig_to_transformed_order,
        crop_info,
        resize_info,
        cam_matrices,
        dataset="nons3d",
        smooth_mcube=False,
        use_graph_cut=False,
        graph_cut_info={"bias": 0.0, "pair_weight": 10.0},
    ):
        """
        This function queries point's 3D occupany via querying that point's 2D projection onto occupany image.

        Ref:
        - https://www.cs.princeton.edu/courses/archive/spr11/cos426/notes/cos426_s11_lecture03_warping.pdf
        - http://graphics.cs.cmu.edu/courses/15-463/2011_fall/Lectures/morphing.pdf

        occ_planes: [#planes, H, W]
        plane_depths: [#planes, ]
        """

        tmp_eps = 1e-8

        grid_dims = np.array([n_bins, n_bins, n_bins])

        n_occ_planes, h, w = occ_planes.shape
        assert plane_depths.shape[0] == n_occ_planes, f"{n_occ_planes}, {plane_depths.shape}"
        assert (
            n_bins == n_occ_planes
        ), f"Currently, only support the situation where #planes equals n_bins: {n_occ_planes}, {n_bins}"

        # camera_coords: [#pixs, 3], px/py: [#pixs]
        if dataset == "s3d":
            get_visible_pix_cam_coords_func = self.get_visible_pix_cam_coords_s3d
        elif dataset == "nons3d":
            get_visible_pix_cam_coords_func = self.get_visible_pix_cam_coords_nons3d
        else:
            raise ValueError

        (vis_cam_coords_in_s3d, px, py, px_normalized, py_normalized,) = get_visible_pix_cam_coords_func(
            orig_h=orig_h,
            orig_w=orig_w,
            vis=vis,
            depth=depth,
            orig_to_transformed_order=orig_to_transformed_order,
            crop_info=crop_info,
            resize_info=resize_info,
            cam_matrices=cam_matrices,
        )

        min_z = np.min(plane_depths)
        max_z = np.max(plane_depths)

        # [#points, 3]
        vis_cam_coords_min_z = vis_cam_coords_in_s3d * min_z / (vis_cam_coords_in_s3d[:, 2:] + tmp_eps)
        vis_cam_coords_max_z = vis_cam_coords_in_s3d * max_z / (vis_cam_coords_in_s3d[:, 2:] + tmp_eps)

        # [2 * #points, 3]
        vis_cam_coords_range = np.concatenate((vis_cam_coords_min_z, vis_cam_coords_max_z), axis=0)

        # [3, ]
        min_vis_cam_coords = np.min(vis_cam_coords_range, axis=0)
        max_vis_cam_coords = np.max(vis_cam_coords_range, axis=0)

        # [3, ]
        scale = n_bins / (n_bins - 1) * (max_vis_cam_coords - min_vis_cam_coords)
        translate = min_vis_cam_coords - 0.5 / n_bins * scale

        # These coords are in raw image. [#bins x #bins x #bins]
        if dataset == "s3d":
            compute_binvox_to_2d_pix_coords_func = self.compute_binvox_to_2d_pix_coords_s3d
        elif dataset == "nons3d":
            compute_binvox_to_2d_pix_coords_func = self.compute_binvox_to_2d_pix_coords_nons3d
        else:
            raise ValueError

        # - pix_coords_x_raw/pix_coords_y_raw: [#bins x #bins x #planes, ]
        # - cam_coords_debug: [#bins x #bins x #planes, 4]
        (pix_coords_x_raw, pix_coords_y_raw, cam_coords_debug,) = compute_binvox_to_2d_pix_coords_func(
            orig_h=orig_h,
            orig_w=orig_w,
            n_bins=n_bins,
            m_dims=grid_dims,
            m_scale=scale,
            m_translate=translate,
            orig_to_transformed_order=orig_to_transformed_order,
            crop_info=crop_info,
            resize_info=resize_info,
            cam_matrices=cam_matrices,
        )

        if False:
            # for visualization
            # For grid, 1st dim is for X, 2nd dim is for Y, 3rd dim is for Z
            pix_coords_x_raw_grid = pix_coords_x_raw.reshape((n_bins, n_bins, n_bins))
            pix_coords_y_raw_grid = pix_coords_y_raw.reshape((n_bins, n_bins, n_bins))

            # order: ZXY
            pix_coords_x_raw_grid = np.moveaxis(pix_coords_x_raw_grid, (0, 1, 2), (1, 2, 0))
            pix_coords_y_raw_grid = np.moveaxis(pix_coords_y_raw_grid, (0, 1, 2), (1, 2, 0))

            raw_h = h
            raw_w = w
            grid_2d_img = np.zeros((n_bins, raw_h, raw_w))
            for k in range(n_bins):
                valid_row = (pix_coords_y_raw_grid[k, ...] >= 0) & (pix_coords_y_raw_grid[k, ...] < raw_h)
                valid_col = (pix_coords_x_raw_grid[k, ...] >= 0) & (pix_coords_x_raw_grid[k, ...] < raw_w)
                valid_flag = valid_row & valid_col

                pix_rows_raw = pix_coords_y_raw_grid[k, valid_flag].astype(int)
                pix_cols_raw = pix_coords_x_raw_grid[k, valid_flag].astype(int)
                grid_2d_img[k, pix_rows_raw, pix_cols_raw] = 1

            if resize_info is not None:
                # [#bins, 1, raw_h, raw_w]
                grid_2d_img = torch.FloatTensor(grid_2d_img).unsqueeze(1)
                # grid_2d_img = torch.nn.functional.interpolate(grid_2d_img, size=(int(h / resize_info["row_resize_factor"]), int(w / resize_info["col_resize_factor"])))
                # grid_2d_img = torch.nn.functional.interpolate(grid_2d_img, size=(int(crop_info["end_row"] - crop_info["start_row"]), int(crop_info["end_col"] - crop_info["start_col"])))
                grid_2d_img = torch.nn.functional.interpolate(
                    grid_2d_img,
                    size=(
                        int(resize_info["rows_before_resize"]),
                        int(resize_info["cols_before_resize"]),
                    ),
                )
                grid_2d_img = grid_2d_img[:, 0, ...].numpy()
            if crop_info is not None:
                grid_2d_img_orig = np.zeros((n_bins, orig_h, orig_w))
                grid_2d_img_orig[
                    :,
                    crop_info["start_row"] : crop_info["end_row"],
                    crop_info["start_col"] : crop_info["end_col"],
                ] = grid_2d_img
                grid_2d_img = grid_2d_img_orig

        # Normalized to range [-1, 1]
        pix_coords_x_normalized = pix_coords_x_raw / w * 2 - 1
        pix_coords_y_normalized = pix_coords_y_raw / h * 2 - 1

        if False:
            # NOTE: DEBUG to show the sampling area indeed covers the object.
            plt.scatter(pix_coords_x_normalized, pix_coords_y_normalized)
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            plt.show()

        # [#bins x #bins x #bins, 2]
        # pix_Y corresponds to row (1st dim in image).
        # For grid_sample, 1st dim is used for cols/horizontal, corresponding to pix_X.
        pix_coords = np.stack((pix_coords_x_normalized, pix_coords_y_normalized), axis=1)

        # NOTE: when we use meshgrid in self.compute_binvox_to_2d_pix_coords_s3d,
        # the axis order for changing speed from slowest to fastest is XYZ.
        # Therefore, to slice Z-axis, we need to index the 3rd dim.
        pix_coords = pix_coords.reshape((n_bins * n_bins, n_bins, 2))

        # [#bins, #bins x #bins, 2]
        pix_coords = np.moveaxis(pix_coords, (0, 1, 2), (1, 0, 2))

        if False:
            # NOTE: DEBUG to show the sampling area indeed covers the object.
            plt.scatter(pix_coords[10, :, 0], pix_coords[10, :, 1])
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            plt.show()

        grid_occs = torch.zeros((n_bins, n_bins, n_bins))

        # NOTE: we assume occ_planes store as closest plane at index=0.
        # However, when using np.meshgrid in self.compute_binvox_to_2d_pix_coords_s3d,
        # Z_index=0 corresponds to smallest Z-axis value, namely furthest plane.
        # Therefore, we need to flip it.
        occ_planes_flipped = np.ascontiguousarray(occ_planes[::-1, ...])

        # [#planes, 1, H, W]
        occ_planes_torch = torch.FloatTensor(occ_planes_flipped).unsqueeze(1)

        # [#bins, 1, #bins x #bins, 2]
        pix_coords_torch = torch.FloatTensor(pix_coords).unsqueeze(1)

        # [#bins, 1, 1, #bins x #bins]
        sampled_occs = torch.nn.functional.grid_sample(
            occ_planes_torch, pix_coords_torch, mode="nearest", align_corners=False
        )  # mode="bilinear", align_corners=False)

        # NOTE: DEBUG
        # plt.imshow(torch.sum(occ_planes_torch, dim=[0, 1]) > 0)
        # plt.show()

        # 1st dim for Z-axis, 2nd dim for X, 3rd dim for Y
        grid_occs = sampled_occs.squeeze().reshape((n_bins, n_bins, n_bins)).numpy()

        # # Un-do flipping to make storage with closest plane at 1st_index=0
        # grid_occs = np.ascontiguousarray(grid_occs[::-1, ...])

        # ZXY -> XYZ
        grid_occs = np.moveaxis(grid_occs, (0, 1, 2), (2, 0, 1))

        # NOTE:
        # - In grid_occs, 1st dim for X; 2nd dim for Y; 3rd dim for Z
        # - For each dim, index=0 corresponds to smallest value the axis that dimension represents.
        #    - For example, 3rd dim index=0 corresponds to smallest value in Z, namely furthest plane.
        # - Pay attention to this as when plt.imshow(np.sum(debug_grid_occs, axis=0) > 0), you may see upside-down image etc.

        if False:
            # # NOTE: DEBUG
            # # debug_pcl = trimesh.Trimesh(vertices=cam_coords_debug[:, :3])
            # # # save PCL of binvox's camera coords into disk
            # pcl_flag = grid_occs.reshape(-1) > 0
            # debug_pcl = trimesh.Trimesh(vertices=cam_coords_debug[pcl_flag, :3])
            # pcl_color = (grid_occs.reshape((-1, 1)) * 255).astype(np.uint8)[pcl_flag, :]
            # print("pcl_color: ", pcl_color.shape)
            # debug_pcl.visual.vertex_colors = np.tile(pcl_color, (1, 3))
            # _ = debug_pcl.export("./binvox_pcl_reverse.ply")
            pass

        if use_graph_cut:

            from medpy.graphcut.graph import GCGraph

            bias = graph_cut_info["bias"]
            pair_weight = graph_cut_info["pair_weight"]

            # inverse of sigmoid
            grid_logits = np.log(1e-8 + grid_occs / (1 - grid_occs + 1e-8))

            # Must use logit
            tmp = np.reshape(grid_logits, -1)
            tweights = [(max((x - bias), 0), max(-(x - bias), 0)) for x in tmp]
            nweights = {
                (x, x + (n_bins**y)): (pair_weight, pair_weight)
                for x in range(n_bins**3)
                for y in [0, 1, 2]
                if ((x // (n_bins**y)) % (n_bins)) != (n_bins - 1)
            }
            assert len(nweights) == n_bins * n_bins * (n_bins - 1) * 3
            gcgraph = GCGraph(n_bins**3, n_bins * n_bins * (n_bins - 1) * 3)
            gcgraph.set_tweights_all(tweights)
            gcgraph.set_nweights(nweights)
            tmp = gcgraph.get_graph()
            flow = tmp.maxflow()
            result = np.zeros_like(grid_occs)
            for idx in range(result.size):
                result[np.unravel_index(idx, result.shape)] = -1 if tmp.termtype.SINK == tmp.what_segment(idx) else 1
            grid_occs = result

        if smooth_mcube:
            pad_for_watertight = False

            if use_graph_cut:
                smoothed_grid_occs = mcubes.smooth(grid_occs < 0.0)
            else:
                # https://github.com/autonomousvision/occupancy_networks/blob/406f79468fb8b57b3e76816aaa73b1915c53ad22/im2mesh/onet/generation.py#L174
                smoothed_grid_occs = mcubes.smooth(grid_occs < 0.5)
            vertices, triangles = mcubes.marching_cubes(smoothed_grid_occs, 0)
        else:
            pad_for_watertight = True

            if use_graph_cut:
                raise NotImplementedError
            else:
                # https://github.com/autonomousvision/occupancy_networks/blob/406f79468fb8b57b3e76816aaa73b1915c53ad22/im2mesh/onet/generation.py#L174
                # This will result in mesh shift as grid_dims chagnes.
                grid_occs = np.pad(grid_occs, 1, "constant", constant_values=-1e6)
                vertices, triangles = mcubes.marching_cubes(grid_occs, 0.5)

        if pad_for_watertight:
            # Undo padding
            vertices -= 1

        if dataset == "s3d":
            verts_in_cam_coord = (vertices + 0.5) / grid_dims * scale + translate
        elif dataset == "nons3d":
            verts_in_s3d_cam_coord = (vertices + 0.5) / grid_dims * scale + translate

            verts_in_cam_coord = np.matmul(cam_matrices["s3d_to_nons3d"], verts_in_s3d_cam_coord.T).T
        else:
            raise ValueError

        return verts_in_cam_coord, triangles, grid_occs
