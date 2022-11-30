import os
import sys
import glob
import tqdm
import traceback
import trimesh
import logging
import shutil
import random
import argparse
import pickle
import subprocess
import numpy as np
import multiprocessing as mp
import numpy as np
from re import M
from PIL import Image, ImageOps
from PIL.ImageFilter import GaussianBlur
from skimage import transform
from scipy.ndimage import maximum_filter, gaussian_filter
from scipy.ndimage.morphology import binary_erosion
from oplanes.utils.img_utils import crop_vis_map

import torch


N_POINTS = 500000

# sigma_ratio = 0.02 is estimated from PIFuHD:
# it uses 3 or 5 cm. Their bbox is 128.
# Therefore 3 / 128 = 0.02
SIGMA_RATIO = 0.02
HEURISTIC_DEPTH_RANGE = 2.0


class MeshObj:
    def __init__(self, fn, fnroot=""):
        self.raw_fn = fn
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

        self.name = "-".join([self.csName, arr[-3], arr[-2]])

        self.LoadRMatrices()

        self.avoid_nan_eps = 1e-8

    def LoadRMatrices(self):
        rmatrices_file = sorted(glob.glob(self.fn[:-8] + "draw_*/rage_matrices_bin.csv"))[0]
        rage_matrices = np.fromfile(rmatrices_file, dtype=np.float32).astype(self.float_type)
        rage_matrices = rage_matrices.reshape((4, 4, 4))
        self.VP = np.dot(np.linalg.inv(rage_matrices[0, :, :]), rage_matrices[2, :, :])
        self.VP_inverse = np.linalg.inv(self.VP)  # multiply this matrix to convert from NDC to world coordinate
        self.P = np.dot(np.linalg.inv(rage_matrices[1, :, :]), rage_matrices[2, :, :])
        self.P_inverse = np.linalg.inv(self.P)  # multiply this matrix to convert from NDC to camera coordinate

    def get_mesh_bbox(self):
        mesh = trimesh.load(self.wt_mesh_f)
        assert mesh.is_watertight, f"{self.fn}"

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

    def project_mesh_to_img(self):
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

    def project_pts_to_img(self, pts, crop_info=None):
        ndcpts = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1) @ self.P
        ndcpts = ndcpts[:, 0:2] / ndcpts[:, -1:]
        xx, yy = self.ndcs_to_pixels(ndcpts[:, 0], ndcpts[:, 1])
        xxi = np.rint(xx).astype(int)
        yyi = np.rint(yy).astype(int)
        if crop_info is None:
            select = np.logical_and(
                np.logical_and(xxi > 0, xxi <= self.IMAGE_SIZE[1]),
                np.logical_and(yyi > 0, yyi <= self.IMAGE_SIZE[0]),
            )
        else:
            select = np.logical_and(
                np.logical_and(xxi >= crop_info["start_col"], xxi <= crop_info["end_col"]),
                np.logical_and(yyi >= crop_info["start_row"], yyi <= crop_info["end_row"]),
            )
        return xx, yy, xxi, yyi, select

    def project_mesh_to_image(self):
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

    def compute_visible_mesh_depth_range(self):
        xx, yy, xxi, yyi, select, obj = self.project_mesh_to_image()
        vis_verts = np.array(obj.vertices)[select, :]
        min_coords = np.min(vis_verts, axis=0)
        max_coords = np.max(vis_verts, axis=0)
        depth_range = max_coords[2] - min_coords[2]
        return depth_range, max_coords[2]

    def _compute_bbox(self, vis, depth, depth_range):
        # get camera coords
        py_orig, px_orig = np.nonzero(vis)
        px = px_orig
        py = py_orig
        ndcx, ndcy = self.pixels_to_ndcs(px, py)
        ndcz = depth[py, px]

        ndc_coord = np.stack([ndcx, ndcy, ndcz, np.ones_like(ndcz)], axis=1)  # NDC
        camera_coord = ndc_coord @ self.P_inverse  # convert to camera coordinate, [#pixels, 3]
        camera_coord = camera_coord[:, 0:3] / camera_coord[:, -1:]  # divide, [#pixels, 3]

        threshold = 0.25

        tmp_max_z = np.max(camera_coord[:, 2])
        # max_z = tmp_max_z + threshold * depth_range
        max_z = tmp_max_z  # no need to sample before depth map
        min_z = tmp_max_z - depth_range * (1 + threshold)

        camera_coord_min = camera_coord * min_z / (camera_coord[:, 2:] + 1e-8)
        camera_coord_max = camera_coord * max_z / (camera_coord[:, 2:] + 1e-8)

        all_camera_coord = np.concatenate((camera_coord_min, camera_coord_max), axis=0)

        # compute bounding box
        min_coords = np.min(all_camera_coord, axis=0)
        max_coords = np.max(all_camera_coord, axis=0)

        # NOTE: Z-range is not the true range as Z values come from depth maps, which displays the front
        coords_range = max_coords - min_coords

        b_min = min_coords - threshold * coords_range
        b_max = max_coords + threshold * coords_range

        # NOTE, we need a heuristic depth range to compute coord_range for Z.
        b_min[2] = min_z
        b_max[2] = max_z

        return b_min, b_max, camera_coord, px_orig, py_orig

    def _get_cam_depth(self, depth):
        h, w = self.IMAGE_SIZE
        all_rows, all_cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        all_pys = all_rows.reshape(-1)
        all_pxs = all_cols.reshape(-1)

        ndcx, ndcy = self.pixels_to_ndcs(all_pxs, all_pys)
        ndcz = depth.reshape(-1)
        ndc_coord = np.stack([ndcx, ndcy, ndcz, np.ones_like(ndcz)], axis=1)  # NDC, [#pixels, 4]

        camera_coord = ndc_coord @ self.P_inverse  # convert to camera coordinate, [#pixels, 3]
        camera_coord = camera_coord[:, 0:3] / camera_coord[:, -1:]  # divide, [#pixels, 3]

        cam_depth = camera_coord[:, 2].reshape((h, w))

        return cam_depth

    # def get_data(self, target_h=512, target_w=512, crop_expand_ratio=0.1, heuristic_depth_range=2.0, is_train=True):
    def get_data(self, crop_expand_ratio=0.1, depth_range=None, compute_crop_info=True):

        vis_orig = np.load(self.visfn) == self.objID
        depth = np.load(self.depthfn).astype(self.float_type) / 6.0 - 4e-5

        # vis = maximum_filter(vis_orig, footprint=np.ones((5, 5)))
        # NOTE: we need this to avoid misalignment between depth and visible mask
        vis = binary_erosion(vis_orig, np.ones((10, 10)))

        if depth_range is None:
            depth_range, _ = self.compute_visible_mesh_depth_range()
        # print("\ndepth_range: ", depth_range, "\n")

        if compute_crop_info:
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

            assert (
                end_row - start_row == end_col - start_col
            ), f"\nCurrently only support squared crop. However, we get {start_row}, {end_row}, {start_col}, {end_col}.\n"

            crop_info = {
                "start_row": start_row,
                "end_row": end_row,
                "start_col": start_col,
                "end_col": end_col,
            }
        else:
            crop_info = None

        b_min, b_max, camera_coord, px_orig, py_orig = self._compute_bbox(vis, depth, depth_range)
        # # We need this information, in S3D, +Z is backward.
        # closest_depth = np.max(camera_coord[:, 2])

        depth_cam_orig = -9999 * np.ones(vis_orig.shape)
        depth_cam_orig[py_orig, px_orig] = camera_coord[:, 2]

        return {
            "b_min": b_min,
            "b_max": b_max,
            "depth_cam_orig": depth_cam_orig,
            "crop_info": crop_info,
        }


def filter_and_save_uniform_points(sample_points, n_tgt_points, subject, mesh, save_f, crop_info=None):

    cols, rows, col_int, row_int, flag_valid = subject.project_pts_to_img(sample_points, crop_info=crop_info)
    sample_points = sample_points[flag_valid, :]
    sample_points = sample_points[:n_tgt_points, :]

    inside = mesh.contains(sample_points.copy())

    n_valid_points = sample_points.shape[0]

    save_f = f"{save_f}_{n_valid_points}"
    np.savez(save_f, points=sample_points.astype(float), label=inside)


def filter_and_save_surface_points(sample_points, normals, n_tgt_points, subject, mesh, save_f, crop_info=None):

    cols, rows, col_int, row_int, flag_valid = subject.project_pts_to_img(sample_points, crop_info=crop_info)

    sample_points = sample_points[flag_valid, :]
    normals = normals[flag_valid, :]

    sample_points = sample_points[:n_tgt_points, :]
    normals = normals[:n_tgt_points, :]

    n_valid_points = sample_points.shape[0]

    save_f = f"{save_f}_{n_valid_points}"
    np.savez(save_f, points=sample_points.astype(float), normals=normals)


def sample_and_save_points(
    wt_mesh_f,
    subject,
    b_min,
    b_max,
    num_tgt_points=N_POINTS,
    save_dir=None,
    crop_info=None,
    flag_heuristic=False,
    flag_surface=False,
):

    random.seed(1991)
    np.random.seed(1991)
    torch.manual_seed(1991)

    num_points = int(num_tgt_points * 2)

    mesh = trimesh.load(wt_mesh_f, process=False)

    if flag_surface:
        random_points, face_idx = mesh.sample(num_points, return_index=True)
        face_normals = np.copy(mesh.face_normals)
        save_f = os.path.join(save_dir, f"surface_normal_toward_cam")

        v0_ind = mesh.faces[:, 0]  # [#faces,]
        v0 = mesh.vertices[v0_ind]  # [#faces, 3]
        N = np.copy(face_normals)
        R = (v0 * N).sum(1)  # [#faces,]
        # Normal and vertex has angle > 180 degree
        normal_toward_cam_id = R < 0
        # flip so that all normals point towards camera
        face_normals[~normal_toward_cam_id, :] *= -1

        vert_normals = face_normals[face_idx]

        filter_and_save_surface_points(
            random_points,
            vert_normals,
            num_tgt_points,
            subject,
            mesh,
            save_f,
            crop_info=crop_info,
        )
    else:
        length = (b_max - b_min).reshape((1, 3))
        random_points = np.random.rand(num_points, 3) * length + b_min

        if flag_heuristic:
            save_f = os.path.join(save_dir, f"random_uniform_heuristic_bbox")
        else:
            save_f = os.path.join(save_dir, f"random_uniform_gt_bbox")

        filter_and_save_uniform_points(random_points, num_tgt_points, subject, mesh, save_f, crop_info=crop_info)


def process_single_scene(fn, data_root, bin_f, save_dir, is_1st_time=True):

    wt_mesh_f = os.path.join(save_dir, "wt_mesh.obj")

    cur_subject = MeshObj(fn, fnroot=data_root)

    gen_wt_cmd = f"{bin_f} --input {cur_subject.fn} --output {wt_mesh_f}"

    # https://stackoverflow.com/questions/64770786/how-to-catch-the-errors-of-a-child-process-using-python-subprocess
    try:
        subprocess.run(
            gen_wt_cmd,
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as e:
        print(f"\n{fn} exited with exit status {e.returncode}: {e.stderr}\n")

    assert os.path.exists(wt_mesh_f), f"\nFail to generate wt mesh for {fn}.\n"

    if not is_1st_time:

        # ---------------------------------------------------------------------
        # GT depth_range

        crop_info_f = os.path.join(save_dir, "crop_info_f.pkl")
        with open(crop_info_f, "rb") as f:
            img_crop_info = pickle.load(f)

        gt_bbox_info_f = os.path.join(save_dir, "gt_bbox_info_f.pkl")
        with open(gt_bbox_info_f, "rb") as f:
            gt_bbox_info = pickle.load(f)

        old_gt_sample_f_list = list(glob.glob(os.path.join(save_dir, "*random_uniform_gt_bbox_*.npz")))
        assert len(old_gt_sample_f_list) <= 3, f"{fn}"
        for tmp_f in old_gt_sample_f_list:
            os.remove(tmp_f)

        sample_and_save_points(
            wt_mesh_f,
            cur_subject,
            gt_bbox_info["b_min"],
            gt_bbox_info["b_max"],
            num_tgt_points=N_POINTS,
            save_dir=save_dir,
            crop_info=img_crop_info,
            flag_heuristic=False,
            flag_surface=False,
        )

        # ---------------------------------------------------------------------
        # heuristic depth_range

        heuristic_bbox_info_f = os.path.join(save_dir, "heuristic_bbox_info_f.pkl")
        with open(heuristic_bbox_info_f, "rb") as f:
            heuristic_bbox_info = pickle.load(f)

        old_heuristic_sample_f_list = list(glob.glob(os.path.join(save_dir, "*random_uniform_heuristic_bbox_*.npz")))
        assert len(old_heuristic_sample_f_list) <= 1, f"{fn}"
        for tmp_f in old_heuristic_sample_f_list:
            os.remove(tmp_f)

        sample_and_save_points(
            wt_mesh_f,
            cur_subject,
            heuristic_bbox_info["b_min"],
            heuristic_bbox_info["b_max"],
            num_tgt_points=N_POINTS,
            save_dir=save_dir,
            crop_info=img_crop_info,
            flag_heuristic=True,
            flag_surface=False,
        )

        # ---------------------------------------------------------------------
        # surface

        print("\nDone\n")

        sample_and_save_points(
            wt_mesh_f,
            cur_subject,
            None,
            None,
            num_tgt_points=N_POINTS,
            save_dir=save_dir,
            crop_info=img_crop_info,
            flag_heuristic=False,
            flag_surface=True,
        )
    else:
        # ---------------------------------------------------------------------
        # GT depth_range
        gt_data_dict = cur_subject.get_data(depth_range=None)

        img_crop_info = gt_data_dict["crop_info"]
        crop_info_f = os.path.join(save_dir, "crop_info_f.pkl")
        with open(crop_info_f, "wb") as f:
            pickle.dump(img_crop_info, f)

        gt_bbox_info_f = os.path.join(save_dir, "gt_bbox_info_f.pkl")
        with open(gt_bbox_info_f, "wb") as f:
            pickle.dump({"b_min": gt_data_dict["b_min"], "b_max": gt_data_dict["b_max"]}, f)

        depth_cam_f = os.path.join(save_dir, "depth_cam_orig")
        np.savez(depth_cam_f, depth_cam_orig=gt_data_dict["depth_cam_orig"])

        sample_and_save_points(
            wt_mesh_f,
            cur_subject,
            gt_data_dict["b_min"],
            gt_data_dict["b_max"],
            num_tgt_points=N_POINTS,
            save_dir=save_dir,
            crop_info=img_crop_info,
            flag_heuristic=False,
            flag_surface=False,
        )

        # ---------------------------------------------------------------------
        # heuristic depth_range
        heuristic_data_dict = cur_subject.get_data(depth_range=HEURISTIC_DEPTH_RANGE, compute_crop_info=None)

        heuristic_bbox_info_f = os.path.join(save_dir, "heuristic_bbox_info_f.pkl")
        with open(heuristic_bbox_info_f, "wb") as f:
            pickle.dump(
                {
                    "b_min": heuristic_data_dict["b_min"],
                    "b_max": heuristic_data_dict["b_max"],
                    "heurictic_depth_range": HEURISTIC_DEPTH_RANGE,
                },
                f,
            )

        sample_and_save_points(
            wt_mesh_f,
            cur_subject,
            heuristic_data_dict["b_min"],
            heuristic_data_dict["b_max"],
            num_tgt_points=N_POINTS,
            save_dir=save_dir,
            crop_info=img_crop_info,
            flag_heuristic=True,
            flag_surface=False,
        )

        # ---------------------------------------------------------------------
        # surface

        sample_and_save_points(
            wt_mesh_f,
            cur_subject,
            None,
            None,
            num_tgt_points=N_POINTS,
            save_dir=save_dir,
            crop_info=img_crop_info,
            flag_heuristic=False,
            flag_surface=True,
        )

        mesh = trimesh.load(wt_mesh_f, process=False)
        all_verts = np.array(mesh.vertices)
        cols, rows, col_int, row_int, flag_valid = cur_subject.project_pts_to_img(all_verts, crop_info=img_crop_info)
        valid_faces = flag_valid[mesh.faces]
        face_mask = valid_faces.all(axis=1)
        mesh.update_faces(face_mask)
        _ = mesh.export(os.path.join(save_dir, "vis_mesh.obj"))

    # Important: remember to delete wt mesh to save space
    os.remove(wt_mesh_f)

    return "Done"


def process_subproc(subproc_input):

    proc_id, bin_f, chunk_lines, data_root, save_root_dir, is_1st_time = subproc_input

    for cur_line in tqdm.tqdm(chunk_lines):

        cur_save_dir = os.path.join(save_root_dir, cur_line[2:-8])
        os.makedirs(cur_save_dir, exist_ok=True)

        process_single_scene(cur_line, data_root, bin_f, cur_save_dir, is_1st_time=is_1st_time)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc", type=int, required=False, default=10)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--split_f", type=str, required=True)
    parser.add_argument("--save_root_dir", type=str, required=True)
    parser.add_argument("--bin_f", type=str, required=True)
    parser.add_argument("--is_1st_time", type=int, required=True)
    args = parser.parse_args()

    nproc = args.nproc

    with open(args.split_f) as f:
        all_lines = f.read().splitlines()

    print(f"\nFind {len(all_lines)} files.\n")

    f_chunk = [[] for _ in range(nproc)]
    for i, tmp_f in enumerate(all_lines):
        f_chunk[i % nproc].append(tmp_f)

    split_fname = os.path.basename(args.split_f).split(".")[0]
    save_dir = os.path.join(args.save_root_dir, split_fname)

    # NOTE: np.matmul may freeze when using default "fork"
    # https://github.com/ModelOriented/DALEX/issues/412
    with mp.get_context("spawn").Pool(nproc) as pool:
        gathered_ret_dicts = pool.map(
            process_subproc,
            zip(
                range(nproc),
                [args.bin_f for _ in range(nproc)],
                f_chunk,
                [args.data_root for _ in range(nproc)],
                [save_dir for _ in range(nproc)],
                [bool(args.is_1st_time) for _ in range(nproc)],
            ),
        )
        pool.close()
        pool.join()
