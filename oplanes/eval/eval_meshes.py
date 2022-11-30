import os
import glob
import argparse
import trimesh
import tqdm
import joblib
import random
import numpy as np
import pandas as pd
import multiprocessing as mp

import torch

from external.onet.im2mesh.eval import MeshEvaluator


SEED = 1


N_POINTS = 100000
CHAMFER_UNIT_RATIO = 0.1
PRED_MESH_FNAME = "pred_non_smooth.ply"
IOU_FNAME = "random_uniform_heuristic_bbox"


def sample_surface_pts(mesh, num_points):
    random_points, face_idx = mesh.sample(num_points, return_index=True)
    face_normals = np.copy(mesh.face_normals)

    v0_ind = mesh.faces[:, 0]  # [#faces,]
    v0 = mesh.vertices[v0_ind]  # [#faces, 3]
    N = np.copy(face_normals)
    R = (v0 * N).sum(1)  # [#faces,]
    # Normal and vertex has angle > 180 degree
    normal_toward_cam_id = R < 0

    # flip so that all normals point towards camera
    face_normals[~normal_toward_cam_id, :] *= -1
    vert_normals = np.array(face_normals[face_idx])

    return random_points, vert_normals


def eval_single_mesh(*, cur_line, pred_dir, gt_dir):

    # Evaluator
    evaluator = MeshEvaluator(n_points=N_POINTS)

    eval_dict = {"name": cur_line}

    # Rescale chamfer s.t. the unit 1 if 1/10 of the longest edge of bbox.
    # Ref: Sec. 4 of https://arxiv.org/pdf/1812.03828.pdf
    vis_mesh_f = os.path.join(gt_dir, "vis_mesh.obj")
    vis_mesh = trimesh.load(vis_mesh_f, process=False)
    vis_coord_min = np.min(np.array(vis_mesh.vertices), axis=0)
    vis_coord_max = np.max(np.array(vis_mesh.vertices), axis=0)
    bbox_edge_lens = np.abs(vis_coord_max - vis_coord_min)
    unit_len = CHAMFER_UNIT_RATIO * np.max(bbox_edge_lens)
    eval_dict["unit_len"] = unit_len

    # start evaluation
    mesh_f = os.path.join(pred_dir, PRED_MESH_FNAME)

    if os.path.exists(mesh_f):
        mesh = trimesh.load(mesh_f, process=False, force="mesh")
        eval_dict["fail_to_gen_mesh"] = False
    else:
        print(f"\n{cur_line} does not have mesh.\n")
        eval_dict["fail_to_gen_mesh"] = True
        mesh = trimesh.Trimesh()

    # load surface points on GT mesh
    surface_f_list = list(glob.glob(os.path.join(gt_dir, "surface_normal_toward_cam_*.npz")))
    assert len(surface_f_list) == 1, f"{gt_dir}"
    surface_f = surface_f_list[0]

    surface_pcl = np.load(surface_f)
    pointcloud_tgt = surface_pcl["points"]
    normals_tgt = np.array(surface_pcl["normals"])

    surface_idxs = np.arange(pointcloud_tgt.shape[0])
    np.random.shuffle(surface_idxs)
    surface_idxs = surface_idxs[:N_POINTS]
    n_pcl_points = surface_idxs.shape[0]
    eval_dict["n_pcl_points"] = n_pcl_points

    pointcloud_tgt = pointcloud_tgt[surface_idxs, :]
    normals_tgt = normals_tgt[surface_idxs, :]

    # sample points on pred mesh
    if mesh.vertices.shape[0] == 0:
        pointcloud_pred = np.zeros((0, 3))
        normals_pred = np.zeros((0, 3))
    else:
        try:
            pointcloud_pred, normals_pred = sample_surface_pts(mesh, n_pcl_points)
        except:
            print("\n", cur_line, "\n")
            import traceback
            import sys

            traceback.print_exc()
            err = sys.exc_info()[0]
            print(err)

    # load uniform points
    uniform_f_list = list(glob.glob(os.path.join(gt_dir, f"{IOU_FNAME}_*.npz")))
    assert len(uniform_f_list) == 1, f"{gt_dir}"
    uniform_f = uniform_f_list[0]

    uniform_pcl = np.load(uniform_f)
    points_iou = uniform_pcl["points"][:N_POINTS]
    occ_tgt = uniform_pcl["label"][:N_POINTS]

    eval_dict["n_iou_points"] = points_iou.shape[0]

    # rescale by unit_len
    pointcloud_tgt = pointcloud_tgt / unit_len
    normals_tgt = normals_tgt / unit_len
    pointcloud_pred = pointcloud_pred / unit_len
    normals_pred = normals_pred / unit_len

    eval_dict_mesh = evaluator.eval_mesh(
        mesh,
        pointcloud_tgt.astype(np.float32),
        normals_tgt.astype(np.float32),
        pointcloud_pred.astype(np.float32),
        normals_pred.astype(np.float32),
        points_iou.astype(np.float32),
        occ_tgt.astype(np.float32),
    )
    eval_dict.update(eval_dict_mesh)

    return eval_dict


def eval_subproc(subproc_input):

    (
        proc_id,
        chunk_lines,
        save_root_dir,
        gt_root_dir,
        pred_root_dir,
    ) = subproc_input

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    all_eval_dicts = {}

    for cur_line in tqdm.tqdm(chunk_lines):
        cur_gt_dir = os.path.join(gt_root_dir, cur_line[2:-8])
        cur_pred_dir = os.path.join(pred_root_dir, cur_line[2:-8])
        cur_eval_dict = eval_single_mesh(
            cur_line=cur_line,
            pred_dir=cur_pred_dir,
            gt_dir=cur_gt_dir,
        )
        all_eval_dicts[cur_line] = cur_eval_dict

    with open(os.path.join(save_root_dir, f"eval_dict_proc_{proc_id}.pt"), "wb") as f:
        joblib.dump(all_eval_dicts, f)

    return all_eval_dicts


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc", type=int, required=False, default=10)
    parser.add_argument("--ckpt_f", type=str, required=True)
    parser.add_argument("--split_f", type=str, required=True)
    parser.add_argument("--n_bins", type=int, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    args = parser.parse_args()

    split_fname = os.path.basename(args.split_f).split(".")[0]

    ckpt_path = args.ckpt_f
    print("\nckpt_f: ", ckpt_path, "\n")

    # example: /XXX/checkpoints/XXX.pt
    eval_root_dir = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), f"eval")

    folder_name = f"{args.n_bins}"
    pred_dir = os.path.join(eval_root_dir, folder_name, split_fname)

    save_root_dir = os.path.join(eval_root_dir, f"results/{folder_name}", split_fname)

    gt_dir = os.path.join(args.gt_dir, split_fname)

    os.makedirs(save_root_dir, exist_ok=True)

    print("\neval_root_dir: ", eval_root_dir, "\n")
    print("\npred_dir: ", pred_dir, "\n")
    print("\nsave_root_dir: ", save_root_dir, "\n")

    nproc = args.nproc

    with open(args.split_f) as f:
        all_lines = f.read().splitlines()

    np.random.shuffle(all_lines)

    print(f"\nFind {len(all_lines)} files.\n")

    f_chunk = [[] for _ in range(nproc)]
    for i, tmp_f in enumerate(all_lines):
        f_chunk[i % nproc].append(tmp_f)

    # NOTE: np.matmul may freeze when using default "fork"
    # https://github.com/ModelOriented/DALEX/issues/412
    with mp.get_context("spawn").Pool(nproc) as pool:
        gathered_eval_dicts = pool.map(
            eval_subproc,
            zip(
                range(nproc),
                f_chunk,
                [save_root_dir for _ in range(nproc)],
                [gt_dir for _ in range(nproc)],
                [pred_dir for _ in range(nproc)],
            ),
        )
        pool.close()
        pool.join()

    all_eval_dicts = {}
    for elem in gathered_eval_dicts:
        all_eval_dicts.update(elem)

    with open(os.path.join(save_root_dir, "eval_dicts_all.pt"), "wb") as f:
        joblib.dump(all_eval_dicts, f)

    fail_cnt = 0
    for k in all_eval_dicts:
        if ("fail_to_gen_mesh" in all_eval_dicts[k]) and all_eval_dicts[k]["fail_to_gen_mesh"]:
            fail_cnt += 1
    print(f"\nNumber of failing cases: {fail_cnt}.\n")

    # Create pandas dataframe and save
    eval_df = pd.DataFrame(list(all_eval_dicts.values()))
    eval_df.to_pickle(os.path.join(save_root_dir, "eval_df.pkl"))

    # Print results
    eval_df.loc["mean"] = eval_df.mean()
    print(eval_df)
    print("\n", eval_df.mean(), "\n")
