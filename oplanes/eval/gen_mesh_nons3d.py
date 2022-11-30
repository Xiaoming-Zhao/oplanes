import os
import tqdm
import glob
import copy
import joblib
import pathlib
import pickle
import trimesh
import argparse
import numpy as np
import multiprocessing as mp
from PIL import Image
from skimage import transform

import torch

from oplanes.data.meshdata_nons3d import DataObjNonS3D
from oplanes.trainer.oplanes_trainer import OPlanes_Trainer
from oplanes.utils.config import Config, get_config, convert_cfg_to_dict
from oplanes.models.mesh_generator import MeshGenerator

pixel_mean = np.array([123.675, 116.280, 103.530]).reshape((1, 1, 3))
pixel_std = np.array([58.395, 57.120, 57.375]).reshape((1, 1, 3))

HEURISTIC_DEPTH_RANGE = 2.0

SMOOTH_MESH_FNAME = "pred_smooth"
NONSMOOTH_MESH_FNAME = "pred_non_smooth"


def extract_mesh(
    *,
    rgb_f,
    depth_f,
    mask_f,
    cam_mat_f,
    device,
    model,
    config,
    save_dir,
    save_mesh_f,
    n_bins=128,
    smooth_mcube=False,
    use_graph_cut=False,
    graph_cut_info={"bias": 0.0, "pair_weight": 10.0},
    disable_tqdm=True,
):

    given_depth_range = HEURISTIC_DEPTH_RANGE  # use None for GT depth range

    cur_obj = DataObjNonS3D(rgb_f, depth_f, mask_f, cam_mat_f)

    tmp_mb = 2

    val_data_cpu = cur_obj.load_data(
        n_planes=n_bins,
        is_val=True,
        for_debug=True,
        use_masked_out_img=config.TRAIN.PROCESS_DATA.use_masked_out_img,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        given_depth_range=given_depth_range,
    )

    val_data_cuda = {
        k: v.unsqueeze(0).to(device)
        for k, v in val_data_cpu.items()
        if k not in ["idx", "fn", "p_d", "crop_info", "resize_info", "sampled_bins_vec"]
    }

    # -----------------------------------------------------------------------------------
    # Run the model
    all_infer_pred_logits = []
    all_flag_after_mesh = []

    tmp_mb = 2

    for tmp_start in tqdm.tqdm(range(0, n_bins, tmp_mb), disable=disable_tqdm):
        tmp_end = min(tmp_start + tmp_mb, n_bins)
        # torch.cuda.empty_cache()
        # [1, #planes]
        tmp_p_d = val_data_cpu["p_d"][:, tmp_start:tmp_end]
        val_data_cuda["p_d"] = tmp_p_d.to(device)
        if "pred_z_range" in config.MODEL.name:
            tmp_out, info_for_log = model(val_data_cuda)
        else:
            tmp_out = model(val_data_cuda)  # [B, C, feat_h, feat_w]
        if not isinstance(tmp_out, list):
            tmp_out = [tmp_out]
        # The final prediction is the last element of the list.
        tmp_out = tmp_out[-1]
        tmp_infer_pred_logits, tmp_flag_after_mesh = tmp_out
        tmp_infer_pred_logits = tmp_infer_pred_logits.detach().cpu()
        tmp_flag_after_mesh = tmp_flag_after_mesh.detach().cpu()
        all_infer_pred_logits.append(tmp_infer_pred_logits)
        all_flag_after_mesh.append(tmp_flag_after_mesh)

    all_infer_pred_logits = torch.cat(all_infer_pred_logits, dim=1)
    all_flag_after_mesh = torch.cat(all_flag_after_mesh, dim=1)

    # [#bins, fine_H, fine_W]
    all_infer_pred_probs = torch.sigmoid(all_infer_pred_logits)[0, ...]
    all_flag_after_mesh = all_flag_after_mesh[0, ...]
    # We use depth map to filter out any prediction before the mesh.
    all_infer_pred_probs = all_infer_pred_probs * all_flag_after_mesh

    # Prepare resize information
    orig_h, orig_w = cur_obj.mask.shape

    cur_h = all_infer_pred_probs.shape[1]
    cur_w = all_infer_pred_probs.shape[2]

    transform_order = "crop_resize"
    crop_info = val_data_cpu["crop_info"]
    resize_info = {
        "row_resize_factor": cur_h / (crop_info["end_row"] - crop_info["start_row"]),
        "col_resize_factor": cur_w / (crop_info["end_col"] - crop_info["start_col"]),
    }
    resize_info["rows_after_resize"] = cur_h
    resize_info["colss_after_resize"] = cur_w
    resize_info["rows_before_resize"] = crop_info["end_row"] - crop_info["start_row"]
    resize_info["cols_before_resize"] = crop_info["end_col"] - crop_info["start_col"]

    cam_matrices = {
        "intri_K": cur_obj.intri_K,
        "inv_K": cur_obj.inv_K,
        "nons3d_to_s3d": cur_obj.nons3d_to_s3d_coord_sys,
        "s3d_to_nons3d": np.linalg.inv(cur_obj.nons3d_to_s3d_coord_sys),
    }

    occ_planes = all_infer_pred_probs.numpy()
    plane_depths = val_data_cpu["p_d"].reshape((n_bins, 1)).numpy()

    resized_vis = transform.resize(val_data_cpu["vis"].numpy(), (cur_h, cur_w), order=0)
    resized_depth = transform.resize(val_data_cpu["depth"].numpy(), (cur_h, cur_w), order=0)

    occ_planes = all_infer_pred_probs.numpy()
    plane_depths = val_data_cpu["p_d"].reshape((n_bins, 1)).numpy()

    resized_vis = transform.resize(val_data_cpu["vis"].numpy(), (cur_h, cur_w), order=0)
    resized_depth = transform.resize(val_data_cpu["depth"].numpy(), (cur_h, cur_w), order=0)

    mesh_gen_debug = MeshGenerator()

    vertices, triangles, debug_grid_occs = mesh_gen_debug.gen_mesh_reverse_mapping(
        orig_h=orig_h,
        orig_w=orig_w,
        occ_planes=occ_planes,
        plane_depths=plane_depths,
        n_bins=n_bins,
        vis=resized_vis,
        depth=resized_depth,
        orig_to_transformed_order=transform_order,
        crop_info=crop_info,
        resize_info=resize_info,
        cam_matrices=cam_matrices,
        dataset="nons3d",
        smooth_mcube=smooth_mcube,
        use_graph_cut=use_graph_cut,
        graph_cut_info=graph_cut_info,
    )

    assert vertices.shape[0] > 0, f"{rgb_f}"
    assert triangles.shape[0] > 0, f"{rgb_f}"

    if smooth_mcube:
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles[:, [1, 0, 2]])
        # _ = mesh.export(os.path.join(save_dir, SMOOTH_MESH_FNAME))
    else:
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

    _ = mesh.export(save_mesh_f)

    # save mesh in world coordinate system
    world_mesh = copy.deepcopy(mesh)
    cam_verts = np.array(world_mesh.vertices)
    cam_vert_homo = np.concatenate((cam_verts, np.ones_like(cam_verts[:, :1])), axis=1)  # [#points, 4]
    world_vert_homo = np.matmul(cur_obj.inv_extri_mat, cam_vert_homo.T).T
    world_verts = world_vert_homo[:, :3] / (world_vert_homo[:, 3:])
    world_mesh.vertices = world_verts

    save_mesh_f_split = os.path.basename(save_mesh_f).split(".")
    save_mesh_fname = ".".join(save_mesh_f_split[:-1])
    save_mesh_ext = save_mesh_f_split[-1]
    save_mesh_world_f = os.path.join(os.path.dirname(save_mesh_f), f"{save_mesh_fname}_world.{save_mesh_ext}")
    _ = world_mesh.export(save_mesh_world_f)

    # save model input information
    crop_info_f = os.path.join(save_dir, "crop_info_f.pkl")
    with open(crop_info_f, "wb") as f:
        pickle.dump(val_data_cpu["crop_info"], f)

    tmp_im = val_data_cpu["raw_im"].numpy()  # [H, W, 3], range [0, 1]
    tmp_vis = (val_data_cpu["vis"].numpy() * 255).astype(np.uint8)  # [H, W]
    tmp_vis_orig = (val_data_cpu["vis_orig"].numpy() * 255).astype(np.uint8)  # [H, W]

    Image.fromarray(tmp_im).save(os.path.join(save_dir, "image.png"))
    Image.fromarray(tmp_vis).save(os.path.join(save_dir, "vis.png"))
    Image.fromarray(tmp_vis_orig).save(os.path.join(save_dir, "vis_orig.png"))


def gen_mesh_subproc(subproc_input):

    (
        proc_id,
        device_id,
        ckpt_f,
        n_bins,
        rgb_f_list,
        save_root,
        smooth_mcube,
        use_graph_cut,
        graph_cut_info,
        disable_tqdm,
    ) = subproc_input

    device = torch.device(f"cuda:{device_id}")
    print("\ndevice_id: ", device_id, "\n")

    # ckpt_f: /model_root/checkpoints/XXX.pth
    config_f = os.path.join(os.path.dirname(os.path.dirname(ckpt_f)), "config.pth")
    config = Config(init_dict=torch.load(config_f, map_location="cpu")["config"])

    model = OPlanes_Trainer.load_from_checkpoint(
        ckpt_f,
        model_name=config.MODEL.name,
        backbone_name=config.MODEL.backbone,
        pos_enc_kwargs=convert_cfg_to_dict(config.MODEL.pos_enc),
        feat_channels=config.MODEL.feat_channels,
        coeff_ce=config.TRAIN.coeff_ce,
        coeff_ce_pos=config.TRAIN.coeff_ce_pos,
        coeff_ce_neg=config.TRAIN.coeff_ce_neg,
        coeff_dice=config.TRAIN.coeff_dice,
        intermediate_supervise=config.TRAIN.intermediate_supervise,
    )
    model.to(device)
    model.eval()

    cnt = 0
    for rgb_f in tqdm.tqdm(rgb_f_list):

        cur_dir = pathlib.Path(rgb_f).parent.resolve()
        cur_fname = pathlib.Path(rgb_f).stem.split("_")[0]

        cur_save_dir = os.path.join(save_root, f"meshes/{cur_fname}")
        os.makedirs(cur_save_dir, exist_ok=True)

        depth_f = cur_dir / f"{cur_fname}_depth.npz"
        cam_mat_f = cur_dir / f"{cur_fname}_cam_mat.npz"
        mask_f = cur_dir / f"{cur_fname}_mask.png"

        if use_graph_cut:
            gc_bias = graph_cut_info["bias"]
            gc_pair_w = graph_cut_info["pair_weight"]
            gc_suffix = f"_b_{gc_bias}_pair_{gc_pair_w}"
        else:
            gc_suffix = ""

        if smooth_mcube:
            tmp_mesh_f = os.path.join(cur_save_dir, f"{SMOOTH_MESH_FNAME}{gc_suffix}.ply")
        else:
            tmp_mesh_f = os.path.join(cur_save_dir, f"{NONSMOOTH_MESH_FNAME}{gc_suffix}.ply")

        cnt += 1
        # if not os.path.exists(tmp_mesh_f):
        # print(tmp_mesh_f)
        extract_mesh(
            rgb_f=rgb_f,
            depth_f=depth_f,
            mask_f=mask_f,
            cam_mat_f=cam_mat_f,
            device=device,
            model=model,
            config=config,
            save_dir=cur_save_dir,
            n_bins=n_bins,
            smooth_mcube=smooth_mcube,
            disable_tqdm=disable_tqdm,
            save_mesh_f=tmp_mesh_f,
            use_graph_cut=use_graph_cut,
            graph_cut_info=graph_cut_info,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc", type=int, required=False, default=10)
    parser.add_argument("--rgb_f_list", nargs="+", type=str, required=True)
    parser.add_argument("--ckpt_f", type=str, required=True)
    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument("--smooth_mcube", type=int, required=True)
    parser.add_argument("--use_graph_cut", type=int, required=True)
    parser.add_argument("--graph_cut_bias", type=float, default=0.0)
    parser.add_argument("--graph_cut_pair_weight", type=float, default=5)
    parser.add_argument("--n_bins", type=int, required=True)
    args = parser.parse_args()

    num_of_gpus = torch.cuda.device_count()
    print(f"\nFind {num_of_gpus} GPUs.\n")

    ckpt_f = args.ckpt_f

    print("\nckpt_f: ", ckpt_f, "\n")

    nproc = args.nproc

    if nproc > 1:
        disable_tqdm = True
    else:
        disable_tqdm = False

    graph_cut_info = {
        "bias": args.graph_cut_bias,
        "pair_weight": args.graph_cut_pair_weight,
    }

    rgb_f_list = args.rgb_f_list

    np.random.shuffle(rgb_f_list)

    print(f"\nFind {len(rgb_f_list)} files.\n")

    chunk_rgb_f_list = [[] for _ in range(nproc)]
    for i, tmp_f in enumerate(rgb_f_list):
        chunk_rgb_f_list[i % nproc].append(tmp_f)

    device_id_list = [_ % num_of_gpus for _ in range(nproc)]

    print("\ndevice_id_list: ", device_id_list, "\n")

    # NOTE: np.matmul may freeze when using default "fork"
    # https://github.com/ModelOriented/DALEX/issues/412
    with mp.get_context("spawn").Pool(nproc) as pool:
        gathered_ret_dicts = pool.map(
            gen_mesh_subproc,
            zip(
                range(nproc),
                device_id_list,
                [ckpt_f for _ in range(nproc)],
                [args.n_bins for _ in range(nproc)],
                chunk_rgb_f_list,
                [args.save_root for _ in range(nproc)],
                [bool(args.smooth_mcube) for _ in range(nproc)],
                [bool(args.use_graph_cut) for _ in range(nproc)],
                [graph_cut_info for _ in range(nproc)],
                [disable_tqdm for _ in range(nproc)],
            ),
        )
        pool.close()
        pool.join()
