import os
import tqdm
import glob
import joblib
import pickle
import trimesh
import argparse
import numpy as np
import multiprocessing as mp
from PIL import Image
from skimage import transform

import torch

from oplanes.data.meshdata import MeshObj
from oplanes.trainer.oplanes_trainer import OPlanes_Trainer
from oplanes.utils.config import Config, get_config, convert_cfg_to_dict
from oplanes.models.mesh_generator import MeshGenerator


pixel_mean = np.array([123.675, 116.280, 103.530]).reshape((1, 1, 3))
pixel_std = np.array([58.395, 57.120, 57.375]).reshape((1, 1, 3))

HEURISTIC_DEPTH_RANGE = 2.0

SMOOTH_MESH_FNAME = "pred_smooth.ply"
NONSMOOTH_MESH_FNAME = "pred_non_smooth.ply"


def extract_mesh(
    device,
    model,
    cur_line,
    config,
    save_dir,
    mesh_data_root,
    binvoxPathPrefix="",
    n_bins=128,
    smooth_mcube=False,
    given_crop_info_root=None,
    disable_tqdm=False,
):

    given_depth_range = HEURISTIC_DEPTH_RANGE  # use None for GT depth range

    cur_obj = MeshObj(cur_line, mesh_data_root, binvoxPathPrefix, pure_infer=True)

    # We use GT's crop information to ensure every quantitative result is comparable
    given_crop_info_dir = os.path.join(given_crop_info_root, cur_line[2:-8])
    given_crop_info_f = os.path.join(given_crop_info_dir, "crop_info_f.pkl")
    assert os.path.exists(given_crop_info_f), f"{given_crop_info_f}"

    with open(given_crop_info_f, "rb") as f:
        given_crop_info = pickle.load(f)

    val_data_cpu = cur_obj.load_data(
        num_depth=n_bins,
        is_val=True,
        pure_infer=True,
        use_masked_out_img=config.TRAIN.PROCESS_DATA.use_masked_out_img,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        given_depth_range=given_depth_range,
        for_debug=False,
        given_crop_info=given_crop_info,
    )

    for k in given_crop_info:
        assert (
            given_crop_info[k] == val_data_cpu["crop_info"][k]
        ), f"{k}, {given_crop_info[k]}, {val_data_cpu['crop_info'][k]}"

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
    orig_h, orig_w = np.load(cur_obj.visfn).shape

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
    cam_matrices = {"cam_to_ndc": cur_obj.P, "ndc_to_cam": cur_obj.P_inverse}

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
        dataset="s3d",
        smooth_mcube=smooth_mcube,
    )

    assert vertices.shape[0] > 0, f"{cur_line}"
    assert triangles.shape[0] > 0, f"{cur_line}"

    if smooth_mcube:
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles[:, [1, 0, 2]])
        _ = mesh.export(os.path.join(save_dir, SMOOTH_MESH_FNAME))
    else:
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        _ = mesh.export(os.path.join(save_dir, NONSMOOTH_MESH_FNAME))

    # save model input information
    crop_info_f = os.path.join(save_dir, "crop_info_f.pkl")
    with open(crop_info_f, "wb") as f:
        pickle.dump(given_crop_info, f)

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
        chunk_lines,
        data_root,
        save_root_dir,
        smooth_mcube,
        given_crop_info_root,
        disable_tqdm,
    ) = subproc_input
    device = torch.device(f"cuda:{device_id}")
    print("\ndevice_id: ", device_id, "\n")

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
    for cur_line in tqdm.tqdm(chunk_lines):
        cur_save_dir = os.path.join(save_root_dir, cur_line[2:-8])
        os.makedirs(cur_save_dir, exist_ok=True)

        if smooth_mcube:
            tmp_mesh_f = os.path.join(cur_save_dir, SMOOTH_MESH_FNAME)
        else:
            tmp_mesh_f = os.path.join(cur_save_dir, NONSMOOTH_MESH_FNAME)

        cnt += 1

        # if not os.path.exists(tmp_mesh_f):
        extract_mesh(
            device,
            model,
            cur_line,
            config,
            cur_save_dir,
            data_root,
            n_bins=n_bins,
            smooth_mcube=smooth_mcube,
            given_crop_info_root=given_crop_info_root,
            disable_tqdm=disable_tqdm,
        )


def check_mesh_exist(cur_line, save_root_dir, smooth_mcube):
    cur_save_dir = os.path.join(save_root_dir, cur_line[2:-8])
    if smooth_mcube:
        tmp_mesh_f = os.path.join(cur_save_dir, SMOOTH_MESH_FNAME)
    else:
        tmp_mesh_f = os.path.join(cur_save_dir, NONSMOOTH_MESH_FNAME)

    if os.path.exists(tmp_mesh_f):
        try:
            tmp = trimesh.load(tmp_mesh_f, process=False)
            assert tmp.vertices.shape[0] > 0
            assert tmp.faces.shape[0] > 0
        except:
            return False
        return True
    else:
        return False


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc", type=int, required=False, default=10)
    parser.add_argument("--ckpt_f", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--split_f", type=str, required=True)
    parser.add_argument("--smooth_mcube", type=int, required=True)
    parser.add_argument("--n_bins", type=int, required=True)
    parser.add_argument("--given_crop_info_root", type=str, required=True)
    args = parser.parse_args()

    num_of_gpus = torch.cuda.device_count()
    print(f"\nFind {num_of_gpus} GPUs.\n")

    ckpt_f = args.ckpt_f
    print("\nckpt_f: ", ckpt_f, "\n")

    split_fname = os.path.basename(args.split_f).split(".")[0]
    save_dir = os.path.join(os.path.dirname(os.path.dirname(ckpt_f)), f"eval/{args.n_bins}", split_fname)

    print("\nsave_dir: ", save_dir, "\n")

    given_crop_info_root = os.path.join(args.given_crop_info_root, split_fname)

    nproc = args.nproc

    if nproc > 1:
        disable_tqdm = True
    else:
        disable_tqdm = False

    with open(args.split_f) as f:
        raw_all_lines = f.read().splitlines()

    all_lines = []
    for tmp in tqdm.tqdm(raw_all_lines):
        if not check_mesh_exist(tmp, save_dir, bool(args.smooth_mcube)):
            all_lines.append(tmp)

    np.random.shuffle(all_lines)

    print(f"\nFind {len(all_lines)} files.\n")

    if len(all_lines) > 0:

        f_chunk = [[] for _ in range(nproc)]
        for i, tmp_f in enumerate(all_lines):
            f_chunk[i % nproc].append(tmp_f)

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
                    f_chunk,
                    [args.data_root for _ in range(nproc)],
                    [save_dir for _ in range(nproc)],
                    [bool(args.smooth_mcube) for _ in range(nproc)],
                    [given_crop_info_root for _ in range(nproc)],
                    [disable_tqdm for _ in range(nproc)],
                ),
            )
            pool.close()
            pool.join()
