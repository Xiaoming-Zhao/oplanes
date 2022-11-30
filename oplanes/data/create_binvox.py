import os
import igl
import sys
import argparse
import tempfile
import multiprocessing
import trimesh
import numpy as np
from joblib import Parallel, delayed

import oplanes.data.binvox_rw as binvox_rw


BINVOX_FNAME = "voxel_256.binvox2"


def create_binvox(obj_in, binvox_out, nBins=256):
    if os.path.isfile(binvox_out):
        return
    obj = trimesh.load(obj_in, process=False)
    minCoord = np.min(obj.vertices, axis=0)
    maxCoord = np.max(obj.vertices, axis=0)
    xRange = np.linspace(minCoord[0], maxCoord[0], nBins)
    yRange = np.linspace(minCoord[1], maxCoord[1], nBins)
    zRange = np.linspace(minCoord[2], maxCoord[2], nBins)
    X, Y, Z = np.meshgrid(xRange, yRange, zRange)
    coords = np.stack([np.reshape(X, -1), np.reshape(Y, -1), np.reshape(Z, -1)], axis=1)
    data = igl.fast_winding_number_for_meshes(obj.vertices, obj.faces, coords) > 0.5
    # data = igl.winding_number(obj.vertices, obj.faces, coords)>0.5
    data = np.reshape(data, X.shape)
    data_test = np.transpose(data, (1, 0, 2))
    axis_order = "xyz"
    dims = [nBins, nBins, nBins]
    scale = nBins / (nBins - 1) * (maxCoord - minCoord)
    translate = minCoord - 0.5 / nBins * scale
    model = binvox_rw.Voxels(data_test, dims, translate, scale, axis_order)

    path = os.path.dirname(binvox_out)
    os.makedirs(path, exist_ok=True)
    with open(binvox_out, "wb") as fp:
        binvox_rw.write(model, fp)
    return 0


def process(data_root, save_root, id, obj_raw, tmpdir, exe_path):
    """
    Process the mesh (path: obj_raw)
    """
    # tmp name for the watertight mesh
    obj_watertight = tmpdir + "/" + next(tempfile._get_candidate_names()) + "_" + str(id) + ".obj"

    # binvox file path
    binvox_out = os.path.join(save_root, obj_raw[2:].replace("mesh.obj", BINVOX_FNAME))

    if os.path.exists(binvox_out):
        return

    obj_raw = os.path.join(data_root, obj_raw[2:])

    # create watertight mesh
    os.system(f"{exe_path} --input {obj_raw} --output {obj_watertight}")

    # create binvox file with resolution 256
    create_binvox(obj_watertight, binvox_out, 256)

    # remove the intermediate watertight mesh file
    os.system(f"rm {obj_watertight}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifoldplus_exe_f", type=str, required=False)
    parser.add_argument("--process_f", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=False)
    parser.add_argument("--save_root", type=str, required=False)
    args = parser.parse_args()

    # use ManifoldPlus to get the watertight mesh
    exe_path = args.manifoldplus_exe_f

    tmpdir = tempfile.gettempdir()

    data_root = args.data_root
    save_root = args.save_root

    process_f = args.process_f

    f = open(process_f)
    process_f_list = []
    for ln in f:
        process_f_list.append(ln.rstrip())

    # multiprocessing version
    num_cores = int(multiprocessing.cpu_count() / 4)
    print("using " + str(num_cores))
    print(len(process_f_list))
    Parallel(n_jobs=num_cores)(
        delayed(process)(data_root, save_root, idx, obj_raw, tmpdir, exe_path)
        for idx, obj_raw in enumerate(process_f_list)
    )

    # # single-process version
    # for idx, obj_raw in enumerate(process_f_list):
    #     obj_raw = ln.rstrip()
    #     process(data_root, save_root, idx, obj_raw, tmpdir, exe_path)
