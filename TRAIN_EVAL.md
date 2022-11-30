
<h1 align="center">Training and Evaluation</h1>

## Table of Contents

- [Dataset](#dataset-preparation)
  - [Download Raw Data](#download-raw-data)
  - [Install ManifoldPlus](#install-manifoldPlus)
  - [Generate Binvox Files](#generate-binvox-files)
- [Training](#training)
- [Evaluation](#evaluation)
  - [Install ONet](#install-onet)
  - [Sample GT Points](#sample-gt-points)
  - [Generate Mesh for S3D](#generate-mesh-for-s3d)
  - [Run Evaluation](#run-evaluation)

## Dataset Preparation

```bash
cd /path/to/this/repo
export OPLANES_ROOT=$PWD
```

### Download Raw Data

OPlanes are trained on [SAIL-VOS 3D (S3D)](https://sailvos.web.illinois.edu/_site/_site/index.html). Please download the dataset following instructions on the website.

Place the downloaded dataset at `${OPLANES_ROOT}/datasets/s3d/raw`.

### Install ManifoldPlus

We use [ManifoldPlus](https://github.com/hjwdzh/ManifoldPlus) to create a watertight mesh from the raw mesh. Please follow the official instruction to install the package.

You can install it anywhere you prefer. Here we install it to `${OPLANES_ROOT}/external` for illustartion purpose:
```bash
cd ${OPLANES_ROOT}/external

git clone https://github.com/hjwdzh/ManifoldPlus.git

export ManfioldPlusRoot=${OPLANES_ROOT}/external/ManifoldPlus

cd ${ManfioldPlusRoot}
git submodule update --init --recursive

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

### Generate Binvox Files

Since it is time-consuming to generate binvox files for all meshes, here we use `train_dryrun.txt` and `val_dryrun.txt` to illustrate how to run the command.

Please replace them with `train_27588.txt` and `val_4300.txt` for full training and evaluation.

The generated binvox files will be saved in `${OPLANES_ROOT}/datasets/s3d/binvox_256`.

```bash
cd ${OPLANES_ROOT}

# for train split
conda activate oplanes && \
export PYTHONPATH=${OPLANES_ROOT}:$PYTHONPATH && \
python ${OPLANES_ROOT}/oplanes/data/create_binvox.py \
--manifoldplus_exe_f ${ManfioldPlusRoot}/build/manifold \
--process_f ${OPLANES_ROOT}/datasets/s3d/splits/train_dryrun.txt \
--data_root ${OPLANES_ROOT}/datasets/s3d/raw \
--save_root ${OPLANES_ROOT}/datasets/s3d/binvox_256

# for val split
conda activate oplanes && \
export PYTHONPATH=${OPLANES_ROOT}:$PYTHONPATH && \
python ${OPLANES_ROOT}/oplanes/data/create_binvox.py \
--manifoldplus_exe_f ${ManfioldPlusRoot}/build/manifold \
--process_f ${OPLANES_ROOT}/datasets/s3d/splits/val_dryrun.txt \
--data_root ${OPLANES_ROOT}/datasets/s3d/raw \
--save_root ${OPLANES_ROOT}/datasets/s3d/binvox_256
```

## Training

The following command will train OPlanes models. If you have generated full binvox files in the above step, set `--dryrun 0` instead.

Checkpoints will be saved in `${OPLANES_ROOT}/exps`.

```bash
conda activate oplanes && \
export NCCL_P2P_DISABLE=1 && \
export PYTHONPATH=${OPLANES_ROOT}:$PYTHONPATH && \
python ${OPLANES_ROOT}/oplanes/run.py \
--config_f ${OPLANES_ROOT}/configs/default.yml \
--dryrun 1 \
--log 1
```

## Evaluation

### Install ONet

We utilize [Occupancy Network (ONet)](https://github.com/autonomousvision/occupancy_networks)'s utility functions. Therefore, we need to install ONet.

For ONet's utility functions, we modify two things:

1. We comment out the specific requirements for GPU architecture in [`libfusiongpu/CMakeLists.txt`](./external/onet/external/mesh-fusion/libfusiongpu/CMakeLists.txt) such that it can be compiled against more GPU categories.
2. Following [suggestions from this answer](https://github.com/mcfletch/pyopengl/issues/11#issuecomment-522440708), we modify [`libfusiongpu/cyfusion.cpp`](./external/onet/external/mesh-fusion/libfusiongpu/cyfusion.cpp), [`librender/pyrender.cpp`](./external/onet/external/mesh-fusion/librender/pyrender.cpp), [`libmcubes/mcubes.cpp`](./external/onet/external/mesh-fusion/libmcubes/mcubes.cpp).

Please follow the [official instructions](https://github.com/autonomousvision/occupancy_networks/tree/406f79468fb8b57b3e76816aaa73b1915c53ad22/external/mesh-fusion#installation) to install it:

```bash
# setup ONet
cd ${OPLANES_ROOT}/external/onet
python setup.py build_ext --inplace

cd ${OPLANES_ROOT}/external/onet/external/mesh-fusion

# build pyfusion
# use libfusioncpu alternatively!
cd libfusiongpu
mkdir build
cd build
cmake ..
make
cd ..
python setup.py build_ext --inplace

cd ..
# build PyMCubes
cd libmcubes
python setup.py build_ext --inplace
```

### Sample GT Points

We sample ground-truth (GT) points and save them to disk for fast evaluation.

Points and information for evaluation will be saved in `${OPLANES_ROOT}/datasets/s3d/gt_pts_for_eval`.

```bash
cd ${OPLANES_ROOT}

conda activate oplanes && \
export PYTHONPATH=${OPLANES_ROOT}:$PYTHONPATH && \
export NCCL_P2P_DISABLE=1 && \
python ${OPLANES_ROOT}/oplanes/eval/sample_gt_points.py \
--nproc 1 \
--data_root ${OPLANES_ROOT}/datasets/s3d/raw \
--save_root_dir ${OPLANES_ROOT}/datasets/s3d/gt_pts_for_eval \
--bin_f ${ManfioldPlusRoot}/build/manifold \
--split_f ${OPLANES_ROOT}/datasets/s3d/splits/val_dryrun.txt \
--is_1st_time 1
```

Please replace `val_dryrun.txt` with `val_4300.txt` for full evaluation.

### Generate Mesh for S3D

We first generate meshes for S3D.

Please set `--ckpt_f` to a checkpoint file you prefer. Here we use a pretrained checkpoint for the illustration purpose.

Generated meshes will be saved in a folder `eval` in the same directory as the checkpoint file. In this example case, it is `${OPLANES_ROOT}/ckpts/pretrained/seed_0/eval`.

Note, we do not use any post-processing when evaluting for quantitative results. Namely, we set `--smooth_mcube 0`.

```bash
cd ${OPLANES_ROOT}

conda activate oplanes && \
export PYTHONPATH=${OPLANES_ROOT}:$PYTHONPATH && \
python ${OPLANES_ROOT}/oplanes/eval/gen_mesh.py \
--nproc 1 \
--data_root ${OPLANES_ROOT}/datasets/s3d/raw \
--ckpt_f ${OPLANES_ROOT}/ckpts/pretrained/seed_0/checkpoints/model-oplanes-s3d-epoch=14-val_acc=0.9204.ckpt \
--smooth_mcube 0 \
--n_bins 256 \
--given_crop_info_root ${OPLANES_ROOT}/datasets/s3d/gt_pts_for_eval \
--split_f ${OPLANES_ROOT}/datasets/s3d/splits/val_dryrun.txt

```

Please replace `val_dryrun.txt` with `val_4300.txt` for full evaluation.


### Run Evaluation

Results will be saved in `${OPLANES_ROOT}/ckpts/pretrained/seed_0/eval/results/`.

A binary file `eval_dicts_all.pt` stores all the results. You can load it as `joblib.load(eval_dicts_all.pt)`.

```bash
conda activate oplanes && \
export PYTHONPATH=${OPLANES_ROOT}:$PYTHONPATH && \
python ${OPLANES_ROOT}/oplanes/eval/eval_meshes.py \
--nproc 1 \
--ckpt_f ${OPLANES_ROOT}/ckpts/pretrained/seed_0/checkpoints/model-oplanes-s3d-epoch=14-val_acc=0.9204.ckpt \
--n_bins 256 \
--gt_dir ${OPLANES_ROOT}/datasets/s3d/gt_pts_for_eval \
--split_f ${OPLANES_ROOT}/datasets/s3d/splits/val_dryrun.txt
```

Please replace `val_dryrun.txt` with `val_4300.txt` for full evaluation.