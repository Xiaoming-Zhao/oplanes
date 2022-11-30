import os
import pathlib
import argparse
import datetime
import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    ProgressBar,
)

from oplanes.data.meshdata import MyData
from oplanes.trainer.oplanes_trainer import OPlanes_Trainer
from oplanes.utils.config import (
    Config,
    get_config,
    convert_cfg_to_dict,
    update_config_log,
)
from oplanes.utils.pl_utils import EpochProgressBar
from oplanes.utils.registry import registry


def _get_dataset(config, repo_path, is_val=False, is_dryrun=False):
    # ImageNet's mean and std
    pixel_mean = np.array([123.675, 116.280, 103.530]).reshape((1, 1, 3))
    pixel_std = np.array([58.395, 57.120, 57.375]).reshape((1, 1, 3))

    dryrun_mode = "dryrun" if is_dryrun else "full"

    if is_val:
        split_f = config.DATASET.val[dryrun_mode]
    else:
        split_f = config.DATASET.train[dryrun_mode]

    split_f = str(repo_path / split_f)

    print("\nsplit_f: ", split_f, "\n")

    mesh_data_root = str(repo_path / config.DATASET.mesh_data_root)
    binvox_path_prefix = str(repo_path / config.DATASET.binvox_path)

    print("\nmesh_data_root: ", mesh_data_root, "\n")
    print("\nbinvox_path_prefix: ", binvox_path_prefix, "\n")

    cur_set = MyData(
        fn=split_f,
        threshold=0.5,
        is_val=is_val,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        max_depth_range=config.TRAIN.PROCESS_DATA.max_depth_range,
        n_planes_for_train=config.TRAIN.PROCESS_DATA.n_planes_for_train,
        n_planes_for_val=config.TRAIN.PROCESS_DATA.n_planes_for_val,
        mesh_data_root=mesh_data_root,
        data_h=config.TRAIN.PROCESS_DATA.data_h,
        data_w=config.TRAIN.PROCESS_DATA.data_w,
        crop_expand_ratio=config.TRAIN.PROCESS_DATA.crop_expand_ratio,
        binvox_path_prefix=binvox_path_prefix,
        depth_range_expand_ratio=config.TRAIN.PROCESS_DATA.depth_range_expand_ratio,
        use_masked_out_img=config.TRAIN.PROCESS_DATA.use_masked_out_img,
    )

    return cur_set


def main(args):

    repo_path = pathlib.Path(__file__).parent.parent.resolve()

    config = get_config(args.config_f, None)

    config.defrost()
    config.DATASET.binvox_path = config.DATASET.binvox_path_dict[config.DATASET.binvox_type]
    config.LOG_DIR = str(repo_path / config.LOG_DIR)
    config.freeze()

    if args.resume_ckpt is not None:
        print(f"\nResume from {args.resume_ckpt}\m")

        config_f = os.path.join(os.path.dirname(os.path.dirname(args.resume_ckpt)), "config.pth")
        config = Config(init_dict=torch.load(config_f, map_location="cpu")["config"])
        config.freeze()

        default_log_root = os.path.dirname(config.LOG_DIR)

        config.defrost()
        config.LOG_DIR = default_log_root

        config.freeze()

    cur_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
    log_folder_name = "seed_{}-{}-{}-n_gpu_{}_{}-feat_{}-inter_loss_{}-mask_{}-ptrain_{}_pval_{}-max_d_range_{}_{}-h_{}_w_{}-lr_{}-ce_{}_{}_{}-dice_{}-{}".format(
        config.SEED,
        config.MODEL.name,
        config.MODEL.backbone,
        config.TRAIN.n_gpus,
        config.TRAIN.batch_size,
        config.MODEL.feat_channels,
        int(config.TRAIN.intermediate_supervise),
        int(config.TRAIN.PROCESS_DATA.use_masked_out_img),
        config.TRAIN.PROCESS_DATA.n_planes_for_train,
        config.TRAIN.PROCESS_DATA.n_planes_for_val,
        config.TRAIN.PROCESS_DATA.max_depth_range,
        config.TRAIN.PROCESS_DATA.depth_range_expand_ratio,
        config.TRAIN.PROCESS_DATA.data_h,
        config.TRAIN.PROCESS_DATA.data_w,
        config.TRAIN.learning_rate,
        config.TRAIN.coeff_ce,
        config.TRAIN.coeff_ce_pos,
        config.TRAIN.coeff_ce_neg,
        config.TRAIN.coeff_dice,
        cur_time,
    )
    log_dir = os.path.join(config.LOG_DIR, log_folder_name)

    run_type = "train"
    config = update_config_log(config, run_type, log_dir)

    torch.save(
        {"config": convert_cfg_to_dict(config)},
        os.path.join(config.LOG_DIR, "config.pth"),
    )
    print("\nconfig: ", config, "\n")
    with open(os.path.join(config.LOG_DIR, "config.txt"), "w") as f:
        f.write(f"{config}")

    pl.seed_everything(config.SEED, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    trainset = _get_dataset(config, repo_path, is_val=False, is_dryrun=bool(args.dryrun))
    trainloader = DataLoader(
        trainset,
        batch_size=config.TRAIN.batch_size,
        shuffle=True,
        num_workers=config.TRAIN.num_workers,
    )

    valset = _get_dataset(config, repo_path, is_val=True, is_dryrun=bool(args.dryrun))
    valloader = DataLoader(valset, batch_size=1, num_workers=config.TRAIN.num_workers)

    oplanes_trainer = OPlanes_Trainer(
        model_name=config.MODEL.name,
        backbone_name=config.MODEL.backbone,
        pos_enc_kwargs=convert_cfg_to_dict(config.MODEL.pos_enc),
        feat_channels=config.MODEL.feat_channels,
        trainset=trainset,
        learning_rate=config.TRAIN.learning_rate,
        coeff_ce=config.TRAIN.coeff_ce,
        coeff_ce_pos=config.TRAIN.coeff_ce_pos,
        coeff_ce_neg=config.TRAIN.coeff_ce_neg,
        coeff_dice=config.TRAIN.coeff_dice,
        intermediate_supervise=config.TRAIN.intermediate_supervise,
    )

    logger = TensorBoardLogger(config.TENSORBOARD_DIR, name="oplanes_train_s3d") if args.log != 0 else None

    if logger is None:
        callbacks = []
    else:
        callbacks = [
            LearningRateMonitor(logging_interval="step", log_momentum=True),
        ]
        callbacks = callbacks + [
            ModelCheckpoint(
                save_top_k=-1,
                save_last=True,
                monitor="val/acc",
                mode="max",
                dirpath=config.CHECKPOINT_FOLDER,
                filename="model-oplanes-s3d-epoch={epoch:02d}-val_acc={val/acc:.4f}",
                auto_insert_metric_name=False,
            ),
        ]
        if args.val_only:
            callbacks = callbacks + [ProgressBar()]
        else:
            callbacks = callbacks + [
                EpochProgressBar(),
            ]

    if config.TRAIN.n_gpus > 1:
        strategy = "ddp"
    else:
        strategy = None
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=config.TRAIN.n_gpus,
        strategy=strategy,
        max_epochs=config.TRAIN.n_epochs,
        profiler="simple",
        check_val_every_n_epoch=1,
        reload_dataloaders_every_epoch=False,
        reload_dataloaders_every_n_epochs=0,
        logger=logger,
        log_every_n_steps=5,
        progress_bar_refresh_rate=10,
        deterministic=False,
        callbacks=callbacks,
    )

    if args.val_only:
        trainer.validate(
            model=oplanes_trainer,
            dataloaders=valloader,
            ckpt_path=args.val_ckpt_path,
            verbose=True,
        )
    else:
        if args.resume_ckpt is None:
            trainer.fit(model=oplanes_trainer, train_dataloaders=trainloader, val_dataloaders=valloader)
        else:
            trainer.fit(
                model=oplanes_trainer,
                train_dataloaders=trainloader,
                val_dataloaders=valloader,
                ckpt_path=args.resume_ckpt,
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_f", type=str, default="./configs/default.yml")
    parser.add_argument("--val_only", action="store_true")
    parser.add_argument("--val_ckpt_path", type=str, default="./val.ckpt")
    parser.add_argument("--log", type=int, default=1)
    parser.add_argument("--dryrun", type=int, default=0)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    args = parser.parse_args()

    main(args)
