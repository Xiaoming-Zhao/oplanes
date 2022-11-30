from tqdm import tqdm

import torch
import pytorch_lightning as pl


class EpochProgressBar(pl.callbacks.progress.TQDMProgressBar):

    """
    https://github.com/PyTorchLightning/pytorch-lightning/issues/2189#issuecomment-841264461
    """

    def __init__(self):
        super().__init__()
        self.bar = None

    def on_train_start(self, trainer, pl_module) -> None:
        self.main_progress_bar = self.init_train_tqdm()
        self.global_bar = tqdm(
            desc="Global Epochs",
            leave=False,
            dynamic_ncols=True,
            total=trainer.max_epochs,
        )

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        super().on_train_epoch_end(trainer, pl_module)
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/2189#issuecomment-843696725
        print()
        self.global_bar.update(1)
