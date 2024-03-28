import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import torch
import torch.nn.functional as F

from model.res_unet import ResUNet
from utils.patches import split_into_patches, combine_from_patches
from config import PATCH_SIZE, CONV_OFFSET, PATCH_OVERLAP

from typing import Callable


class LumenStoneSegmentation(pl.LightningModule):
    def __init__(
        self,
        n_classes: int,
        n_channels: int,
        n_filters: int,
        BN: bool,
        loss_func: Callable,
        optimizer: Callable,
        lr: float,
    ):
        super().__init__()

        self.model = ResUNet(n_classes, n_channels, n_filters, BN)
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.lr = lr

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs, target = batch

        patches = split_into_patches(
            inputs,
            patch_size=PATCH_SIZE,
            conv_offset=CONV_OFFSET,
            overlap=PATCH_OVERLAP,
        )

        logits_patches = []
        for patch in patches:
            logits_patch = self(patch.float())
            logits_patches.append(logits_patch.detach())

        logits = combine_from_patches(
            logits_patches,
            patch_size=PATCH_SIZE,
            conv_offset=CONV_OFFSET,
            overlap=PATCH_OVERLAP,
            src_shape=inputs.shape,
        )

        loss = self.loss_func(logits, target)

        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5
        )

        lr_dict = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
        }

        return [optimizer], [lr_dict]
