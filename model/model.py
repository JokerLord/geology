import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import torch
import torch.nn.functional as F

from model.res_unet import ResUNet
from utils.patches import split_into_patches, combine_from_patches

from typing import Callable


class LumenStoneSegmentation(pl.LightningModule):
    def __init__(
        self,
        n_classes: int,
        n_channels: int,
        n_filters: int,
        BN: bool,
        patch_size: int,
        batch_size: int,
        offset: int,
        patch_overlap: int,
        loss_func: Callable,
        optimizer: Callable,
        lr: float,
    ):
        super().__init__()

        self.model = ResUNet(n_classes, n_channels, n_filters, BN)

        self.patch_size = patch_size
        self.batch_size = batch_size
        self.offset = offset
        self.patch_overlap = patch_overlap

        self.loss_func = loss_func
        self.optimizer = optimizer
        self.lr = lr

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        inputs = inputs[0]
        target = target[0]

        patches = split_into_patches(
            inputs,
            patch_size=self.patch_size,
            offset=self.offset,
            overlap=self.patch_overlap,
        )
        init_n_patches = len(patches)

        while (len(patches) % self.batch_size != 0):
            patches.append(patches[-1])

        logits_patches = []
        for i in range(0, len(patches), self.batch_size):
            batch = torch.stack(patches[i: i + self.batch_size])
            logits_batch = self(batch.float())

            print(torch.cuda.memory_allocated() / 1024 ** 3)
            print(torch.cuda.max_memory_allocated() / 1024 ** 3)
            for logits_patch in logits_batch:
                logits_patches.append(logits_patch)

        logits_patches = logits_patches[:init_n_patches]

        logits = combine_from_patches(
            logits_patches,
            patch_size=self.patch_size,
            offset=self.offset,
            overlap=self.patch_overlap,
            src_shape=inputs.shape[-2:],
        )
        logits = logits[None, ...]
        target = target[None, ...]

        loss = self.loss_func(logits, target.long())

        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
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
            src_shape=inputs.shape[-2:],
        )
        loss = self.loss_func(logits, target.long())

        pred = torch.argmax(F.softmax(logits, dim=1), dim=1)

        return {"val_loss": loss}

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
