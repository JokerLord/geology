import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchmetrics import Accuracy, JaccardIndex

import torch
import torch.nn.functional as F

from res_unet import ResUNet
from utils import split_into_patches, combine_from_patches

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
        patch_overlap: float,
        loss_func: Callable,
        optimizer: Callable,
        lr: float,
        gpu_index: int,
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

        self.step_outputs = {
            "loss": [],
            "accuracy": [],
            "iou": [],
        }

        device = torch.device(f"cuda:{gpu_index}")

        self.metrics = {
            "accuracy": Accuracy(task="multiclass",
                                 threshold=0.5,
                                 num_classes=n_classes,
                                 validate_args=False,
                                 ignore_index=None,
                                 average="micro").to(device),
            "iou": JaccardIndex(task="multiclass",
                                threshold=0.5,
                                num_classes=n_classes,
                                validate_args=False,
                                ignore_index=None,
                                average="macro").to(device),
        }

    def forward(self, inputs):
        return self.model(inputs)

    def shared_step(self, batch, stage: str) -> torch.Tensor:
        print("NEW STEP")
        print(stage)
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
            logits_batch = self(batch.to(torch.float32))
            print(i)
            print(torch.cuda.memory_allocated() / 1024 ** 3)
            # print(torch.cuda.max_memory_allocated() / 1024 ** 3)

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

        activated = F.softmax(input=logits, dim=1)
        predictions = torch.argmax(activated, dim=1)

        loss = self.loss_func(logits, target.to(torch.int64))

        accuracy = self.metrics["accuracy"](predictions, target)
        iou = self.metrics["iou"](predictions, target)

        self.step_outputs["loss"].append(loss)
        self.step_outputs["accuracy"].append(accuracy)
        self.step_outputs["iou"].append(iou)

        return loss

    def shared_epoch_end(self, stage: str):
        loss = torch.mean(torch.tensor([
            loss for loss in self.step_outputs["loss"]
        ]))

        accuracy = torch.mean(torch.tensor([
            accuracy for accuracy in self.step_outputs["accuracy"]
        ]))

        iou = torch.mean(torch.tensor([
            iou for iou in self.step_outputs["iou"]
        ]))

        for key in self.step_outputs.keys():
            self.step_outputs[key].clear()

        metrics = {
            f"{stage}_loss": loss,
            f"{stage}_accuracy": accuracy,
            f"{stage}_iou": iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch=batch, stage="train")

    def on_train_epoch_end(self):
        return self.shared_epoch_end(stage="train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch=batch, stage="val")

    def on_validation_epoch_end(self):
        return self.shared_epoch_end(stage="val")

    def step_step(self, batch, batch_idx):
        return self.shared_step(batch=batch, stage="test")

    def on_test_epoch_end(self):
        return self.shared_epoch_end(stage="test")
    
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5
        )

        scheduler_dict = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "monitor": "val_loss",
        }

        optim_dict = {"optimizer": optimizer, "lr_shedular": scheduler_dict}
        return optim_dict
