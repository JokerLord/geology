import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Subset, DataLoader
from torch import Tensor

from res_unet import ResUNet
from dataset import LumenStoneDataset
from utils import split_into_patches, combine_from_patches, one_hot
from config import *
from metrics import iou_per_class, accuracy

import random
from sklearn.model_selection import train_test_split
from typing import Callable


class Trainer:
    def __init__(self, model, criterion, optimizer, device, max_epochs) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.max_epochs = max_epochs

        self._train_dataset = LumenStoneDataset(
            root_dir=LUMENSTONE_PATH, train=True, transform=TRAIN_TRANSFORM
        )
        self._val_dataset = LumenStoneDataset(
            root_dir=LUMENSTONE_PATH, train=True, transform=VAL_TRANSFORM
        )


    def _split_dataset(self) -> tuple[DataLoader, DataLoader]:
        indices = list(range(len(self._train_dataset)))
        train_indices, val_indices = train_test_split(
            indices, train_size=SPLIT_RATIO, shuffle=True
        )
        train_dataset = Subset(self._train_dataset, train_indices)
        val_dataset = Subset(self._val_dataset, val_indices)
        train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=4)
        return train_dataloader, val_dataloader


    def _train_image(self, batch: tuple[Tensor, Tensor]) -> dict[str, float]:
        inputs, target = batch # (1, 3, H, W), (1, H, W)
        inputs, target = torch.squeeze(inputs), torch.squeeze(target) # (3, H, W), (H, W)

        inputs_patches = split_into_patches(
            image=inputs, patch_size=PATCH_SIZE, offset=OFFSET, overlap=PATCH_OVERLAP
        )
        target_patches = split_into_patches(
            image=target, patch_size=PATCH_SIZE, offset=OFFSET, overlap=PATCH_OVERLAP
        )

        """ Make total number of patches a multiple of BATCH_SIZE """
        while len(inputs_patches) % BATCH_SIZE != 0:
            inputs_patches.append(inputs_patches[-1])
            target_patches.append(target_patches[-1])

        """ Shuffle patches """
        zipped_patches = list(zip(inputs_patches, target_patches))
        random.shuffle(zipped_patches)
        inputs_patches, target_patches = zip(*zipped_patches)

        losses = []
        for i in range(0, len(inputs_patches), BATCH_SIZE):
            inputs_batch = torch.stack(inputs_patches[i : i + BATCH_SIZE]).to(
                device=self.device, dtype=torch.float32
            ) # (batch_size, 3, patch_size, patch_size)
            logits_batch = self.model(inputs_batch)  # (batch_size, n_classes, patch_size, patch_size)
            target_batch = torch.stack(target_patches[i : i + BATCH_SIZE]).to(
                device=self.device, dtype=torch.int64
            )  # (batch_size, patch_size, patch_size)
            loss = self.criterion(logits_batch, target_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        avg_loss = sum(losses) / len(losses)
        return {"train_loss": avg_loss}

    
    def _validate_image(self, batch: tuple[Tensor, Tensor]) -> dict[str, float]:
        inputs, target = batch # (1, 3, H, W), (1, H, W)
        inputs, target = torch.squeeze(inputs), torch.squeeze(target) # (3, H, W), (H, W)

        inputs_patches = split_into_patches(
            image=inputs, patch_size=PATCH_SIZE, offset=OFFSET, overlap=PATCH_OVERLAP
        )
        target_patches = split_into_patches(
            image=target, patch_size=PATCH_SIZE, offset=OFFSET, overlap=PATCH_OVERLAP
        )

        """ Make total number of patches a multiple of BATCH_SIZE """
        init_n_patches = len(inputs_patches)
        while len(inputs_patches) % BATCH_SIZE != 0:
            inputs_patches.append(inputs_patches[-1])
            target_patches.append(target_patches[-1])

        with torch.no_grad():
            logits_patches = []
            losses = []
            for i in range(0, len(inputs_patches), BATCH_SIZE):
                inputs_batch = torch.stack(inputs_patches[i : i + BATCH_SIZE]).to(
                    device=self.device, dtype=torch.float32
                ) # (batch_size, 3, patch_size, patch_size) in GPU memory
                logits_batch = self.model(inputs_batch).cpu()  # (batch_size, n_classes, patch_size, patch_size) in CPU memory
                target_batch = torch.stack(target_patches[i: i + BATCH_SIZE]).to(
                    dtype=torch.int64
                ) # (batch_size, patch_size, patch_size)

                loss = self.criterion(logits_batch, target_batch)
                losses.append(loss.item())

                for logits_patch in logits_batch:
                    logits_patches.append(logits_patch)
        avg_loss = sum(losses) / len(losses)

        logits_patches = logits_patches[:init_n_patches]
        logits = combine_from_patches(
            logits_patches,
            patch_size=PATCH_SIZE,
            offset=OFFSET,
            overlap=PATCH_OVERLAP,
            src_shape=inputs.shape[-2:],
        ) # (n_classes, H, W) in CPU memory
    
        activated = F.softmax(input=logits, dim=1) # (n_classes, H, W)
        y_true = one_hot(target, n_classes=len(CLASS_NAMES))
        iou_per_class(y_true, activated)

        mask = torch.argmax(activated, dim=0) # (H, W)
        prediction = one_hot(mask, n_classes=len(CLASS_NAMES))

        

    def fit(self):
        self.model.to(self.device)
        for epoch in range(self.max_epochs):
            train_dataloader, val_dataloader = self._split_dataset()

            self.model.train()
            train_outputs = []
            for batch in train_dataloader:
                train_outputs.append(self._train_image(batch))

            self.model.eval()
            val_outputs = []
            for batch in val_dataloader:
                val_outputs.append(self._validate_image(batch))
