import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Subset, DataLoader
from torch import Tensor

from res_unet import ResUNet
from dataset import LumenStoneDataset
from utils import *
from config import *
from metrics import iou_per_class, accuracy

import random
from sklearn.model_selection import train_test_split
from typing import Callable
from tqdm.auto import tqdm


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device, max_epochs) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
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


    def _train_image(self, inputs: Tensor, target: Tensor) -> dict[str, float]:
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

    
    def _validate_image(self, inputs: Tensor, target: Tensor) -> dict:
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
                logits_batch = self.model(inputs_batch)  # (batch_size, n_classes, patch_size, patch_size) in GPU memory
                target_batch = torch.stack(target_patches[i: i + BATCH_SIZE]).to(
                    device=self.device, dtype=torch.int64
                ) # (batch_size, patch_size, patch_size) in GPU memory

                loss = self.criterion(logits_batch, target_batch)
                losses.append(loss.item())

                for logits_patch in logits_batch:
                    logits_patches.append(logits_patch)

        output_dict = {"val_loss": sum(losses) / len(losses)}

        logits_patches = logits_patches[:init_n_patches]
        logits = combine_from_patches(
            logits_patches,
            patch_size=PATCH_SIZE,
            offset=OFFSET,
            overlap=PATCH_OVERLAP,
            src_shape=inputs.shape[-2:],
            device=self.device
        ) # (n_classes, H, W) in GPU memory
        target = target.to(device=self.device, dtype=torch.float32)
    
        """ Prepare for metrics calculation """
        if OFFSET > 0:
            logits_cropped = logits[..., OFFSET: -OFFSET, OFFSET: -OFFSET]
            target_cropped = target[..., OFFSET: -OFFSET, OFFSET: -OFFSET]

        """ Metrics for output activation """
        activated = F.softmax(input=logits_cropped, dim=1)
        y_true = one_hot(target_cropped, n_classes=len(CLASS_NAMES))
        iou_activated_per_class = iou_per_class(y_true, activated)
        output_dict["iou_activated_per_class"] = iou_activated_per_class

        """ Metrics for prediction """
        mask = torch.argmax(activated, dim=0)
        prediction = one_hot(mask, n_classes=len(CLASS_NAMES))
        iou_pred_per_class = iou_per_class(y_true, prediction)
        output_dict["iou_pred_per_class"] = iou_pred_per_class

        output_dict["mean_iou_activated"] = mean_iou(iou_activated_per_class)
        output_dict["mean_iou_pred"] = mean_iou(iou_pred_per_class)

        output_dict["accuracy"] = accuracy(y_true, prediction)
        return output_dict


    def _epoch_end(self, epoch: int, train_outputs: list[dict], val_outputs: list[dict]) -> None:
        train_losses = [x["train_loss"] for x in train_outputs]
        avg_train_loss = sum(train_losses) / len(train_losses)

        val_losses = [x["val_loss"] for x in val_outputs]
        avg_val_loss = sum(val_losses) / len(val_losses)

        mean_ious_pred = [x["mean_iou_pred"] for x in val_outputs]
        avg_mean_iou_pred = sum(mean_ious_pred) / len(mean_ious_pred)

        accuracies = [x["accuracy"] for x in val_outputs]
        avg_accuracy = sum(accuracies) / len(accuracies)

        """ Print training/validation statistics """
        print(f"""[Epoch: {epoch}] Training loss: {avg_train_loss:.3f}
                                   Validation loss: {avg_val_loss:.3f}
                                   Mean IoU (prediction): {avg_mean_iou_pred:.3f}
                                   Accuracy: {avg_accuracy:.3f}""")

        """ Reduce lr on plateua """
        self.scheduler.step(avg_val_loss)


    def fit(self):
        self.model.to(self.device)
        for epoch in range(self.max_epochs):
            train_dataloader, val_dataloader = self._split_dataset()

            self.model.train()
            train_outputs = []
            bar = tqdm(train_dataloader, postfix={"train_loss": 0.0})
            for inputs, target in bar:
                train_outputs.append(self._train_image(inputs, target))
                bar.set_postfix(ordered_dict={"train_loss": train_outputs[-1]["train_loss"]})

            self.model.eval()
            val_outputs = []
            bar = tqdm(val_dataloader, postfix={"val_loss": 0.0, "mean_iou_pred": 0.0, "accuracy": 0.0})
            for inputs, target in bar:
                val_outputs.append(self._validate_image(inputs, target))
                bar.set_postfix(ordered_dict={
                    "val_loss": val_outputs[-1]["val_loss"],
                    "mean_iou_pred": val_outputs[-1]["mean_iou_pred"],
                    "accuracy": val_outputs[-1]["accuracy"],
                })

            self._epoch_end(epoch, train_outputs, val_outputs)
        torch.save(self.model.state_dict(), Path.cwd() / "tmp" / "tmp1.pth")
