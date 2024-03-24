import numpy as np

from torch.utils.data import Dataset

from pathlib import Path
from PIL import Image
from typing import Callable, Optional

from config import CLASS_NAMES
from utils.preprocess_mask import one_hot


class LumenStoneDataset(Dataset):
    def __init__(self, root_dir: str, train: bool, transform: Optional[Callable] = None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            train (bool): If True, train images will be taken, else test ones.
            transform (callable, optional): Optional transform to be applied
                on a sample
        """

        self._items = []
        self._transform = transform

        if train:
            mode = "train"
        else:
            mode = "test"

        img_dir = Path(root_dir) / "imgs" / mode
        mask_dir = Path(root_dir) / "masks" / mode

        for img_path in img_dir.glob("*.jpg"):
            mask_path = (mask_dir / img_path.name).with_suffix(".png")
            self._items.append((img_path, mask_path))

    def __getitem__(self, idx):
        img_path, mask_path = self._items[idx]
        image = np.asarray(Image.open(img_path).convert("RGB"), dtype=float)
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=float)

        mask = one_hot(mask, len(CLASS_NAMES))
        if self._transform:
            transformed = self._transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, mask

    def __len__(self):
        return len(self._items)
