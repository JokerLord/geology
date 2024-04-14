import numpy as np

from torch.utils.data import Dataset

from pathlib import Path
from PIL import Image
from typing import Callable, Optional, Union
from utils import preprocess_mask


class LumenStoneDataset(Dataset):
    def __init__(
        self,
        root_dir: Union[str, Path],
        train: bool,
        transform: Optional[Callable] = None,
    ):
        """
        Arguments:
            root_dir (string or Path): Directory with all the images.
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
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8) 

        if self._transform:
            transformed = self._transform(image=image, mask=mask)
            image = transformed["image"] # (3, H, W), dtype=torch.float32
            mask = transformed["mask"] # (H, W), dtype=torch.uint8

        target = preprocess_mask(mask) # (n_classes, H, W)
        return image, target

    def __len__(self):
        return len(self._items)
