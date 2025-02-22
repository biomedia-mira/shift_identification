from pathlib import Path
from typing import Callable, Dict
import numpy as np

import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
from torchvision.transforms import ToTensor, Resize, CenterCrop
from data_handling.base import BaseDataModuleClass

from data_handling.caching import SharedCache


# Please update this with your own paths.
PADCHEST_ROOT = Path("YOUR PATH")
PADCHEST_IMAGES = PADCHEST_ROOT / "preprocessed"

PROJECT_ROOT = Path(__file__).parent.parent


class PadChestDataModule(BaseDataModuleClass):
    def create_datasets(self):
        label_col = "pneumonia"
        train_df = pd.read_csv(
            PROJECT_ROOT / "data/train_padchest.csv"
        )
        val_df = pd.read_csv(
            PROJECT_ROOT / "data/val_padchest.csv"
        )
        test_df = pd.read_csv(
            PROJECT_ROOT / "data/test_padchest.csv"
        )

        self.dataset_train = PadChestDataset(
            df=train_df,
            transform=self.train_tsfm,
            label_column=label_col,
            cache=self.config.data.cache,
        )

        self.dataset_val = PadChestDataset(
            df=val_df,
            transform=self.val_tsfm,
            label_column=label_col,
            cache=self.config.data.cache,
        )

        self.dataset_test = PadChestDataset(
            df=test_df,
            transform=self.val_tsfm,
            label_column=label_col,
            cache=self.config.data.cache,
        )

    @property
    def dataset_name(self):
        return "padchest"

    @property
    def num_classes(self):
        return 2


class PadChestDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        label_column: str = "pneumonia",
        transform: Callable = torch.nn.Identity(),
        cache: bool = False,
    ):
        super().__init__()
        print(f"Len {len(df)}")
        self.label_col = label_column
        self.pneumonia = df.pneumonia.astype(int).values
        self.img_paths = df.ImageID.values
        self.genders = df.PatientSex_DICOM.values
        self.ages = df.PatientAge.values
        self.manufacturers = df.Manufacturer.values
        self.cache = cache
        self.transform = transform

        if cache:
            self.cache = SharedCache(
                size_limit_gib=24,
                dataset_len=self.img_paths.shape[0],
                data_dims=[1, 224, 224],
                dtype=torch.float32,
            )
        else:
            self.cache = None

    def __len__(self):
        return len(self.img_paths)

    def read_image(self, idx):
        try:
            img = io.imread(PADCHEST_IMAGES / self.img_paths[idx], as_gray=True)
        except:  # noqa
            from PIL import ImageFile

            ImageFile.LOAD_TRUNCATED_IMAGES = True
            print(self.img_paths[idx])
            img = io.imread(PADCHEST_IMAGES / self.img_paths[idx], as_gray=True)
            print("success")
            ImageFile.LOAD_TRUNCATED_IMAGES = False
        img = img / (img.max() + 1e-12)
        img = CenterCrop(224)(Resize(224, antialias=True)(ToTensor()(img)))
        return img

    def __getitem__(self, idx: int) -> Dict:
        if self.cache is not None:
            img = self.cache.get_slot(idx)
            if img is None:
                img = self.read_image(idx)
                self.cache.set_slot(idx, img, allow_overwrite=True)

        else:
            img = self.read_image(idx)

        sample = {}
        sample["pneumonia"] = self.pneumonia[idx]
        sample["age"] = self.ages[idx] / 100
        sample["sex"] = 0 if self.genders[idx] == "M" else 1
        sample["scanner"] = 0 if self.manufacturers[idx] == "Phillips" else 1
        sample["y"] = sample[self.label_col]
        sample["shortpath"] = self.img_paths[idx]

        img = self.transform(img).float()

        sample["x"] = img

        return sample
