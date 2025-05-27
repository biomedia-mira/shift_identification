"""
This file contains all dataset classes for the RETINA dataset.

IMPORTANT: Pre-requisites.
1. Download the three base datasets, APTOS, MESSIDOR and RSNA
Diabetic Retinopathy, unzip and update the corresponding
paths in the default_paths.py file
2. Run the data_handling/preprocess_and_splits_creation/1-generate-splits/retina_df_creation.ipynb
to prepare the csv and save our train/val/split files.
"""

from typing import Dict, Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from data_handling.base import BaseDataModuleClass
from data_handling.caching import SharedCache
from torchvision.transforms import ToTensor, Resize, CenterCrop

from default_paths import ROOT


class RetinaDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform: Callable = torch.nn.Identity(),
        cache: bool = False,
    ):
        super().__init__()
        print(f"Len {len(df)}")
        self.sites = df.site.astype(int).values
        self.labels = df.binary_diagnosis.astype(int).values
        self.sublabels = df.diagnosis.astype(int).values
        self.img_paths = df.img_path.values

        self.cache = cache
        self.transform = transform

        if cache:
            self.cache = SharedCache(
                size_limit_gib=24,
                dataset_len=self.img_paths.shape[0],
                data_dims=[3, 224, 224],
                dtype=torch.float32,
            )
        else:
            self.cache = None

    def __len__(self):
        return len(self.img_paths)

    def read_image(self, idx):
        img = Image.open(self.img_paths[idx])
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
        sample["y"] = self.labels[idx]
        sample["dr"] = self.sublabels[idx]
        sample["site"] = self.sites[idx]

        img = self.transform(img).float()

        sample["x"] = img

        return sample


class RetinaDataModule(BaseDataModuleClass):
    def create_datasets(self):
        train_df = pd.read_csv(ROOT / "experiments" / "retina_train.csv")
        val_df = pd.read_csv(ROOT / "experiments" / "retina_val.csv")
        test_df = pd.read_csv(ROOT / "experiments" / "retina_test.csv")

        self.dataset_train = RetinaDataset(
            df=train_df,
            transform=self.train_tsfm,
            cache=self.config.data.cache,
        )

        self.dataset_val = RetinaDataset(
            df=val_df,
            transform=self.val_tsfm,
            cache=self.config.data.cache,
        )

        self.dataset_test = RetinaDataset(
            df=test_df,
            transform=self.val_tsfm,
            cache=self.config.data.cache,
        )

    @property
    def dataset_name(self):
        return "retina"

    @property
    def num_classes(self):
        return 2
