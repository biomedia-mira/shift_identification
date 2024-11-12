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
DATA_DIR_RSNA = Path("/vol/biomedic3/mb121/rsna-pneumonia-detection-challenge")
DATA_DIR_RSNA_PROCESSED_IMAGES = DATA_DIR_RSNA / "preprocess_224_224"
PATH_TO_PNEUMONIA_WITH_METADATA_CSV = (
    Path(__file__).parent / "pneumonia_dataset_with_metadata.csv"
)

if Path("/data/PadChest").exists():
    PADCHEST_ROOT = Path("/data/PadChest/PadChest")
    PADCHEST_IMAGES = PADCHEST_ROOT / "preprocessed"
else:
    PADCHEST_ROOT = Path("/vol/biodata/data/chest_xray/BIMCV-PADCHEST")
    PADCHEST_IMAGES = PADCHEST_ROOT / "images"


class PadChestDataModule(BaseDataModuleClass):
    def create_datasets(self):
        label_col = "pneumonia"
        train_df = pd.read_csv(
            "/vol/biomedic3/mb121/shift_identification/shift_exploration/train_padchest.csv"
        )
        val_df = pd.read_csv(
            "/vol/biomedic3/mb121/shift_identification/shift_exploration/val_padchest.csv"
        )
        test_df = pd.read_csv(
            "/vol/biomedic3/mb121/shift_identification/shift_exploration/test_padchest.csv"
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


class RNSAPneumoniaDetectionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform: Callable,
        cache: bool = False,
    ) -> None:
        """
        Torchvision dataset for loading RSNA dataset.
        Args:
            root: the data directory where the images can be found
            dataframe: the csv file mapping patient id, metadata, file names and label.
            transform: the transformation (i.e. preprocessing and / or augmentation)
            to apply to the image after loading them.

        This dataset returns a dictionary with the image data, label and metadata.
        """
        super().__init__()
        self.transform = transform
        self.df = df
        self.targets = self.df.label_rsna_pneumonia.values.astype(np.int64)
        self.subject_ids = self.df.patientId.unique()
        self.filenames = [
            DATA_DIR_RSNA_PROCESSED_IMAGES / f"{subject_id}.png"
            for subject_id in self.subject_ids
        ]
        self.genders = self.df["Patient Gender"].values
        self.ages = self.df["Patient Age"].values.astype(int)
        if cache:
            self.cache = SharedCache(
                size_limit_gib=24,
                dataset_len=len(self.filenames),
                data_dims=[1, 224, 224],
                dtype=torch.float32,
            )
        else:
            self.cache = None

    def read_image(self, idx):
        img = io.imread(self.filenames[idx], as_gray=True)
        img = img / (img.max() + 1e-12)
        img = CenterCrop(224)(Resize(224, antialias=True)(ToTensor()(img)))
        return img

    def __getitem__(self, index: int):
        img = self.read_image(index)
        sample = {
            "y": self.targets[index],
            "gender": self.genders[index],
            "pneumonia": self.targets[index],
            "sex": 1 if self.genders[index] == "M" else 0,
            "age": self.ages[index],
            "scanner": np.nan,
        }

        img = self.transform(img).float()

        sample["x"] = img

        return sample

    def __len__(self) -> int:
        return len(self.filenames)

    def load_counterfactual_image(self, index):
        raise NotImplementedError


class RSNAPneumoniaDataModule(BaseDataModuleClass):
    def create_datasets(self):
        """
        Pytorch Lightning DataModule defining train / val / test splits for the RSNA dataset.
        """
        if not DATA_DIR_RSNA_PROCESSED_IMAGES.exists():
            print(
                f"Data dir: {DATA_DIR_RSNA_PROCESSED_IMAGES} does not exist."
                + " Have you updated default_paths.py?"
            )

        if not PATH_TO_PNEUMONIA_WITH_METADATA_CSV.exists():
            print(
                """
                The dataset can be found at
                https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
                This dataset is originally a (relabelled) subset of the NIH dataset
                https://www.kaggle.com/datasets/nih-chest-xrays/data from
                which i took the metadata.
                To get the full csv with all the metadata please run
                data_handling/csv_generation_code/rsna_generate_full_csv.py
                """
            )

        train_df = pd.read_csv(
            "/vol/biomedic3/mb121/shift_identification/shift_exploration/train_rsna.csv"
        )
        val_df = pd.read_csv(
            "/vol/biomedic3/mb121/shift_identification/shift_exploration/val_rsna.csv"
        )
        test_df = pd.read_csv(
            "/vol/biomedic3/mb121/shift_identification/shift_exploration/test_rsna.csv"
        )

        self.dataset_train = RNSAPneumoniaDetectionDataset(
            df=train_df,
            transform=self.train_tsfm,
            cache=self.config.data.cache,
        )
        self.dataset_val = RNSAPneumoniaDetectionDataset(
            df=val_df,
            transform=self.val_tsfm,
            cache=self.config.data.cache,
        )

        self.dataset_test = RNSAPneumoniaDetectionDataset(
            df=test_df,
            transform=self.val_tsfm,
            cache=self.config.data.cache,
        )

        print("#train: ", len(self.dataset_train))
        print("#val:   ", len(self.dataset_val))
        print("#test:  ", len(self.dataset_test))

    @property
    def num_classes(self):
        return 2
