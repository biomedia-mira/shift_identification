"""
This scripts read the dicoms from RSNA raw dataset
and saves them processed as 224x224 images into a
separate folder DATA_DIR_RSNA_PROCESSED_IMAGES that
you need to specify in default_paths.py
"""

import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm
from default_paths import DATA_DIR_RSNA, DATA_DIR_RSNA_PROCESSED_IMAGES

from PIL import Image
from torchvision.transforms import Resize


def run_rsna_preprocessing_script():
    tf = Resize(224)

    df = pd.read_csv(
        DATA_DIR_RSNA / "stage_2_train_images" / "stage_2_train_labels.csv"
    ).drop_duplicates()
    DATA_DIR_RSNA_PROCESSED_IMAGES.mkdir(parents=True, exist_ok=True)
    subject_ids = df.patientId.values
    filenames = [DATA_DIR_RSNA / f"{subject_id}.dcm" for subject_id in subject_ids]
    for file in filenames:
        scan_image = pydicom.filereader.dcmread(file).pixel_array.astype(np.float32)
        scan_image = (
            (scan_image - scan_image.min())
            * 255.0
            / (scan_image.max() - scan_image.min())
        )
        image = Image.fromarray(scan_image).convert("L")
        image = tf(image)
        image.save(DATA_DIR_RSNA_PROCESSED_IMAGES / str(file.stem + ".png"))


if __name__ == "__main__":
    run_rsna_preprocessing_script()
