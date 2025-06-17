"""
This file contains all the paths for the
downloaded datasets.
!! Please update all these paths this with your own data paths.!!
"""

from pathlib import Path


ROOT = Path(__file__).parent

#### Paths for RSNA dataset ####
# Download from https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data
DATA_DIR_RSNA = Path("/vol/biomedic3/mb121/rsna-pneumonia-detection-challenge")
# This is the target path for data_handling/preprocess_and_splits_creation/2-create-preprocessed-images/rsna_preprocess.py
DATA_DIR_RSNA_PROCESSED_IMAGES = DATA_DIR_RSNA / "preprocess_224_224"
# Download from https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-dataset-mappings_2018.json
PATH_NIH_TO_RSNA_MAPPING = "pneumonia-challenge-dataset-mappings_2018.json"
# Download from https://www.kaggle.com/datasets/nih-chest-xrays/data
NIH_METADATA_CSV = "Data_Entry_2017.csv"

#### Path for PadChest dataset ####
# Download from https://bimcv.cipf.es/bimcv-projects/padchest/
PADCHEST_ROOT = Path("/vol/biodata/data/chest_xray/BIMCV-PADCHEST")

#### Path for RETINA datasets ####
# Download from https://www.adcis.net/en/third-party/messidor2/
MESSIDOR_ROOT = Path("/vol/biomedic3/mb121/data/messidor/")
# Download from https://www.kaggle.com/competitions/aptos2019-blindness-detection/data
APTOS_ROOT = Path("/vol/biomedic3/mb121/data/aptos2019/")
# Download from https://www.kaggle.com/c/diabetic-retinopathy-detection/data
DIABETIC_ROOT = Path("/vol/biodata/data/diabetic_retino")

#### Path for EMBED ####
# Downlaad https://github.com/Emory-HITI/EMBED_Open_Data/tree/main
EMBED_ROOT = Path("/vol/biomedic3/data/EMBED")


#### Path to pretrained model weights
# Download from https://github.com/AndrewAtanov/simclr-pytorch
PATH_TO_SIMCLR_IMAGENET = "/vol/biomedic3/mb121/shift_identification/experiments/pretrained_models/resnet50_imagenet_bs2k_epochs600.pth.tar"
# Download at https://github.com/rmaphoh/RETFound_MAE
PATH_TO_RETFOUND = "/vol/biomedic3/mb121/harmonisation/RETFound_cfp_weights.pth"
