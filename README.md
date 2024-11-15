# Automatic dataset shift identification to support root cause analysis of AI performance drift

This repository contains the code associated with the paper [Automatic dataset shift identification to support root cause analysis of AI performance drift](https://arxiv.org/abs/2411.07940). 

## Overview
The code is divided into the following main folders:
* [classification](classification/) contains all the code related to training the task models as well as pre-training the self-supervised encoders. 
* [configs](configs/) contains all the experiment configurations for training the above models.
* [data_handling](data_handling) everything related to data loading (dataset, data modules, augmentations etc.)
* [shift_identification](shift_identification) contains all the necessary tools for dataset shift detection (BBSD tests, MMD tests) and identification (prevalence shift estimation, shift identification) 
* [experiments](experiments/) all the code related to experiments presented in the paper: inference code for each dataset group, shift generation code and plotting notebooks. 


## Pre-requisites

### Pip requirements
All required python packages are listed in [requirements.txt](requirements.txt). Please install with pip.

### Datasets

You will need to download the relevant datasets to run our code. 

#### Download links
All datasets are publicly available and be downloaded at:
* PadChest [https://bimcv.cipf.es/bimcv-projects/padchest/](https://bimcv.cipf.es/bimcv-projects/padchest/), [https://www.sciencedirect.com/science/article/pii/S1361841520301614](https://www.sciencedirect.com/science/article/pii/S1361841520301614)
* RSNA Pneumonia Detection Dataset: [https://www.kaggle.com/c/rsna-pneumonia-detection-challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
* EMBED [https://pubs.rsna.org/doi/full/10.1148/ryai.220047](https://pubs.rsna.org/doi/full/10.1148/ryai.220047), [https://github.com/Emory-HITI/EMBED_Open_Data/tree/main](https://github.com/Emory-HITI/EMBED_Open_Data/tree/main)
* Kaggle EyePacs dataset [https://www.kaggle.com/c/diabetic-retinopathy-detection/data](https://www.kaggle.com/c/diabetic-retinopathy-detection/data)
* Kaggle Aptos dataset [https://www.kaggle.com/competitions/aptos2019-blindness-detection/data](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data)
* MESSIDOR-v2 dataset (https://www.adcis.net/en/third-party/messidor2/)[https://www.adcis.net/en/third-party/messidor2/]

#### Splits generation
For every dataset we provide our train/test/split generation code to ensure reproducibility. Please run all the notebooks in [data_handling/df_creation_notebooks](data_handling/df_creation_notebooks/) to create and save the necessary splits csv.

#### Update dataset paths
Once you have downloaded the datasets, please update the corresponding paths at the top of the `mammo.py` and `xray.py` and `retina.py` files.

## Shift identification - Workflow example
Here we detail the full workflow to get the shift identification results for the mammography dataset. 
1. Train the task model with `python classification/train.py experiment=base_density`
2. Train the self-supervised encoder with `python classification/train.py experiment=simclr_embed`
3. Run inference and shift detection experiment with `python experiments/mammo_all.py --encoder_type={ENCODER} --shift={SHIFT}`. 
    * `ENCODER` should specify which encoder to use for the MMD / Duo / shift identification test. It can take values `random` (random ResNet50 encoder), `imagenet` (ResNet50 with ImageNet weights, supervised pretraining), `simclr_imagenet` (ResNet50 SimCLR pretraining on ImageNet), `simclr_modality_specific` (ResNet50 pretraining on the modality i.e. point 2), `model` (encoder from classification task model). 
    * `SHIFT` can take values `prevalence`, `acquisition` `gender`, `acquisition_prev`, `gender_prev`, `no_shift`, `all`. Defaults to `all`.
4. Plot the results with `plot_all_results.ipynb`
