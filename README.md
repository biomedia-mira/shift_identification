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
All required python packages are listed in [requirements.txt](requirements.txt). Please install with pip (this should take less than 5 minutes).

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

## Shift identification - Workflow example to reproduce paper experiments
Here we detail the full workflow to reproduce all shift identification results for the mammography dataset. 
1. Train the task model with `python classification/train.py experiment=base_density`. This should only take a couple of hours to train (on a single GPU).
2. Train the self-supervised encoder with `python classification/train.py experiment=simclr_embed`. This is optional, only if you want to reproduce the detection results with the SimCLR Modality Specific encoder. If you just want to run shift identification this is not necessary. It takes a couple of days to train.
3. Run inference and shift detection experiment with `python experiments/mammo_all.py --encoder_type={ENCODER} --shift={SHIFT}`.
    * `ENCODER` should specify which encoder to use for the MMD / Duo / shift identification test. It can take values `random` (random ResNet50 encoder), `imagenet` (ResNet50 with ImageNet weights, supervised pretraining), `simclr_imagenet` (ResNet50 SimCLR pretraining on ImageNet), `simclr_modality_specific` (ResNet50 pretraining on the modality i.e. point 2), `model` (encoder from classification task model). 
    * `SHIFT` can take values `prevalence`, `acquisition` `gender`, `acquisition_prev`, `gender_prev`, `no_shift`, `all`. Defaults to `all`.
    * For each tested shift scenario, this script will save detection outputs for each bootstrap sample in a csv (one csv per shift). Running the identification experiments for all tested shifts should take a couple of hours (with 200 bootstrap samples).
4. Plot the results with `plot_all_results.ipynb`

## Main shift identification pipeline function
The main shift identification pipeline function can be found in [shift_identification_detection/shift_identification.py](shift_identification/shift_identification.py) in the `run_shift_identification` function.
This function runs one iteration shift identification/detection tests for a fixed reference and test set. 
It takes the following arguments:
```
Args:
   - task_output: output dict for task model. Should have two key 'val' and 'test' containing the results on the full validation (reference) and test sets. task_output['val'] should be a dictionary with at least a field 'y' with the ground truth, and 'probas' for the predicted probability by the task model. task_output['test'] should be a dictionary with a field 'probas' for the probability predicted by the model on the test set.
  - encoder_output: output dict for encoder. Should have two key 'val' and 'test' containing the results on the full validation (reference) and test sets. encoder_output[<split_name>] should be a dictionary with a key 'feats' containing the extracted features for each image in the set.
   - idx_shifted: which indices of the test split form sampled the target set. If the full test set should be considered as the test set, simply use np.arange(test_set_size).
   - val_idx: which indices of the val split should be used for the current reference set, if the full validation set should be used for the reference set, simply use np.arange(val_set_size).
   - num_classes: num classes in the task model, defaults to 2.
   - alpha: defaults to 0.05, significance level for the statistical tests.
```
A demo on how to use this function on some dummy dataset is provided in [shift_identification_detection/dummy_shift_identification_pipeline_demo.ipynb]([shift_identification_detection/dummy_shift_identification_pipeline_demo.ipynb]). 