import numpy as np
import pandas as pd
from typing import Tuple

from sklearn.model_selection import train_test_split

"""
This file contains all the shift generation functions for all datasets;
"""

def sample_shift_padchest(
    test_df,
    target_prev_phillips=0.418,
    target_pneumonia=0.036,
    target_dataset_size=None,
    random_state=None,
):
    imaging_test = test_df.loc[test_df["Manufacturer"] == "Imaging"]
    phillips_test = test_df.loc[test_df["Manufacturer"] == "Phillips"]
    imaging_negative = imaging_test.loc[~imaging_test.pneumonia]
    imaging_positive = imaging_test.loc[imaging_test.pneumonia]
    phillips_test_positive = phillips_test.loc[phillips_test.pneumonia]
    phillips_test_negative = phillips_test.loc[~phillips_test.pneumonia]

    if isinstance(target_pneumonia, Tuple):
        target_pneumonia = float(
            np.random.uniform(target_pneumonia[0], target_pneumonia[1], 1)
        )

    if isinstance(target_prev_phillips, Tuple):
        target_prev_phillips = float(
            np.random.uniform(target_prev_phillips[0], target_prev_phillips[1], 1)
        )

    min_sizes_available = np.asarray(
        [
            len(phillips_test_positive),
            len(imaging_positive),
            len(phillips_test_negative),
            len(imaging_negative),
        ]
    )
    n_positive_phillips = len(phillips_test_positive)
    n_positive_imaging = (
        (1 - target_prev_phillips) * n_positive_phillips / target_prev_phillips
    )
    n_negative_phillips = (
        (1 - target_pneumonia) * n_positive_phillips / target_pneumonia
    )
    n_negative_imaging = (1 - target_pneumonia) * n_positive_imaging / target_pneumonia

    target_size = np.asarray(
        [
            n_positive_phillips,
            n_positive_imaging,
            n_negative_phillips,
            n_negative_imaging,
        ]
    )
    target_size = np.round(target_size)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = target_size / min_sizes_available
    min_factor = np.max(factor[~np.isinf(factor)])
    print(factor, min_factor)
    target_size /= min_factor
    while np.any((min_sizes_available - target_size) < 0):
        target_size *= 0.9
        target_size = np.round(target_size)

    if target_dataset_size is not None:
        if target_size.sum() < target_dataset_size:
            raise ValueError(
                f"Dataset can only be {target_size.sum()} without replacement. You asked for {target_dataset_size}"
            )
        else:
            factor = target_dataset_size / target_size.sum()
            target_size *= factor
            target_size = np.round(target_size)

    target_size = target_size.astype(int)

    sub_sampled_phillips_pos = phillips_test_positive.sample(
        axis=0, replace=False, n=target_size[0], random_state=random_state
    )
    sub_sampled_phillips_neg = phillips_test_negative.sample(
        axis=0, replace=False, n=target_size[2], random_state=random_state
    )
    sub_sampled_imaging_pos = imaging_positive.sample(
        axis=0, replace=False, n=target_size[1], random_state=random_state
    )
    sub_sampled_imaging_neg = imaging_negative.sample(
        axis=0, replace=False, n=target_size[3], random_state=random_state
    )

    manufacturer_prevalence_shift = pd.concat(
        [
            sub_sampled_imaging_pos,
            sub_sampled_imaging_neg,
            sub_sampled_phillips_pos,
            sub_sampled_phillips_neg,
        ]
    )
    return manufacturer_prevalence_shift


def padchest_gender_shift(
    test_df, target_female_proportion, target_dataset_size=None, random_state=None
):
    if isinstance(target_female_proportion, Tuple):
        target_female_proportion = float(
            np.random.uniform(
                target_female_proportion[0], target_female_proportion[1], 1
            )
        )
    female_test = test_df.loc[test_df["PatientSex_DICOM"] == "F"]
    male_test = test_df.loc[test_df["PatientSex_DICOM"] == "M"]
    n_females = len(female_test)
    n_males = (1 - target_female_proportion) * n_females / target_female_proportion
    target_size = np.array([n_females, n_males])
    min_sizes = np.array([len(female_test), len(male_test)])
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = target_size / min_sizes
    min_factor = np.max(factor[~np.isinf(factor)])
    print(factor, min_factor)
    target_size /= min_factor

    while np.any((min_sizes - target_size) < 0):
        target_size *= 0.90
        target_size = np.round(target_size)

    if target_dataset_size is not None:
        if target_size.sum() < target_dataset_size:
            raise ValueError(
                f"Dataset can only be {target_size.sum()} without replacement. You asked for {target_dataset_size}"
            )
        else:
            factor = target_dataset_size / target_size.sum()
            target_size *= factor
            target_size = np.round(target_size)

    target_size = target_size.astype(int)
    sub_sampled_males = male_test.sample(
        axis=0, replace=False, n=target_size[1], random_state=random_state
    )
    sub_sampled_females = female_test.sample(
        axis=0, replace=False, n=target_size[0], random_state=random_state
    )
    gender_shift_df = pd.concat([sub_sampled_females, sub_sampled_males])
    return gender_shift_df


def padchest_gender_prev_shift(
    test_df,
    target_disease,
    target_female_proportion,
    target_dataset_size=None,
    random_state=None,
):
    gender_shifted_df = padchest_gender_shift(
        test_df, target_female_proportion, random_state=random_state
    )
    return sample_shift_padchest(
        test_df=gender_shifted_df,
        target_pneumonia=target_disease,
        target_dataset_size=target_dataset_size,
        random_state=random_state,
    )


# modelname_map = {
#     "Selenia Dimensions": 0,
#     "Senographe Pristina": 1,
#     "Senograph 2000D ADS_17.4.5": 2,
#     "Senograph 2000D ADS_17.5": 2,
#     "Lorad Selenia": 3,
#     "Clearview CSm": 4,
#     "Senographe Essential VERSION ADS_53.40": 5,
#     "Senographe Essential VERSION ADS_54.10": 5,
# }


def mammo_acq_prev_shift(
    test_df,
    target_manufacturer_distribution=np.array(
        [0.784, 0.005, 0.049, 0.042, 0.069, 0.049]
    ),
    target_density_distribution=np.array([0.073, 0.385, 0.472, 0.069]),
    target_dataset_size=None,
    random_state=None,
):
    test_df["orig_idx"] = np.arange(len(test_df))
    all_subgroups_list = []
    starting_n = len(test_df)
    target_sizes = []
    min_sizes = []

    for i in range(6):
        for j in range(4):
            list_patients_i_j = test_df.loc[
                (test_df["SimpleModelLabel"] == i) & (test_df["tissueden"] == j)
            ].acc_anon.unique()
            target_size_i_j = (
                target_density_distribution[j]
                * target_manufacturer_distribution[i]
                * starting_n
            )
            min_size_i_j = len(list_patients_i_j)
            target_sizes.append(target_size_i_j)
            min_sizes.append(min_size_i_j)
            all_subgroups_list.append(list_patients_i_j)

    min_sizes = np.asarray(min_sizes)
    target_sizes = np.asarray(target_sizes)

    with np.errstate(divide="ignore", invalid="ignore"):
        factor = target_sizes / min_sizes
    min_factor = np.max(factor[~np.isinf(factor) & ~np.isnan(factor)])
    target_sizes /= min_factor

    while np.any((min_sizes - np.round(target_sizes)) < 0):
        target_sizes *= 0.95

    if target_dataset_size is not None:
        if np.round(target_sizes).sum() < target_dataset_size:
            raise ValueError(
                f"Dataset can only be {target_sizes.sum()} without replacement. You asked for {target_dataset_size}"
            )
        else:
            factor = target_dataset_size / target_sizes.sum()
            target_sizes *= factor
    target_sizes = np.round(target_sizes)
    target_sizes = target_sizes.astype(int)
    df_shift = np.concatenate(
        [
            np.random.choice(all_subgroups_list[i], size=target_sizes[i], replace=False)
            for i in range(len(all_subgroups_list))
        ]
    )
    return test_df.loc[test_df.acc_anon.isin(df_shift)]


def simple_val_sampling_base(val_df, x):
    ids = np.random.choice(np.arange(len(val_df)), size=x, replace=False)
    return val_df.loc[ids]


def simple_val_sampling_embed(val_df, x):
    exams = np.random.choice(val_df.acc_anon.unique(), size=x, replace=False)
    return val_df.loc[val_df.acc_anon.isin(exams)]


def retina_acq_prev_shift(
    test_df,
    target_site_distribution=np.array([0.04, 0.09, 0.87]),
    target_prevalence=0.78,
    target_dataset_size=None,
    random_state=None,
):
    test_df["orig_idx"] = np.arange(len(test_df))
    all_subgroups_list = []
    starting_n = len(test_df)
    target_sizes = []
    min_sizes = []
    target_dr_distribution = [1 - target_prevalence, target_prevalence]
    for i in range(3):
        for j in range(2):
            list_images_i_j = test_df.loc[
                (test_df["site"] == i + 1)
                & (test_df["binary_diagnosis"].astype(int) == j)
            ].img_path
            target_size_i_j = (
                target_dr_distribution[j] * target_site_distribution[i] * starting_n
            )
            min_size_i_j = len(list_images_i_j)
            target_sizes.append(target_size_i_j)
            min_sizes.append(min_size_i_j)
            all_subgroups_list.append(list_images_i_j)

    min_sizes = np.asarray(min_sizes)
    target_sizes = np.asarray(target_sizes)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = target_sizes / min_sizes
    min_factor = np.max(factor[~np.isinf(factor)])
    target_sizes /= min_factor
    target_sizes = np.round(target_sizes)
    while np.any((min_sizes - target_sizes) < 0):
        target_sizes *= 0.95
        target_sizes = np.round(target_sizes)
    if target_dataset_size is not None:
        if target_sizes.sum() < target_dataset_size:
            raise ValueError(
                f"Dataset can only be {target_sizes.sum()} without replacement. You asked for {target_dataset_size}"
            )
        else:
            factor = target_dataset_size / target_sizes.sum()
            target_sizes *= factor
            target_sizes = np.round(target_sizes)

    target_sizes = target_sizes.astype(int)
    df_shift = np.concatenate(
        [
            np.random.choice(all_subgroups_list[i], size=target_sizes[i], replace=False)
            for i in range(len(all_subgroups_list))
        ]
    )
    return test_df.loc[test_df.img_path.isin(df_shift)]


def simple_val_sampling_embed_stratified(val_df, x):
    tmp = val_df.groupby("acc_anon")["tissueden"].unique()
    ids, y = tmp.index, tmp.apply(lambda x: x[0]).values
    _, test_ids = train_test_split(ids, stratify=y, test_size=x)
    return val_df.loc[val_df.acc_anon.isin(test_ids)]
