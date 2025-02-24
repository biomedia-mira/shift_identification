import time
import numpy as np
import torch

from tqdm.autonotebook import tqdm

import pandas as pd
from shift_identification_detection.mmd_test import run_mmd_permutation_test
from shift_identification_detection.prevalence_shift_adaptation import get_cpmcn_probabilities
from sklearn.decomposition import PCA
from shift_identification_detection.bbsd_tests import run_bbsd

ALL_SHIFTS = [
    "prevalence",
    "no_shift",
    "acquisition",
    "gender",
    "gender_prev",
    "acquisition_prev",
]


def get_type(is_detected_global, is_detected_resampled):
    """
    Returns identified shift based on p-values of global
    and resample shift detection tests.

    Args:
        is_detected_global: bool. Whether the shift has been detected
        between the raw target/source datasets.
        is_detected_resampled: bool. Whether a shift has been detected
        after resampling of the source dataset to match the target prevalence.
    """
    if is_detected_global:
        if not is_detected_resampled:
            return "Prevalence shift"
        return "Other shift"
    return "No shift"


def identify_shift(
    duo_is_significant,
    bbsd_is_significant,
    bbsd_resampled_is_significant,
    mmd_resampled_is_significant,
):
    """
    Shift identification logic. Returns identified shift as string.
    """
    # Shift detection (step 3 in figure)
    if not duo_is_significant:
        return "No shift"

    # If detected then do identification:
    # If there is difference in features after resampling
    if mmd_resampled_is_significant:
        if bbsd_is_significant and (not bbsd_resampled_is_significant):
            return "Covariate + Prevalence"
        else:
            return "Covariate only"
    else:
        return "Prevalence shift"


def run_shift_identification(
    task_output,
    encoder_output,
    idx_shifted,
    val_idx,
    num_classes=2,
    alpha=0.05,
    is_embed=False,
):
    """
    Runs one iteration shift identification/detection tests for a fixed reference and test set.
    Args:
        task_output: output dict for task model
        encoder_output: output dict for encoder
        idx_shifted: which indices of the test split form sampled the target set
        val_idx: which indices of the val split for the current reference set
        num_classes: num classes
        alpha: significance level for the statistical tests
        is_embed: flag to indicate whether it is the EmBED dataset (re-sampling at exam level instead of image level)
        run_mmd_on_early_feats: flag on whether to run MMD test using early features (for ablation study)
    """
    y_val = task_output["val"]["y"]
    n_val = y_val.shape[0]
    encoder_feats_val = encoder_output["val"]["feats"][val_idx]
    probas_val = task_output["val"]["probas"][val_idx] + 1e-16
    y_val = y_val[val_idx]
    encoder_feats_test = encoder_output["test"]["feats"][idx_shifted]
    probas_test = task_output["test"]["probas"][idx_shifted] + 1e-16
    assert task_output["test"]["probas"].shape[0] == encoder_output["test"]["feats"].shape[0]
    assert task_output["val"]["probas"].shape[0] == encoder_output["val"]["feats"].shape[0]
    n_val = encoder_feats_val.shape[0]

    # Run BBSD
    bbsd_is_significant, bbsd_pvalue = run_bbsd(
        probas_val, probas_test, return_p_value=True
    )

    # Run MMD test
    t1 = time.time()
    pca = PCA(n_components=32)
    feats32pca = pca.fit_transform(
        torch.concatenate([encoder_feats_val, encoder_feats_test])
    )
    print(f"Took {time.time() - t1} for PCA")
    t1 = time.time()
    mmd_pvalue = run_mmd_permutation_test(
        feats32pca[:n_val],
        feats32pca[n_val:],
        structure_permutation_fn=embed_patient_permutations if is_embed else None,
    )
    mmd_is_significant = mmd_pvalue < alpha
    print(f"Took {time.time() - t1} for MMD")

    # Test sample idx
    idx_test = np.arange(n_val, n_val + encoder_feats_test.shape[0])

    # Estimate prevalence on target set
    cmp_out = get_cpmcn_probabilities(
        probas_test=probas_test.numpy(),
        y_val=y_val.numpy(),
        num_classes=num_classes,
        probas_val=probas_val.numpy(),
    )
    q_y = cmp_out["w_opt"] * cmp_out["p_y"]

    t1 = time.time()

    # Resample reference set
    idx = (
        embed_resample_reference_set(num_classes, y_val, q_y)
        if is_embed
        else resample_reference_set(num_classes, y_val, q_y)
    )
    print(f"Took {time.time() - t1} for resampling")

    n_val = encoder_feats_val[idx].shape[0]
    # Run BBSD resampled
    resampled_bbsd_is_significant, resampled_bbsd_pvalue = run_bbsd(
        probas_val[idx], probas_test, alpha=alpha, return_p_value=True
    )

    # Run MMD resampled
    mmd_resample_pvalue = run_mmd_permutation_test(
        feats32pca[idx],
        feats32pca[idx_test],
        structure_permutation_fn=embed_patient_permutations if is_embed else None,
    )

    mmd_resample_is_significant = mmd_resample_pvalue < alpha
    # Run duo SSL + classifier
    alph_bonferonni = alpha / (bbsd_pvalue.shape[-1] + 1)
    duo_is_significant = (mmd_pvalue < alph_bonferonni) or np.any(
        bbsd_pvalue < alph_bonferonni
    )

    result = {
        "bbsd_is_significant": bbsd_is_significant,
        "mmd_pvalue": mmd_pvalue,
        "mmd_is_significant": mmd_is_significant,
        "mmd_resample_is_significant": mmd_resample_is_significant,
        "bbsd_resampled_is_significant": resampled_bbsd_is_significant,
        "mmd_resampled_pvalue": mmd_resample_pvalue,
        "duo_is_significant": duo_is_significant,
        "final_identified_shift": identify_shift(
            duo_is_significant,
            bbsd_is_significant,
            resampled_bbsd_is_significant,
            mmd_resample_is_significant,
        ),
    }

    return result


def run_multi_detection_identification(
    task_output,
    encoder_output,
    shift_generating_func,
    source_resampling_func,
    val_sizes=[None],
    test_sizes=[None],
    n_boostraps=5,
    alpha=0.05,
    num_classes=2,
    is_embed=False,
    run_mmd_on_early_feats=False,
):
    """
    Runs shift detection/identification multiple times for evaluation.
    Performs test set sampling according to `shift_generating_func`,
    val set sampling according to `source_resampling_func`. The number
    of bootstrap samples is determined by `n_boostraps`.

    Args:
    task_outputs:               dict of task model outputs as returned by `get_or_save_outputs`. Should have both
                                val and test results. Used by BBSd test.
    `encoder_output`:               dict of encoder outputs as returned by `get_or_save_outputs`. Should have both
                                val and test results. Used by MMD test.
        shift_generating_func:  function that takes test_set size as argument and generates a resampled
                                dataframe according to desired target shift. The dataframe should have a
                                column with `idx_in_original` contain the idx of the given df row in the
                                original test set.
        source_resampling_func: function that takes val_set size as argument and generates a resampled
                                dataframe without shift (to resample reference set). The dataframe should have a
                                column with `idx_in_original` contain the idx of the given df row in the original
                                test set.
        val_sizes:              Optional array of ints. Sizes of validation sets to resample. N bootstrap samples
                                will be generated for each test set size, val set size combinations. If None no
                                resampling will occur for reference set.
        test_sizes:             Optional array of ints. Sizes of tests sets to resample. N bootstrap samples will
                                be generated for each test set size, val set size combinations. If None no resampling
                                will occur for test set.
        n_bootstraps: int.      Number of bootstrap repetitions.
        alpha:                  type I error level for statistical tests.
        num_classes:            number of classes for task model outputs.
        is_embed:               bool. If true sampling is done at the exam level (instead of image level).
    Returns:
        df                      Pandas dataframe with results for each tests (column-wise), each row is one
                                boostrap repetition.
    """
    res = pd.DataFrame()
    for val_size in val_sizes:
        print(val_size)
        for test_size in tqdm(test_sizes):
            for i in tqdm(range(n_boostraps)):
                shift_df = shift_generating_func(test_size)
                print(f"generated {len(shift_df)}")
                small_val_df = source_resampling_func(val_size)
                outputs = run_shift_identification(
                    task_output,
                    encoder_output,
                    shift_df["idx_in_original"].values,
                    val_idx=small_val_df["idx_in_original"].values,
                    alpha=alpha,
                    num_classes=num_classes,
                    is_embed=is_embed,
                    run_mmd_on_early_feats=run_mmd_on_early_feats,
                )
                outputs.update({"n_test": test_size, "boot": i, "val_size": val_size})
                print(outputs)
                current = pd.DataFrame(outputs, index=[0])
                res = pd.concat([res, current], ignore_index=True)
    return res


def resample_reference_set(
    num_classes, reference_set_labels, target_label_distribution
):
    """
    This function resamples the reference set according to the target prevalence.
    All samples are considered independent.

    Args:
        num_classes
        reference_set_labels: nd array [N,] with all the labels in the reference set
        target_label_distribution: nd array [N_classes,] with the estimated distribution of classes in the target set

    Returns:
        idx: the indices of the samples to keep in the reference to under-sample the reference to the target prevalence.
    """
    # Get class idx in validation set
    classes_idx = []
    for k in range(num_classes):
        classes_idx.append(np.where(reference_set_labels == k)[0])

    n = reference_set_labels.shape[0]
    n_classes = np.zeros(num_classes)

    for k in range(num_classes):
        n_classes[k] = target_label_distribution[k] * n

    true_n = np.asarray([cidx.shape[0] for cidx in classes_idx])
    while np.any(n_classes > true_n):
        n_classes *= 0.95

    n_classes = n_classes.astype(int)

    keep_idx = []
    for k in range(num_classes):
        keep_idx.append(
            np.random.choice(classes_idx[k], size=n_classes[k], replace=False)
        )

    idx = np.concatenate(keep_idx)
    return idx


def embed_resample_reference_set(num_classes, y_val, q_y):
    """
    EMBED specific function to resample the reference set according to the target prevalence
    without splitting exams (i.e. all 4 images taken simultaneously in a case are either
    included or excluded from the resampled dataset.).
    Here all images from the same patients have the same class.
    """
    size = y_val.shape[0]
    patient_class = y_val[np.arange(start=0, stop=int(size), step=4)]
    idx_patient_to_keep = resample_reference_set(num_classes, patient_class, q_y)
    idx_images_to_keep = idx_patient_to_keep.repeat(4) * 4 + np.tile(
        np.arange(4), idx_patient_to_keep.shape[0]
    )
    return idx_images_to_keep


def embed_patient_permutations(size):
    """
    EMBED specific function for generating permutations for the permutation test.
    To ensure that the permutation is done at the exam level, i.e. images from the
    same exam should not be in different sets (4 images per exam).
    """
    patients_permutations = np.random.permutation(int((size) / 4))
    img_permutation = patients_permutations.repeat(4) * 4 + np.tile(
        np.arange(4), int((size) / 4)
    )
    return img_permutation
