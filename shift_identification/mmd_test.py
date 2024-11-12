import multiprocessing
from sklearn import metrics
import numpy as np
from multiprocessing import Pool


def get_mmd_from_all_distances(distances, n1):
    """
    Computes MMD estimator as per Gretton et al. `A Kernel Two-Sample Test`

    Args:
        distances: [n1+n2, n1+n2] array. Pairwise distance matrix over joint source and target sets.
        n1: int. Size of source dataset.
    """
    XX = distances[:n1, :n1]
    YY = distances[n1:, n1:]
    XY = distances[:n1, n1:]
    n2 = distances.shape[0] - n1
    return (
        (XX.sum() - np.trace(XX)) / (n1**2 - n1)
        + (YY.sum() - np.trace(YY)) / (n2**2 - n2)
        - 2 * XY.mean()
    )


def permutation_worker(p, n1, rbf_distances, observed):
    """
    Wrapper function to run multiprocessing permuation test
    Args:
        p: permutation of indices
        n1: size of source dataset
        rbf_distances: [n1+n2, n1+n2] array. Pairwise distance matrix over joint source and target sets.
        observed: observed MMD value between source and target sets.
    """
    null_observed = get_mmd_from_all_distances(rbf_distances[p][:, p], n1)
    return 1 if null_observed >= observed else 0


def run_mmd_permutation_test(
    source, target, n_permutations=1000, structure_permutation_fn=None
):
    """
    Run full MMM permutaton test in parallel (across 20 cores).
    Args:
        source: [n1, feats_dim]. Features from source set.
        target: [n2, feats_dim]. Features from target set.
        n_permutations: int. Number of permutations to run for the test.
        structure_permutation_fn: Optional[Callable]. If naive permuation is not suited,
        e.g. if the data has some specific sub-structure run custom permuation function. For EMBED
        this means permuting at the exam level instead of at the image level to avoid putting images from
        the same patient/exam in different splits of the permutation.

    Returns:
        p-value.
    """
    if structure_permutation_fn is None:
        structure_permutation_fn = np.random.permutation
    n1, n2 = source.shape[0], target.shape[0]
    p_to_use = multiprocessing.cpu_count() - 1
    distances = metrics.pairwise_distances(
        np.concatenate([source, target], axis=0), n_jobs=p_to_use
    )
    sigma = np.median(distances)
    gamma = 1 / sigma
    B = np.concatenate([source, target], axis=0)
    rbf_distances = metrics.pairwise.rbf_kernel(B, B, gamma)
    observed = get_mmd_from_all_distances(rbf_distances, n1)
    with Pool(processes=p_to_use) as pool:
        results = pool.starmap(
            permutation_worker,
            [
                (structure_permutation_fn(n1 + n2), n1, rbf_distances, observed)
                for _ in range(n_permutations)
            ],
        )
        larger = sum(results) + 1
    return larger / n_permutations
