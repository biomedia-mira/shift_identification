import numpy as np
from sklearn import metrics
import torch


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


# ==================== GPU PATH (CUDA) ====================

def _gpu_rbf_kernel(X, gamma):
    """Compute RBF kernel matrix on GPU using torch."""
    dists_sq = torch.cdist(X, X, p=2.0).pow(2)
    return torch.exp(-gamma * dists_sq)


def _gpu_batch_mmd(rbf, permutations, n1):
    """
    Vectorized batch MMD on GPU.
    Same math as CPU version, just using torch tensors on CUDA.
    """
    n = rbf.shape[0]
    n2 = n - n1
    n_perms = permutations.shape[0]

    src_idx = permutations[:, :n1]
    tgt_idx = permutations[:, n1:]

    chunk_size = max(1, min(n_perms, 200_000_000 // (n1 * n1 + n2 * n2 + n1 * n2)))
    chunk_size = min(chunk_size, n_perms)

    null_mmds = torch.empty(n_perms, device=rbf.device)

    for start in range(0, n_perms, chunk_size):
        end = min(start + chunk_size, n_perms)
        cs = src_idx[start:end]
        ct = tgt_idx[start:end]

        XX = rbf[cs[:, :, None], cs[:, None, :]]
        XX_sum = XX.sum(dim=(1, 2))
        XX_trace = torch.einsum('cii->c', XX)
        xx_term = (XX_sum - XX_trace) / (n1**2 - n1)

        YY = rbf[ct[:, :, None], ct[:, None, :]]
        YY_sum = YY.sum(dim=(1, 2))
        YY_trace = torch.einsum('cii->c', YY)
        yy_term = (YY_sum - YY_trace) / (n2**2 - n2)

        XY = rbf[cs[:, :, None], ct[:, None, :]]
        xy_term = XY.mean(dim=(1, 2))

        null_mmds[start:end] = xx_term + yy_term - 2 * xy_term

    return null_mmds


def _run_mmd_gpu(source_np, target_np, n_permutations, structure_permutation_fn):
    """Full MMD permutation test on CUDA GPU."""
    device = torch.device("cuda")
    n1, n2 = source_np.shape[0], target_np.shape[0]
    n = n1 + n2

    B = torch.from_numpy(
        np.concatenate([source_np, target_np], axis=0)
    ).float().to(device)

    # Pairwise distances + median heuristic for bandwidth
    dists = torch.cdist(B, B, p=2.0)
    sigma = torch.median(dists).item()
    gamma = 1.0 / sigma

    # RBF kernel
    rbf = _gpu_rbf_kernel(B, gamma)

    # Observed MMD (on GPU)
    XX = rbf[:n1, :n1]
    YY = rbf[n1:, n1:]
    XY = rbf[:n1, n1:]
    observed = (
        (XX.sum() - XX.trace()) / (n1**2 - n1)
        + (YY.sum() - YY.trace()) / (n2**2 - n2)
        - 2 * XY.mean()
    )

    # Generate permutations on CPU (may use structure_permutation_fn), move to GPU
    perms_np = np.array([structure_permutation_fn(n) for _ in range(n_permutations)])
    permutations = torch.from_numpy(perms_np).long().to(device)

    # Batch MMD on GPU
    null_mmds = _gpu_batch_mmd(rbf, permutations, n1)

    larger = int(torch.sum(null_mmds >= observed).item()) + 1
    return larger / n_permutations


# ==================== CPU PATH (NumPy) ====================

def _cpu_batch_mmd(rbf_distances, permutations, n1):
    """
    Vectorized batch MMD on CPU using NumPy.
    """
    n = rbf_distances.shape[0]
    n2 = n - n1
    n_perms = permutations.shape[0]

    src_idx = permutations[:, :n1]
    tgt_idx = permutations[:, n1:]

    chunk_size = max(1, min(n_perms, 500_000_000 // (n1 * n1 + n2 * n2 + n1 * n2)))
    chunk_size = min(chunk_size, n_perms)

    null_mmds = np.empty(n_perms)

    for start in range(0, n_perms, chunk_size):
        end = min(start + chunk_size, n_perms)
        cs = src_idx[start:end]
        ct = tgt_idx[start:end]

        XX = rbf_distances[cs[:, :, None], cs[:, None, :]]
        XX_sum = XX.sum(axis=(1, 2))
        XX_trace = np.einsum('cii->c', XX)
        xx_term = (XX_sum - XX_trace) / (n1**2 - n1)

        YY = rbf_distances[ct[:, :, None], ct[:, None, :]]
        YY_sum = YY.sum(axis=(1, 2))
        YY_trace = np.einsum('cii->c', YY)
        yy_term = (YY_sum - YY_trace) / (n2**2 - n2)

        XY = rbf_distances[cs[:, :, None], ct[:, None, :]]
        xy_term = XY.mean(axis=(1, 2))

        null_mmds[start:end] = xx_term + yy_term - 2 * xy_term

    return null_mmds


def _run_mmd_cpu(source, target, n_permutations, structure_permutation_fn):
    """Full MMD permutation test on CPU using NumPy."""
    n1, n2 = source.shape[0], target.shape[0]
    n = n1 + n2
    B = np.concatenate([source, target], axis=0)

    distances = metrics.pairwise_distances(B)
    sigma = np.median(distances)
    gamma = 1 / sigma
    rbf_distances = metrics.pairwise.rbf_kernel(B, B, gamma)

    observed = get_mmd_from_all_distances(rbf_distances, n1)

    permutations = np.array([structure_permutation_fn(n) for _ in range(n_permutations)])

    null_mmds = _cpu_batch_mmd(rbf_distances, permutations, n1)

    larger = np.sum(null_mmds >= observed) + 1
    return larger / n_permutations


# ==================== PUBLIC API ====================

# Detect CUDA once at import time
_USE_CUDA = torch.cuda.is_available()


def run_mmd_permutation_test(
    source, target, n_permutations=1000, structure_permutation_fn=None
):
    """
    Run full MMD permutation test.

    Automatically uses CUDA GPU if available, otherwise falls back to
    vectorized NumPy on CPU. Results are mathematically identical.

    Args:
        source: [n1, feats_dim]. Features from source set.
        target: [n2, feats_dim]. Features from target set.
        n_permutations: int. Number of permutations to run for the test.
        structure_permutation_fn: Optional[Callable]. Custom permutation function
            for structured data (e.g. EMBED exam-level permutations).

    Returns:
        p-value.
    """
    if structure_permutation_fn is None:
        structure_permutation_fn = np.random.permutation

    # Ensure numpy arrays
    if isinstance(source, torch.Tensor):
        source = source.numpy()
    if isinstance(target, torch.Tensor):
        target = target.numpy()

    if _USE_CUDA:
        return _run_mmd_gpu(source, target, n_permutations, structure_permutation_fn)
    else:
        return _run_mmd_cpu(source, target, n_permutations, structure_permutation_fn)
