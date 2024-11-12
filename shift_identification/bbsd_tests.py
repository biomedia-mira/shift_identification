from scipy.stats import ks_2samp
import numpy as np


def run_bbsd(p, q, alpha=0.05, return_p_value=False):
    """
    Runs multiple univariate K-S tests between multi-dimensional
    distributions p and q.
    Args:
        p: [n_samples_source, n_dim]. Source distribution
        q: [n_samples_target, n_dim]. Target distribution
        alpha: type I error. Final significance returns after
        applying Bonferroni correction.
        return_p_value: bool. If True returns all raw uncorrected
        p-values for each n_dim tests.
    """
    n_dim = p.shape[1]
    all_p_values = np.ones(n_dim)
    for d in range(n_dim):
        all_p_values[d] = ks_2samp(p[:, d], q[:, d], method="exact").pvalue
    alph_bonferonni = alpha / n_dim
    if return_p_value:
        return bool(np.any(all_p_values < alph_bonferonni)), all_p_values
    return bool(np.any(all_p_values < alph_bonferonni))
