from scipy.optimize import minimize
import torch
from shift_identification_detection.temperature_scaling import VectorScaling
import numpy as np
from abstention.calibration import inverse_softmax


def get_cpmcn_probabilities(probas_test, y_val, probas_val, num_classes):
    """
    Get recalibrated probabilities and estimated density ratio according to CPMCN paper, Wen et al 2024, ICLR

    Args:
        probas_test:    probabilities on (shifted) testset
        y_val:          labels of validation set
        probas_val:     probabilities predicted on validation set
        num_classes:    number of classes
    Returns:
        dict:           with 'w_opt' estimated density ratio, 'new_probas'
                        re-calibrated probabilities with CPMCN, 'p_y' prevalence on
                        validation set, 'q_cal' test set probabilities recalibrated
                        with simple vector scaling (no label adaptation)

    """
    if isinstance(probas_test, torch.Tensor):
        probas_test = probas_test.numpy()
    if isinstance(probas_val, torch.Tensor):
        probas_val = probas_val.numpy()
    if isinstance(y_val, torch.Tensor):
        y_val = y_val.numpy()
    vs = VectorScaling(num_label=num_classes, print_verbose=False, bias=True)
    vs.fit(inverse_softmax(probas_val), y_val)
    q_cal = vs.calibrate(inverse_softmax(probas_test))
    p_val_cal = vs.calibrate(inverse_softmax(probas_val))
    p_y = torch.tensor([(y_val == k).mean() for k in range(num_classes)]).numpy()
    w = np.ones(num_classes).astype(float)
    w_opt = minimize(
        cost_fn,
        x0=w,
        args=(p_y, q_cal, num_classes),
        bounds=[(0, None) for _ in range(w.shape[0])],
    ).x
    w_opt = w_opt / (w_opt * p_y).sum()
    new_probas = w_opt * q_cal / (w_opt * q_cal).sum(axis=1, keepdims=True)
    return {
        "w_opt": w_opt,
        "new_probas": new_probas,
        "p_y": p_y,
        "q_cal": q_cal,
        "p_val_cal": p_val_cal,
    }


def get_pw(probabilities, w, k):
    return np.mean(probabilities[:, k] / (probabilities @ w))


def cost_fn(w, p, q, num_classes):
    c = 0
    for k in range(num_classes):
        c += (p[k] - get_pw(q, w, k)) ** 2
    return c
