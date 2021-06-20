import numpy as np


def hamming_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the Hamming score http://stackoverflow.com/q/32239577/395857"""
    return y_true[y_true == y_pred].size / y_true.size
