from src.utils import hamming_score
import numpy as np


def test_hamming_score():
    target = np.array([[1, 0, 1], [1, 1, 1], [0, 0, 0], [0, 1, 0]])
    pred = np.array([[1, 0, 1], [0, 0, 1], [0, 0, 0], [0, 1, 1]])
    assert round(hamming_score(pred, target), 2) == 0.75

    target = np.array([1, 1, 1])
    pred = np.array([1, 1, 1])
    assert hamming_score(pred, target) == 1.0

    target = np.array([1, 1, 1])
    pred = np.array([0, 0, 0])
    assert hamming_score(target, pred) == 0.0
