import numpy as np
from main import find_seam


def test_find_seam():
    gradient = np.array([[0.2, 0.4, 0.2, 0.5, 0.3, 0.0],
                        [0.2, 0.4, 0.0, 0.2, 0.4, 0.5],
                        [0.9, 0.0, 0.0, 0.5, 0.9, 0.4],
                        [0.8, 0.0, 0.4, 0.7, 0.2, 0.9],
                        [0.7, 0.8, 0.1, 0.0, 0.8, 0.4],
                        [0.1, 0.8, 0.8, 0.3, 0.5, 0.4]])
    M, seam = find_seam(gradient)


    expected_M = np.array([[0.2, 0.4, 0.2, 0.5, 0.3, 0.0],
                         [0.4, 0.6, 0.2, 0.4, 0.4, 0.5],
                         [1.3, 0.2, 0.2, 0.7, 1.3, 0.8],
                         [1.0, 0.2, 0.6, 0.9, 0.9, 1.7],
                         [0.9, 1.0, 0.3, 0.6, 1.7, 1.3],
                         [1.0, 1.1, 1.1, 0.6, 1.1, 1.7]])

    assert np.allclose(M, expected_M, atol=1e-2), f"Expected {expected_M}, but got {M}"

    expected_seam = np.array([2, 2, 1, 1, 2, 3])
    assert np.allclose(seam, expected_seam), f"Expected {expected_seam}, but got {seam}"

    expected_total_energy = 0.6
    min_energy = (np.sum(gradient[np.arange(gradient.shape[0]), seam]))
    assert np.allclose(min_energy, expected_total_energy), f"Expected {expected_total_energy}, but got {min_energy}"

