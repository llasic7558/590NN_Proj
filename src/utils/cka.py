"""
Centered Kernel Alignment (CKA) implementation.

Used in Exp 2 to measure representational similarity between layers.
Linear CKA (Kornblith et al. 2019) is used as it is efficient and
well-suited for comparing neural network representations.
"""

import numpy as np


def linear_kernel(X: np.ndarray) -> np.ndarray:
    """Compute linear kernel matrix K = X @ X.T"""
    return X @ X.T


def center_kernel(K: np.ndarray) -> np.ndarray:
    """Center a kernel matrix using the centering matrix H = I - 1/n * 11^T."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def hsic(K: np.ndarray, L: np.ndarray) -> float:
    """
    Hilbert-Schmidt Independence Criterion.

    HSIC(K, L) = (1/(n-1)^2) * trace(K_c @ L_c)
    where K_c, L_c are centered kernel matrices.
    """
    n = K.shape[0]
    K_c = center_kernel(K)
    L_c = center_kernel(L)
    return np.trace(K_c @ L_c) / ((n - 1) ** 2)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute Linear CKA between two representation matrices.

    Args:
        X: (n_samples, dim_x) representations from layer/model X
        Y: (n_samples, dim_y) representations from layer/model Y

    Returns:
        CKA similarity in [0, 1]. Higher means more similar.
    """
    K = linear_kernel(X)
    L = linear_kernel(Y)

    hsic_kl = hsic(K, L)
    hsic_kk = hsic(K, K)
    hsic_ll = hsic(L, L)

    denominator = np.sqrt(hsic_kk * hsic_ll)
    if denominator < 1e-10:
        return 0.0
    return hsic_kl / denominator


def compute_cka_matrix(layer_representations: list[np.ndarray]) -> np.ndarray:
    """
    Compute pairwise CKA between all layers.

    Args:
        layer_representations: List of (n_samples, hidden_dim) arrays,
                               one per layer.

    Returns:
        (n_layers, n_layers) CKA similarity matrix.
    """
    n_layers = len(layer_representations)
    cka_matrix = np.zeros((n_layers, n_layers))

    for i in range(n_layers):
        for j in range(i, n_layers):
            sim = linear_cka(layer_representations[i], layer_representations[j])
            cka_matrix[i, j] = sim
            cka_matrix[j, i] = sim

    return cka_matrix
