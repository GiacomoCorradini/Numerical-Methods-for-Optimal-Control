import numpy as np


def nullspace(A, tol=1e-12):
    _, S, Vt = np.linalg.svd(A, full_matrices=True)
    rank = (S > tol).sum()
    return Vt[rank:].T


if __name__ == "__main__":
    # Case 1: wide matrix (2x3), nullity should be 1.
    A1 = np.array([[1, 2, 3], [4, 5, 6]])
    N1 = nullspace(A1)
    rank_A1 = np.linalg.matrix_rank(A1)
    nullity_dim = A1.shape[1] - rank_A1
    assert (
        N1.shape[1] == nullity_dim
    ), f"Expected nullity dimension {nullity_dim}, got {N1.shape[1]}"
    assert np.allclose(A1 @ N1, 0.0, atol=1e-10), "A1 @ N1 is not near zero"

    # Case 2
    A2 = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    N2 = nullspace(A2)
    rank_A2 = np.linalg.matrix_rank(A2)
    nullity_dim2 = A2.shape[1] - rank_A2
    assert (
        N2.shape[1] == nullity_dim2
    ), f"Expected nullity dimension {nullity_dim2}, got {N2.shape[1]}"
    assert np.allclose(A2 @ N2, 0.0, atol=1e-10), "A2 @ N2 is not near zero"

    print("Nullspace check passed.")
    print("N1 shape:", N1.shape)
    print("N2 shape:", N2.shape)
