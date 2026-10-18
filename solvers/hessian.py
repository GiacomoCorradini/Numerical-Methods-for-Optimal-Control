"""Hessian B used by the Newton, SQP and interior-point solvers.

- lagrangian:            L(x, λ, μ) = f + λᵀg + μᵀh
- exact_hessian:         B = ∇²ₓₓL
- gauss_newton_hessian:  B = ∇R ∇Rᵀ, f = 1/2 ||R||²
- hessian_regularization: ensure Zᵀ B Z ≻ 0
"""

import casadi as ca
import numpy as np


def lagrangian(x, f, g=None, h=None):
    """L(x, λ, μ) = f(x) + λᵀg(x) + μᵀh(x)

    Returns the symbolic expression L and the symbols λ, μ.
    If g (or h) is None, λ (or μ) is an empty vector, so that functions
    built from L always have the inputs (x, λ, μ).
    """
    lam = ca.SX.sym("lambda", g.size1_out(0)) if g is not None else ca.SX(0, 1)
    mu = ca.SX.sym("mu", h.size1_out(0)) if h is not None else ca.SX(0, 1)

    L = f(x)
    if g is not None:
        L += lam.T @ g(x)
    if h is not None:
        L += mu.T @ h(x)

    return L, lam, mu


def exact_hessian(x, L):
    """B = ∇²ₓₓL

    L is a symbolic scalar expression of x; returns a symbolic expression.
    """
    return ca.hessian(L, x)[0]


def gauss_newton_hessian(J_R):
    """B = ∇R ∇Rᵀ, for a cost f = 1/2 ||R||²

    J_R = ∂R/∂x = ∇Rᵀ is the (m x n) Jacobian of the residual R. It can be a
    symbolic expression (e.g. ca.jacobian(R(x), x)) or a numerical matrix.
    """
    return J_R.T @ J_R


def nullspace(A, tol=1e-12):
    _, S, Vt = np.linalg.svd(A, full_matrices=True)
    rank = (S > tol).sum()
    return Vt[rank:].T


def delta_j(epsilon: float, Lambda: float | None = None):

    if Lambda is None:
        sp = epsilon
    else:
        sp = max(epsilon, -Lambda)

    return sp


def hessian_regularization(Hx, Z, epsilon: float = 1e-6):
    """Eigenvalue modification of the reduced Hessian"""

    # Compute the eigenvalues of the Hessian
    Lambda, E = np.linalg.eigh(Z.T @ Hx @ Z)

    # Compute the regularization parameter lambda
    Lambda_bar = Lambda.copy()
    for i in range(len(Lambda)):
        if Lambda[i] < epsilon:
            Lambda_bar[i] = delta_j(epsilon, Lambda=Lambda[i])

    # Regularize the Hessian by adding lambda * I
    H_reg = Hx + Z @ E @ np.diag(Lambda_bar - Lambda) @ E.T @ Z.T

    return H_reg, Lambda_bar
