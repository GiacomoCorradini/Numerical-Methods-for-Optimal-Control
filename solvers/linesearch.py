import casadi as ca
import numpy as np


def merit_l1(
    f: ca.Function,
    dfx: np.ndarray,
    Dx: np.ndarray,
    g: ca.Function | None = None,
    h: ca.Function | None = None,
    sigma: float = 0,
):
    """ℓ1 merit function and its directional derivative

    M1(x)    = f(x) + σ (||g(x)||₁ + ||max(h(x), 0)||₁)
    DdxM1(x) = ∇f(x)ᵀΔx - σ (||g(x)||₁ + ||max(h(x), 0)||₁)

    DdxM1 holds only if Δx is the Newton / QP step (g + ∇gᵀΔx = 0).
    """
    x = ca.SX.sym("x", Dx.shape[0])

    M1 = f(x)
    DdxM1 = dfx.T @ Dx

    if g is not None:
        g_x_l1 = ca.norm_1(g(x))
        M1 = M1 + sigma * g_x_l1
        DdxM1 = DdxM1 - sigma * g_x_l1

    if h is not None:
        h_x_l1 = ca.norm_1(ca.fmax(h(x), 0))
        M1 = M1 + sigma * h_x_l1
        DdxM1 = DdxM1 - sigma * h_x_l1

    M1 = ca.Function("M1", [x], [M1])
    DdxM1 = ca.Function("DdxM1", [x], [DdxM1])

    return M1, DdxM1


def merit_l1_ip(
    f: ca.Function,
    dfx: np.ndarray,
    Dx: np.ndarray,
    Ds: np.ndarray,
    g: ca.Function | None = None,
    h: ca.Function | None = None,
    sigma: float = 0,
    tau: float = 1.0,
):
    """ℓ1 merit function with log-barrier (interior point) and its directional derivative

    M1(x, s)    = f(x) - τ Σ log(sᵢ) + σ (||g(x)||₁ + ||h(x) + s||₁)
    DdxM1(x, s) = ∇f(x)ᵀΔx - τ Σ Δsᵢ/sᵢ - σ (||g(x)||₁ + ||h(x) + s||₁)

    DdxM1 holds only if (Δx, Δs) is the IP Newton step
    (g + ∇gᵀΔx = 0, h + s + ∇hᵀΔx + Δs = 0).
    """
    x = ca.SX.sym("x", Dx.shape[0])
    s = ca.SX.sym("s", Ds.shape[0])

    M1 = f(x) - tau * ca.sum1(ca.log(s))
    DdxM1 = dfx.T @ Dx - tau * ca.sum1(Ds / s)

    if g is not None:
        g_x_l1 = ca.norm_1(g(x))
        M1 = M1 + sigma * g_x_l1
        DdxM1 = DdxM1 - sigma * g_x_l1

    if h is not None:
        h_x_l1 = ca.norm_1(h(x) + s)
        M1 = M1 + sigma * h_x_l1
        DdxM1 = DdxM1 - sigma * h_x_l1

    M1 = ca.Function("M1", [x, s], [M1])
    DdxM1 = ca.Function("DdxM1", [x, s], [DdxM1])

    return M1, DdxM1


def armijo_linesearch(
    x: np.ndarray,
    f: ca.Function,
    dfx: np.ndarray,
    Dx: np.ndarray,
    beta: float,
    gamma: float,
    g: ca.Function | None = None,
    h: ca.Function | None = None,
    sigma: float = 0,
    alpha: float = 1.0,
    alpha_min: float = 1e-8,
) -> tuple[float, bool]:
    """Backtracking on Armijo's condition on the ℓ1 merit function

    Returns the step size alpha and a flag telling whether the line search succeeded.
    """

    M1, DdxM1 = merit_l1(f, dfx, Dx, g=g, h=h, sigma=sigma)

    # evaluate merit and directional derivative at the current point
    M1_x = M1(x)
    DdxM1_x = DdxM1(x)

    if DdxM1_x >= 0:
        print(
            "Warning: Dx is not a descent direction for the merit function (check sigma > ||lambda||_inf)"
        )

    while M1(x + alpha * Dx) >= (M1_x + gamma * alpha * DdxM1_x):
        alpha = beta * alpha

        if alpha < alpha_min:
            print(
                f"Warning: Armijo linesearch failed to find a suitable step size. Minimum alpha reached: {alpha_min}"
            )
            return alpha, False

    return alpha, True


def armijo_linesearch_ip(
    x: np.ndarray,
    s: np.ndarray,
    f: ca.Function,
    dfx: np.ndarray,
    Dx: np.ndarray,
    Ds: np.ndarray,
    beta: float,
    gamma: float,
    g: ca.Function | None = None,
    h: ca.Function | None = None,
    sigma: float = 0,
    alpha: float = 1.0,
    alpha_min: float = 1e-8,
    tau: float = 1.0,
) -> tuple[float, bool]:
    """Backtracking on Armijo's condition on the ℓ1 merit function with log-barrier (interior point)

    x and s are moved together along (Dx, Ds). Start from alpha = alpha_bar (fraction to the boundary),
    so that s stays positive for every tested step.

    Returns the step size alpha and a flag telling whether the line search succeeded.
    """

    M1, DdxM1 = merit_l1_ip(f, dfx, Dx, Ds, g=g, h=h, sigma=sigma, tau=tau)

    # evaluate merit and directional derivative at the current point
    M1_x = M1(x, s)
    DdxM1_x = DdxM1(x, s)

    if DdxM1_x >= 0:
        print(
            "Warning: (Dx, Ds) is not a descent direction for the merit function (check sigma > ||(lambda, mu)||_inf)"
        )

    while M1(x + alpha * Dx, s + alpha * Ds) >= (M1_x + gamma * alpha * DdxM1_x):
        alpha = beta * alpha

        if alpha < alpha_min:
            print(
                f"Warning: Armijo linesearch failed to find a suitable step size. Minimum alpha reached: {alpha_min}"
            )
            return alpha, False

    return alpha, True


def fraction_to_boundary(s, ds, mu, dmu, eps=0.01):
    """Fraction to the boundary: largest alpha_bar in (0, 1] that keeps s, mu > 0

    s + alpha_bar * ds >= eps * s  and  mu + alpha_bar * dmu >= eps * mu
    """
    alpha_bar = 1.0
    for v, dv in [(s, ds), (mu, dmu)]:
        decreasing = dv < 0  # only decreasing components limit the step
        if np.any(decreasing):
            alpha_bar = min(
                alpha_bar, np.min(-(1 - eps) * v[decreasing] / dv[decreasing])
            )
    return alpha_bar
