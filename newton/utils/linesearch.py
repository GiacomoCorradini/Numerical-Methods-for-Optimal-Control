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
) -> float:

    M1, DdxM1 = merit_l1(f, dfx, Dx, g=g, h=h, sigma=sigma)

    # evaluate merit at numeric points; DdxM1 returns a constant
    while M1(x + alpha * Dx) >= (M1(x) + gamma * alpha * DdxM1(x)):
        alpha = beta * alpha

        if alpha < alpha_min:
            print(
                f"Warning: Armijo linesearch failed to find a suitable step size. Minimum alpha reached: {alpha_min}"
            )
            break

    return alpha
