import numpy as np


def warning_alpha_min(alpha_min: float, method_name: str) -> None:
    print(
        f"Warning: {method_name} linesearch failed to find a suitable step size. Minimum alpha reached: {alpha_min}"
    )


def armijo_linesearch(
    f,
    g,
    x,
    Dx,
    dfx,
    beta: float,
    gamma: float,
    sigma: float,
    alpha: float = 1.0,
    alpha_min: float = 1e-8,
) -> float:
    f_x = f(x)
    g_x = sigma * np.linalg.norm(g(x), ord=1)
    Ddx = dfx.T @ Dx - g_x

    while (
        f(x + alpha * Dx) + sigma * np.linalg.norm(g(x + alpha * Dx), ord=1)
        >= f_x + g_x + gamma * alpha * Ddx
    ):
        alpha = beta * alpha

        if alpha < alpha_min:
            warning_alpha_min(alpha, "Armijo")
            break

    return alpha
