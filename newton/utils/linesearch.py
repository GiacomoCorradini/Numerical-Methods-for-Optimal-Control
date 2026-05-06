def warning_alpha_min(alpha_min: float, method_name: str) -> None:
    print(
        f"Warning: {method_name} linesearch failed to find a suitable step size. Minimum alpha reached: {alpha_min}"
    )


def backtracking_linesearch(
    f, x, Dx, alpha: float, beta: float, alpha_min: float = 1e-8
) -> float:
    while f(x + alpha * Dx) >= f(x):
        alpha = beta * alpha

        if alpha < alpha_min:
            warning_alpha_min(alpha, "Backtracking")
            break

    return alpha


def armijo_linesearch(
    f, x, Dx, dfx, alpha: float, beta: float, gamma: float, alpha_min: float = 1e-8
) -> float:
    while f(x + alpha * Dx) >= f(x) + gamma * alpha * dfx.T @ Dx:
        alpha = beta * alpha

        if alpha < alpha_min:
            warning_alpha_min(alpha, "Armijo")
            break

    return alpha
