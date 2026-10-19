import casadi as ca
import numpy as np

from solvers.hessian import (
    exact_hessian,
    gauss_newton_hessian,
    hessian_regularization,
    lagrangian,
    nullspace,
)
from solvers.linesearch import armijo_linesearch


def newton_damped(
    f: ca.Function,
    g: ca.Function,
    x_g: np.ndarray,
    l_g: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 200,
    beta: float = 0.5,
    gamma: float = 0.1,
    sigma: float = 0,
    hessian: str = "exact",
    R: ca.Function | None = None,
):
    """Damped Newton method for  min f(x)  s.t.  g(x) = 0

    f, g:        CasADi Functions of x
    x_g, l_g:    initial guess of x and lambda (column vectors)
    tol:         stop when ||(grad L, g)||_inf < tol
    beta, gamma: Armijo line-search parameters
    sigma:       initial weight of the l1 merit function
    hessian:     "exact" or "gauss-newton" (needs R, with f = 1/2 ||R||^2)

    Returns x, lambda and post_proc (iterates, step sizes, KKT violation, sigma).
    """
    # History of the iterations (for plots)
    post_proc = {}
    post_proc["step"] = []
    post_proc["x"] = []
    post_proc["x"].append(x_g.flatten())
    post_proc["alpha"] = []
    post_proc["kkt_violation"] = []
    post_proc["sigma"] = []

    # ---------------- Symbolic functions, built once ----------------
    # Lagrangian L = f + lambda^T g and its sensitivities
    x = ca.SX.sym("x", x_g.shape[0])
    L, lam, mu = lagrangian(x, f, g)
    grad_f = ca.Function("grad_f", [x], [ca.gradient(f(x), x)])  # n x 1
    grad_g = ca.Function("grad_g", [x], [ca.jacobian(g(x), x).T])  # n x m

    # KKT violation ||(∇L, g)||_inf
    kkt_fun = ca.Function(
        "kkt", [x, lam, mu], [ca.norm_inf(ca.vertcat(ca.gradient(L, x), g(x)))]
    )

    # Hessian of the Lagrangian: exact or Gauss-Newton (f = 1/2 ||R||^2)
    if hessian == "exact":
        B_fun = ca.Function("B", [x, lam, mu], [exact_hessian(x, L)])
    elif hessian == "gauss-newton":
        if R is None:
            raise ValueError(
                "The Gauss-Newton Hessian needs the residual R (f = 1/2 ||R||^2)"
            )
        B_fun = ca.Function(
            "B", [x, lam, mu], [gauss_newton_hessian(ca.jacobian(R(x), x))]
        )
    else:
        raise ValueError(f"Unknown hessian '{hessian}': use 'exact' or 'gauss-newton'")

    # ---------------- Initialization ----------------
    step = 0
    success = True  # line-search outcome

    # Check convergence of the initial guess
    kkt_violation = float(kkt_fun(x_g, l_g, []))
    post_proc["kkt_violation"].append(kkt_violation)
    converged = kkt_violation < tol

    # Print header
    print("\n" + "=" * 140)
    print(
        f"{'Step':>5} | {'f(x)':>12} | {'g(x)':>12} | {'KKT Violation':>15} | "
        f"{'Step Size':>10} | {'x':>31} | {'λ':>19}"
    )
    print("=" * 140)

    # ---------------- Newton iterations ----------------
    while not converged and step < max_iter:
        step += 1

        # 1. Evaluate Hessian, constraints and gradients at the current iterate
        d2Lx = B_fun(x_g, l_g, []).full()
        gx = g(x_g).toarray()
        dfx = grad_f(x_g).toarray()
        dgx = grad_g(x_g).toarray()

        # 2. Regularize B on the null space of grad g^T (Z^T B Z > 0 -> descent direction)
        Z = nullspace(dgx.T)
        if not Z.size == 0:
            d2Lx, _ = hessian_regularization(d2Lx, Z)

        # 3. Newton direction from the KKT system (rhs has grad f, so it gives lambda+ directly)
        # [d2Lx, dgx; dgx.T, 0] [dx; lam_plus] = -[dfx; gx]
        zero_block = np.zeros((dgx.T.shape[0], dgx.shape[1]))
        A = np.block([[d2Lx, dgx], [dgx.T, zero_block]])
        rhs = -np.block([[dfx], [gx]])
        try:
            sol = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            raise RuntimeError(
                f"KKT matrix is singular at x = {x_g.flatten()}: is LICQ satisfied?"
            )
        dx = sol[: x_g.shape[0]]
        lam_plus = sol[x_g.shape[0] :]

        # 4. Update the l1 merit weight: exact merit function if sigma > ||lambda+||_inf
        sigma = max(sigma, np.linalg.norm(lam_plus, ord=np.inf) + 1e-4)

        # 5. Armijo line search on the l1 merit function
        alpha, success = armijo_linesearch(
            x_g, f, dfx, dx, beta, gamma, g=g, sigma=sigma
        )
        if not success:
            break

        # 6. Damped step on x and lambda
        x_g = x_g + alpha * dx
        l_g = (1 - alpha) * l_g + alpha * lam_plus

        # Store the iteration (only after a successful step)
        post_proc["step"].append(step)
        post_proc["sigma"].append(sigma)
        post_proc["alpha"].append(alpha)
        post_proc["x"].append(x_g.flatten())

        # 7. Convergence check at the new iterate
        kkt_violation = float(kkt_fun(x_g, l_g, []))
        post_proc["kkt_violation"].append(kkt_violation)
        converged = kkt_violation < tol

        # Print the iteration
        fx = f(x_g).toarray()
        gx = g(x_g).toarray()
        x_str = "[" + ", ".join(f"{v:.4e}" for v in x_g.flatten()) + "]"
        l_str = "[" + ", ".join(f"{v:.4e}" for v in l_g.flatten()) + "]"
        # Truncate if too long
        x_str = x_str[:28] + ("..." if len(x_str) > 28 else "")
        l_str = l_str[:16] + ("..." if len(l_str) > 16 else "")
        print(
            f"{step:>5d} | {fx.flatten()[0]:>12.6f} | {gx.flatten()[0]:>12.3e} | "
            f"{kkt_violation:>15.3e} | {alpha:>10.6f} | {x_str:>31s} | {l_str:>19s}"
        )

    # ---------------- Final message ----------------
    print("=" * 140)
    if converged:
        print(
            f"Converged after {step} iterations with KKT violation = {kkt_violation:.3e}"
        )
    elif not success:
        print(f"Line search failed at iteration {step}: stopping.")
    else:
        print(f"Max iterations ({max_iter}) reached without convergence.")
    print("=" * 140 + "\n")

    return x_g, l_g, post_proc
