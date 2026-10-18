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


def sqp(
    f: ca.Function,
    g: ca.Function,
    h: ca.Function,
    x_g: np.ndarray,
    l_g: np.ndarray,
    m_g: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 200,
    beta: float = 0.5,
    gamma: float = 0.1,
    sigma: float = 0,
    hessian: str = "exact",
    R: ca.Function | None = None,
):
    post_proc = {}
    post_proc["step"] = []
    post_proc["x"] = []
    post_proc["x"].append(x_g.flatten())
    post_proc["alpha"] = []
    post_proc["kkt_violation"] = []
    post_proc["sigma"] = []
    post_proc["qp_convex"] = []

    # Compute sensitivities
    x = ca.SX.sym("x", x_g.shape[0])
    L, lam, mu = lagrangian(x, f, g, h)
    grad_f = ca.Function("grad_f", [x], [ca.gradient(f(x), x)])  # n x 1
    grad_g = ca.Function("grad_g", [x], [ca.jacobian(g(x), x).T])  # n x m
    grad_h = ca.Function("grad_h", [x], [ca.jacobian(h(x), x).T])  # n x p

    # KKT violation ||(∇L, g, max(0, h))||_inf
    kkt_fun = ca.Function(
        "kkt",
        [x, lam, mu],
        [ca.norm_inf(ca.vertcat(ca.gradient(L, x), g(x), ca.fmax(0, h(x))))],
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

    step = 0
    success = True

    # Check convergence of the initial guess
    kkt_violation = float(kkt_fun(x_g, l_g, m_g))
    post_proc["kkt_violation"].append(kkt_violation)
    converged = kkt_violation < tol

    # Print header
    print("\n" + "=" * 140)
    print(
        f"{'Step':>5} | {'f(x)':>12} | {'g(x)':>12} | {'KKT Violation':>15} | "
        f"{'Step Size':>10} | {'QP convex':>9} | {'x':>31} | {'λ':>19}"
    )
    print("=" * 140)

    while not converged and step < max_iter:
        step += 1

        # Compute Hessian of Lagrangian
        d2Lx = B_fun(x_g, l_g, m_g).full()

        # Evaluate functions at guess
        gx = g(x_g).toarray()
        hx = h(x_g).toarray()
        dfx = grad_f(x_g).toarray()
        dgx = grad_g(x_g).toarray()
        dhx = grad_h(x_g).toarray()

        # Active inequality constraints, estimated from the multipliers (mu_i > 0)
        dhx_a = dhx[:, (m_g.flatten() > 1e-5)]

        # Nullspace Z of [dgx, dhx_a]^T: regularize B such that Z^T B Z > 0
        Z = nullspace(np.block([dgx, dhx_a]).T)
        if not Z.size == 0:
            d2Lx, _ = hessian_regularization(d2Lx, Z)

        # The QP is convex only if B >= 0: the regularization only ensures Z^T B Z > 0
        qp_convex = np.min(np.linalg.eigvalsh(d2Lx)) >= -1e-10

        # Compute the newton direction by solving the QP
        Dx = ca.SX.sym("Dx", x_g.shape[0])
        f_qp = 1 / 2 * Dx.T @ d2Lx @ Dx + dfx.T @ Dx
        g_qp = gx + dgx.T @ Dx
        h_qp = hx + dhx.T @ Dx

        qp = {"x": Dx, "f": f_qp, "g": ca.vertcat(g_qp, h_qp)}
        solver = ca.qpsol(
            "solver", "qpoases", qp, {"printLevel": "none", "error_on_fail": False}
        )
        ng, nh = gx.shape[0], hx.shape[0]
        sol = solver(
            lbg=np.vstack([np.zeros((ng, 1)), -np.inf * np.ones((nh, 1))]),
            ubg=np.zeros((ng + nh, 1)),
        )

        # Check that the QP has been solved
        if not solver.stats()["success"]:
            raise RuntimeError(
                f"QP failed at x = {x_g.flatten()} ({solver.stats()['return_status']}): "
                "are the linearized constraints infeasible?"
            )

        # Get solution
        dx = np.array(sol["x"])
        l_qp = np.array(sol["lam_g"][: gx.shape[0]])
        m_qp = np.array(sol["lam_g"][gx.shape[0] :])

        # Update the l1 merit weight: exact merit function if sigma > ||(lambda+, mu+)||_inf
        sigma = max(sigma, np.linalg.norm(np.vstack([l_qp, m_qp]), ord=np.inf) + 1e-4)

        # Linesearch direction Armijo
        alpha, success = armijo_linesearch(
            x_g, f, dfx, dx, beta, gamma, g=g, h=h, sigma=sigma
        )
        if not success:
            break

        # Step
        x_g = x_g + alpha * dx
        l_g = (1 - alpha) * l_g + alpha * l_qp
        m_g = (1 - alpha) * m_g + alpha * m_qp

        post_proc["step"].append(step)
        post_proc["sigma"].append(sigma)
        post_proc["alpha"].append(alpha)
        post_proc["qp_convex"].append(qp_convex)
        post_proc["x"].append(x_g.flatten())

        # KKT violation AFTER step
        kkt_violation = float(kkt_fun(x_g, l_g, m_g))
        post_proc["kkt_violation"].append(kkt_violation)
        converged = kkt_violation < tol

        fx = f(x_g).toarray()
        gx = g(x_g).toarray()
        x_str = "[" + ", ".join(f"{v:.4e}" for v in x_g.flatten()) + "]"
        l_str = "[" + ", ".join(f"{v:.4e}" for v in l_g.flatten()) + "]"
        # Truncate if too long
        x_str = x_str[:28] + ("..." if len(x_str) > 28 else "")
        l_str = l_str[:16] + ("..." if len(l_str) > 16 else "")
        print(
            f"{step:>5d} | {fx.flatten()[0]:>12.6f} | {gx.flatten()[0]:>12.3e} | "
            f"{kkt_violation:>15.3e} | {alpha:>10.6f} | {'yes' if qp_convex else 'no':>9s} | "
            f"{x_str:>31s} | {l_str:>19s}"
        )

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

    return x_g, l_g, m_g, post_proc
