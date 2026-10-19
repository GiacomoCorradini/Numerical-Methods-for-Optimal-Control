import casadi as ca
import numpy as np

from solvers.hessian import (
    exact_hessian,
    gauss_newton_hessian,
    hessian_regularization,
    lagrangian,
    nullspace,
)

from solvers.linesearch import armijo_linesearch_ip, fraction_to_boundary


def interior_point(
    f: ca.Function,
    g: ca.Function,
    h: ca.Function,
    x_g: np.ndarray,
    l_g: np.ndarray,
    m_g: np.ndarray | float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 200,
    beta: float = 0.5,
    gamma: float = 0.1,
    sigma: float = 0,
    tau: float = 1.0,
    hessian: str = "exact",
    R: ca.Function | None = None,
):
    """Primal-dual interior-point method for  min f(x)  s.t.  g(x) = 0,  h(x) <= 0

    The inequalities are written as h(x) + s = 0 with slacks s > 0, and complementarity
    mu_i s_i = 0 is relaxed to mu_i s_i = tau, with the barrier parameter tau -> 0.

    f, g, h:     CasADi Functions of x (h can have zero outputs: no inequalities)
    x_g, l_g:    initial guess of x and lambda (column vectors)
    m_g:         initial guess of mu (> 0, default 1)
    tol:         stop when ||(grad L, g, max(0, h))||_inf < tol and tau = tol
    beta, gamma: Armijo line-search parameters
    sigma:       initial weight of the l1 merit function
    tau:         initial barrier parameter
    hessian:     "exact" or "gauss-newton" (needs R, with f = 1/2 ||R||^2)

    Returns x, lambda, mu, s and post_proc (iterates, step sizes, KKT violation, sigma, tau).
    """
    # History of the iterations (for plots)
    post_proc = {}
    post_proc["step"] = []
    post_proc["x"] = []
    post_proc["x"].append(x_g.flatten())
    post_proc["alpha"] = []
    post_proc["kkt_violation"] = []
    post_proc["sigma"] = []
    post_proc["tau"] = []

    # ---------------- Symbolic functions, built once ----------------
    # Lagrangian L = f + lambda^T g + mu^T h and its sensitivities
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

    # IP-KKT residual r_tau(x, lambda, mu, s, tau) = (∇L, g, h + s, mu * s - tau)
    s_sym = ca.SX.sym("s", h.size1_out(0))  # slack variables
    tau_sym = ca.SX.sym("tau")  # barrier parameter (scalar)
    r_tau_fun = ca.Function(
        "r_tau",
        [x, lam, mu, s_sym, tau_sym],
        [ca.vertcat(ca.gradient(L, x), g(x), h(x) + s_sym, mu * s_sym - tau_sym)],
    )

    # ---------------- Initialization ----------------
    # mu > 0 (default 1) and s = tau / mu, so that mu_i s_i = tau
    n_h = h.size1_out(0)
    if np.isscalar(m_g):
        m_g = np.full((n_h, 1), m_g, dtype=float)
    if np.any(m_g <= 0):
        print(
            "Warning: mu must be > 0 for the interior point, non-positive entries set to 1"
        )
    m_g = np.where(m_g > 0, m_g, 1.0)

    s = tau / m_g  # mu * s = tau

    step = 0
    success = True  # line-search outcome

    # Check convergence of the initial guess
    kkt_violation = float(kkt_fun(x_g, l_g, m_g))
    post_proc["kkt_violation"].append(kkt_violation)
    converged = kkt_violation < tol and (tau <= tol or n_h == 0)  # no inequalities: tau is irrelevant

    # Print header
    print("\n" + "=" * 140)
    print(
        f"{'tau':>9} | {'Step':>5} | {'f(x)':>12} | {'g(x)':>12} | {'KKT Violation':>15} | "
        f"{'Step Size':>10} | {'x':>31} | {'λ':>19}"
    )
    print("=" * 140)

    # ---------------- Interior-point iterations ----------------
    while not converged and step < max_iter:
        step += 1

        # 1. Evaluate Hessian, constraints and gradients at the current iterate
        d2Lx = B_fun(x_g, l_g, m_g).full()
        fx = f(x_g).toarray()
        gx = g(x_g).toarray()
        hx = h(x_g).toarray()
        dfx = grad_f(x_g).toarray()
        dgx = grad_g(x_g).toarray()
        dhx = grad_h(x_g).toarray()

        # 2. Regularize B on the null space of the active constraints
        # (mu > 0 always: a constraint is considered active if its slack is small, s_i < mu_i)
        dhx_a = dhx[:, (s < m_g).flatten()]

        # Nullspace Z of [dgx, dhx_a]^T: regularize B such that Z^T B Z > 0
        Z = nullspace(np.block([dgx, dhx_a]).T)

        if not Z.size == 0:
            d2Lx, _ = hessian_regularization(d2Lx, Z)

        # Define S, M
        S = np.diag(s.flatten())
        M = np.diag(m_g.flatten())

        # fmt: off
        # 3. Newton direction: IP Newton system in full form, grad(r_tau)^T [dx; dl; dmu; ds] = -r_tau
        # The unknowns are increments (dl, dmu), because the rhs contains grad L (not grad f)
        # [d2Lx,  dgx, dhx, 0] [dx ]    [grad L   ]
        # |dgx.T,   0,   0, 0| |dl | = -|g        |
        # |dhx.T,   0,   0, I| |dmu|    |h + s    |
        # [    0,   0,   S, M] [ds ]    [M s - tau]
        n_x = d2Lx.shape[0]
        n_g = dgx.shape[1]
        n_h = dhx.shape[1]

        A = np.block(
            [
                [d2Lx,                 dgx,                  dhx,                  np.zeros((n_x, n_h))],
                [dgx.T,                np.zeros((n_g, n_g)), np.zeros((n_g, n_h)), np.zeros((n_g, n_h))],
                [dhx.T,                np.zeros((n_h, n_g)), np.zeros((n_h, n_h)), np.eye(n_h)         ],
                [np.zeros((n_h, n_x)), np.zeros((n_h, n_g)), S,                    M                   ],
            ]
        )
        rhs = -r_tau_fun(x_g, l_g, m_g, s, tau).full()
        try:
            sol = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            raise RuntimeError(
                f"KKT matrix is singular at x = {x_g.flatten()}: is LICQ satisfied?"
            )
        # fmt: on

        dx = sol[:n_x]
        dl = sol[n_x : n_x + n_g]
        dmu = sol[n_x + n_g : n_x + n_g + n_h]
        ds = sol[n_x + n_g + n_h :]

        # 4. Fraction to the boundary: largest step that keeps s, mu > 0
        alpha_bar = fraction_to_boundary(s, ds, m_g, dmu)

        # 5. Update the l1 merit weight: exact merit function if sigma > ||(lambda+, mu+)||_inf
        sigma = max(
            sigma, np.linalg.norm(np.vstack([l_g + dl, m_g + dmu]), ord=np.inf) + 1e-4
        )

        # 6. Armijo line search on the IP merit function (with barrier), starting from alpha_bar
        alpha, success = armijo_linesearch_ip(
            x_g,
            s,
            f,
            dfx,
            dx,
            ds,
            beta,
            gamma,
            g=g,
            h=h,
            sigma=sigma,
            alpha=alpha_bar,
            tau=tau,
        )
        if not success:
            break

        # 7. Step on all the IP variables (increments)
        x_g = x_g + alpha * dx
        l_g = l_g + alpha * dl
        m_g = m_g + alpha * dmu
        s = s + alpha * ds

        # Store the iteration (only after a successful step)
        post_proc["step"].append(step)
        post_proc["sigma"].append(sigma)
        post_proc["alpha"].append(alpha)
        post_proc["tau"].append(tau)
        post_proc["x"].append(x_g.flatten())

        # KKT violation at the new iterate
        kkt_violation = float(kkt_fun(x_g, l_g, m_g))
        post_proc["kkt_violation"].append(kkt_violation)

        # Print the new iterate, with the tau used to compute this step
        fx = f(x_g).toarray()
        gx = g(x_g).toarray()
        x_str = "[" + ", ".join(f"{v:.4e}" for v in x_g.flatten()) + "]"
        l_str = "[" + ", ".join(f"{v:.4e}" for v in l_g.flatten()) + "]"
        # Truncate if too long
        x_str = x_str[:28] + ("..." if len(x_str) > 28 else "")
        l_str = l_str[:16] + ("..." if len(l_str) > 16 else "")
        print(
            f"{tau:>9.1e} | {step:>5d} | {fx.flatten()[0]:>12.6f} | {gx.flatten()[0]:>12.3e} | "
            f"{kkt_violation:>15.3e} | {alpha:>10.6f} | {x_str:>31s} | {l_str:>19s}"
        )

        # 8. Barrier update: decrease tau once the IP-KKT residual is small enough
        if np.linalg.norm(r_tau_fun(x_g, l_g, m_g, s, tau).full(), ord=np.inf) <= tau:
            tau = max(0.1 * tau, tol)

        # 9. Convergence: KKT satisfied and tau at its minimum (tau is irrelevant without inequalities)
        converged = kkt_violation < tol and (tau <= tol or n_h == 0)

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

    return x_g, l_g, m_g, s, post_proc
