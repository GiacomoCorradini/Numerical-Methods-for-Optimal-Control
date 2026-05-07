import os
import sys
import casadi as ca
import numpy as np

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from newton.utils.nullspace import nullspace
from newton.utils.hessian import exact_hessian, hessian_regularization
from newton.utils.linesearch import armijo_linesearch


def lagrangian(
    x: ca.SX, f: ca.Function, g: ca.Function | None = None, h: ca.Function | None = None
) -> ca.Function:

    inputs = [x]
    L = f(x)

    if g is not None:
        _lambda = ca.SX.sym("lambda", g.shape[0])
        inputs.append(_lambda)
        L = L + _lambda.T @ g(x)

    if h is not None:
        _mu = ca.SX.sym("mu", h.shape[0])
        inputs.append(_mu)
        L = L + _mu.T @ h(x)

    return ca.Function("L", inputs, [L])


#   _   _               _                    _                                _
#  | \ | | _____      _| |_ ___  _ __     __| |_   _ _ __ ___  _ __   ___  __| |
#  |  \| |/ _ \ \ /\ / / __/ _ \| '_ \   / _` | | | | '_ ` _ \| '_ \ / _ \/ _` |
#  | |\  |  __/\ V  V /| || (_) | | | | | (_| | |_| | | | | | | |_) |  __/ (_| |
#  |_| \_|\___| \_/\_/  \__\___/|_| |_|  \__,_|\__,_|_| |_| |_| .__/ \___|\__,_|
#                                                             |_|


def newton_dumped(
    x: ca.SX,
    f: ca.Function,
    g: ca.Function,
    x_g: np.ndarray,
    l_g: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 200,
    beta: float = 0.5,
    gamma: float = 0.1,
    sigma: float = 0,
):
    post_proc = {}
    post_proc["step"] = []
    post_proc["x"] = []
    post_proc["x"].append(x_g.flatten())
    post_proc["alpha"] = []
    post_proc["kkt_violation"] = []
    post_proc["sigma"] = []

    # Compute sensitivities
    df = f.jacobian()
    dg = g.jacobian()

    step = 0

    # Print header
    print("\n" + "=" * 140)
    print(
        f"{'Step':>5} | {'f(x)':>12} | {'g(x)':>12} | {'KKT Violation':>15} | "
        f"{'Step Size':>10} | {'x':>31} | {'λ':>19}"
    )
    print("=" * 140)

    while step < max_iter:
        step += 1
        post_proc["step"].append(step)

        # Compute lagrangian
        L = f(x) + ca.DM(l_g).T @ g(x)
        L = ca.Function("L", [x], [L])

        # Compute Hessian of Lagrangian
        d2Lx, _ = exact_hessian(L, x_g)

        # Evaluate functions at guess
        fx = f(x_g).toarray()
        gx = g(x_g).toarray()
        dfx = df(x_g, fx).toarray().T
        dgx = dg(x_g, gx).toarray().T

        # Regularize Hessian if needed
        Z = nullspace(dgx.T)
        if not Z.size == 0:
            d2Lx, new_eigen = hessian_regularization(d2Lx, Z)
            sigma = np.linalg.norm(new_eigen, ord=np.inf) + 1e-4

        post_proc["sigma"].append(sigma)

        # Compute the newton direction
        # [d2Lx, dgx.T; dgx, 0] [dx; dl] = -[dfx; gx]
        zero_block = np.zeros((dgx.T.shape[0], dgx.shape[1]))
        A = np.block([[d2Lx, dgx], [dgx.T, zero_block]])
        B = -np.block([[dfx], [gx]])
        sol = np.linalg.solve(A, B)
        dx = sol[: x_g.shape[0]]
        dl = sol[x_g.shape[0] :]

        # Linesearch direction Armijo
        alpha = armijo_linesearch(x_g, f, dfx, dx, beta, gamma, g=g, sigma=sigma)
        post_proc["alpha"].append(alpha)

        # Step
        x_g = x_g + alpha * dx
        l_g = (1 - alpha) * l_g + alpha * dl

        post_proc["x"].append(x_g.flatten())

        # Recompute residuals AFTER step
        L = f(x) + ca.DM(l_g).T @ g(x)
        L = ca.Function("L", [x], [L])
        dL = L.jacobian()
        Lx = L(x_g).toarray()
        dLx = dL(x_g, Lx).toarray().T
        gx = g(x_g).toarray()

        # KKT violation
        kkt_violation = np.linalg.norm(
            np.concatenate([np.array(dLx).flatten(), np.array(gx).flatten()]),
            ord=np.inf,
        )
        post_proc["kkt_violation"].append(kkt_violation)

        x_str = "[" + ", ".join(f"{v:.4e}" for v in x_g.flatten()) + "]"
        l_str = "[" + ", ".join(f"{v:.4e}" for v in l_g.flatten()) + "]"
        # Truncate if too long
        x_str = x_str[:28] + ("..." if len(x_str) > 28 else "")
        l_str = l_str[:16] + ("..." if len(l_str) > 16 else "")
        print(
            f"{step:>5d} | {fx.flatten()[0]:>12.6f} | {gx.flatten()[0]:>12.3e} | "
            f"{kkt_violation:>15.3e} | {alpha:>10.6f} | {x_str:>31s} | {l_str:>19s}"
        )

        # Check convergence
        if kkt_violation < tol:
            print("=" * 140)
            print(
                f"Converged after {step} iterations with KKT violation = {kkt_violation:.3e}"
            )
            print("=" * 140 + "\n")
            break

    if step == max_iter:
        print("=" * 140)
        print(f"Max iterations ({max_iter}) reached without convergence.")
        print("=" * 140 + "\n")

    return x_g, l_g, post_proc


#   ____   ___  ____
#  / ___| / _ \|  _ \
#  \___ \| | | | |_) |
#   ___) | |_| |  __/
#  |____/ \__\_\_|


def sqp(
    x: ca.SX,
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
):
    post_proc = {}
    post_proc["step"] = []
    post_proc["x"] = []
    post_proc["x"].append(x_g.flatten())
    post_proc["alpha"] = []
    post_proc["kkt_violation"] = []
    post_proc["sigma"] = []

    # Compute sensitivities
    df = f.jacobian()
    dg = g.jacobian()
    dh = h.jacobian()

    step = 0

    # Print header
    print("\n" + "=" * 140)
    print(
        f"{'Step':>5} | {'f(x)':>12} | {'g(x)':>12} | {'KKT Violation':>15} | "
        f"{'Step Size':>10} | {'x':>31} | {'λ':>19}"
    )
    print("=" * 140)

    while step < max_iter:
        step += 1
        post_proc["step"].append(step)

        # Compute lagrangian
        L = f(x) + ca.DM(l_g).T @ g(x) + ca.DM(m_g).T @ h(x)
        L = ca.Function("L", [x], [L])

        # Compute Hessian of Lagrangian
        d2Lx, _ = exact_hessian(L, x_g)

        # Evaluate functions at guess
        fx = f(x_g).toarray()
        gx = g(x_g).toarray()
        hx = h(x_g).toarray()
        dfx = df(x_g, fx).toarray().T
        dgx = dg(x_g, gx).toarray().T
        dhx = dh(x_g, hx).toarray().T

        # Compute the nullspace of [dgx dhx_a].T @ Z > 0
        dhx_a = dhx[:, (m_g.flatten() > 1e-5)]
        Z = nullspace(np.block([dgx, dhx_a]).T)  # TODO: use only active constraint

        # Regularize Hessian if needed
        if not Z.size == 0:
            d2Lx, new_eigen = hessian_regularization(d2Lx, Z)
            sigma = np.linalg.norm(new_eigen, ord=np.inf) + 1e-3  # 𝜎 > ||λ||_1

        post_proc["sigma"].append(sigma)

        # Compute the newton direction by solving the QP
        Dx = ca.SX.sym("Dx", x_g.shape[0])
        f_qp = 1 / 2 * Dx.T @ d2Lx @ Dx + dfx.T @ Dx
        g_qp = gx + dgx.T @ Dx
        h_qp = hx + dhx.T @ Dx

        qp = {"x": Dx, "f": f_qp, "g": ca.vertcat(g_qp, h_qp)}
        solver = ca.qpsol("solver", "qpoases", qp, {"printLevel": "none"})
        sol = solver(
            lbg=[0, -np.inf],
            ubg=[0, 0],
        )

        # Get solution
        dx = np.array(sol["x"])
        l_qp = np.array(sol["lam_g"][: gx.shape[0]])
        m_qp = np.array(sol["lam_g"][gx.shape[0] :])

        # Linesearch direction Armijo
        alpha = armijo_linesearch(x_g, f, dfx, dx, beta, gamma, g=g, h=h, sigma=sigma)
        post_proc["alpha"].append(alpha)

        # Step
        x_g = x_g + alpha * dx
        l_g = (1 - alpha) * l_g + alpha * l_qp
        m_g = (1 - alpha) * m_g + alpha * m_qp

        post_proc["x"].append(x_g.flatten())

        # Recompute residuals AFTER step
        L = f(x) + ca.DM(l_g).T @ g(x) + ca.DM(m_g).T @ h(x)
        L = ca.Function("L", [x], [L])
        dL = L.jacobian()
        Lx = L(x_g).toarray()
        dLx = dL(x_g, Lx).toarray().T
        gx = g(x_g).toarray()
        hx = h(x_g).toarray()

        # KKT violation
        kkt_violation = np.max(
            np.abs(
                np.concatenate(
                    [
                        np.array(dLx).flatten(),
                        np.array(gx).flatten(),
                        np.array(np.maximum(0, hx)).flatten(),
                    ]
                ),
            )
        )
        print("KKT Violation:", kkt_violation)
        post_proc["kkt_violation"].append(kkt_violation)

        x_str = "[" + ", ".join(f"{v:.4e}" for v in x_g.flatten()) + "]"
        l_str = "[" + ", ".join(f"{v:.4e}" for v in l_g.flatten()) + "]"
        # Truncate if too long
        x_str = x_str[:28] + ("..." if len(x_str) > 28 else "")
        l_str = l_str[:16] + ("..." if len(l_str) > 16 else "")
        print(
            f"{step:>5d} | {fx.flatten()[0]:>12.6f} | {gx.flatten()[0]:>12.3e} | "
            f"{kkt_violation:>15.3e} | {alpha:>10.6f} | {x_str:>31s} | {l_str:>19s}"
        )

        # Check convergence
        if kkt_violation < tol:
            print("=" * 140)
            print(
                f"Converged after {step} iterations with KKT violation = {kkt_violation:.3e}"
            )
            print("=" * 140 + "\n")
            break

    if step == max_iter:
        print("=" * 140)
        print(f"Max iterations ({max_iter}) reached without convergence.")
        print("=" * 140 + "\n")

    return x_g, l_g, m_g, post_proc
