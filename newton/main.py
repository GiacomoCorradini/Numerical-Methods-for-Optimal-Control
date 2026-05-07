# %% Imports
import os
import sys
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from newton.utils.newton_method import newton_dumped


# %% Plot function
def plot_contour_and_kkt(post_proc, f, g, x_g):
    x1 = np.linspace(-10, 10, 100)
    x2 = np.linspace(-10, 10, 100)
    X1, X2 = np.meshgrid(x1, x2)

    # Evaluate functions on mesh
    F = np.zeros_like(X1)
    G = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x_val = np.array([[X1[i, j]], [X2[i, j]]])
            F[i, j] = float(f(x_val))
            G[i, j] = float(g(x_val))

    fig = plt.figure(figsize=(12, 10))

    # x1, x2 contour of f and g=0
    ax = fig.add_subplot(221)
    ax.contour(X1, X2, F, levels=50, cmap="viridis")
    ax.contour(X1, X2, G, levels=[0], colors="red", linewidths=2)
    x_hist_array = np.array(post_proc["x"])
    ax.plot(
        x_hist_array[:, 0],
        x_hist_array[:, 1],
        "k-o",
        label="Iterations",
        linewidth=2,
        markersize=4,
    )
    ax.plot(x_g[0], x_g[1], "ro", label="Optimal Point", markersize=8)
    ax.set_title("Contour of f(x) and g(x)=0")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    ax.grid()

    # KKT violation plot
    ax = fig.add_subplot(222)
    ax.semilogy(
        post_proc["step"], post_proc["kkt_violation"], "b-o", linewidth=2, markersize=4
    )
    ax.set_title("KKT Violation over Iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("KKT Violation")
    ax.grid()

    # alpha plot
    ax = fig.add_subplot(223)
    ax.plot(post_proc["step"], post_proc["alpha"], "m-o", linewidth=2, markersize=4)
    ax.set_title("Step Size (alpha) over Iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Alpha")
    ax.grid()

    # Objective function value plot
    ax = fig.add_subplot(224)
    f_values = [float(f(x.reshape(-1, 1))) for x in post_proc["x"]]
    ax.plot(post_proc["step"], f_values[1:], "g-o", linewidth=2, markersize=4)
    ax.set_title("Objective Function over Iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("f(x)")
    ax.grid()

    plt.tight_layout()
    return fig


# %% Example 1
nx = 2
x = ca.SX.sym("x", nx)

_f = x.T @ x + ca.SX.ones(nx).T @ x
f = ca.Function("f", [x], [_f])

_g = x.T @ x - 1
g = ca.Function("g", [x], [_g])

# %% (a)

# 1
x_g = np.array([[0.0], [1.0]])
l_g = np.array([[0.0]])

try:
    x_g, l_g, post_proc = newton_dumped(
        x, f, g, x_g, l_g, beta=0.5, gamma=0.1, sigma=0, tol=1e-8
    )
    fig = plot_contour_and_kkt(post_proc, f, g, x_g)
except Exception as e:
    print("Error during optimization:", e)

# 2
x_g = np.array([[-1.0], [-1.0]])
l_g = np.array([[0.0]])

try:
    x_g, l_g, post_proc = newton_dumped(
        x, f, g, x_g, l_g, beta=0.5, gamma=0.1, sigma=0, tol=1e-8
    )
    fig = plot_contour_and_kkt(post_proc, f, g, x_g)
except Exception as e:
    print("Error during optimization:", e)

# 3
x_g = np.array([[-1.0], [1.0]])
l_g = np.array([[0.0]])

try:
    x_g, l_g, post_proc = newton_dumped(
        x, f, g, x_g, l_g, beta=0.5, gamma=0.1, sigma=0, tol=1e-8
    )
    fig = plot_contour_and_kkt(post_proc, f, g, x_g)
except Exception as e:
    print("Error during optimization:", e)

# %% (b)

# 1
x_g = np.array([[1.0], [1.0]])
l_g = np.array([[0.0]])

try:
    x_g, l_g, post_proc = newton_dumped(
        x, f, g, x_g, l_g, beta=0.5, gamma=0.1, sigma=0, tol=1e-8
    )
    fig = plot_contour_and_kkt(post_proc, f, g, x_g)
except Exception as e:
    print("Error during optimization:", e)

# 2
x_g = np.array([[1.0], [1.0 + 10e-6]])
l_g = np.array([[0.0]])

try:
    x_g, l_g, post_proc = newton_dumped(
        x, f, g, x_g, l_g, beta=0.5, gamma=0.1, sigma=0, tol=1e-8
    )
    fig = plot_contour_and_kkt(post_proc, f, g, x_g)
except Exception as e:
    print("Error during optimization:", e)

# %% (c)

x_g = np.array([[0.0], [0.0]])
l_g = np.array([[0.0]])

try:
    x_g, l_g, post_proc = newton_dumped(
        x, f, g, x_g, l_g, beta=0.5, gamma=0.1, sigma=0, tol=1e-8
    )
    fig = plot_contour_and_kkt(post_proc, f, g, x_g)
except Exception as e:
    print("Error during optimization:", e)

# %% (d)

x_g = np.array([[0.5], [1.0]])
l_g = np.array([[0.0]])

try:
    x_g, l_g, post_proc = newton_dumped(
        x, f, g, x_g, l_g, beta=0.5, gamma=0.1, sigma=0, tol=1e-8
    )
    fig = plot_contour_and_kkt(post_proc, f, g, x_g)
except Exception as e:
    print("Error during optimization:", e)

# %% Example 2
nx = 3
x = ca.SX.sym("x", nx)

_f = x.T @ x + x[1]
f = ca.Function("f", [x], [_f])

_g1 = x[0] ** 2 - 2 * x[1] ** 3 - x[1] - 10 * x[2]
_g2 = x[1] + 10 * x[2]
g = ca.Function("g", [x], [ca.vertcat(_g1, _g2)])

x_g = np.array([[1.0], [1.0], [0.0]])
l_g = np.array([[0.0], [0.0]])

try:
    x_g, l_g, post_proc = newton_dumped(
        x, f, g, x_g, l_g, beta=0.5, gamma=0.1, sigma=0, tol=1e-9
    )
    print("Optimal x:", x_g.flatten())
    print("Optimal λ:", l_g.flatten())
    fig = plot_contour_and_kkt(post_proc, f, g, x_g)
except Exception as e:
    print("Error during optimization:", e)

# Solve the problem with IPOPT for comparison
try:
    nlp = {"x": x, "f": f(x), "g": g(x)}
    solver = ca.nlpsol("solver", "ipopt", nlp)
    sol = solver(x0=x_g, lbg=0.0, ubg=0.0)
    x_opt_ipopt = sol["x"].full()
    print("Optimal x (IPOPT):", x_opt_ipopt.flatten())
except Exception as e:
    print("Error during IPOPT optimization:", e)

# %% Plotting
plt.show()
