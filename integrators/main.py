# %% Imports
import os
import sys
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from integrators.utils.explicit import expl_euler, expl_rk4

# %% Example 1
nx = 2
x = ca.SX.sym("x", nx)
t = ca.SX.sym("t")
u = ca.SX.sym("u")

_f1 = -16 * x[0] + 12 * x[1] + 16 * ca.cos(t) - 13 * ca.sin(t) + u
_f2 = 12 * x[0] - 9 * x[1] - 11 * ca.cos(t) + 9 * ca.sin(t) + u

f = ca.Function("f", [t, x, u], [ca.vertcat(_f1, _f2)])

# %% Integration
t0 = 0
h = 0.01
n_int = 20

x0 = np.array([[1.0], [0.0]])
control = 0
A0 = np.eye(x.shape[0])
B0 = np.zeros([x.shape[0], u.shape[0]])

# Integrate using explicit Euler
x = [x0]
t = [t0]
A_history = []
B_history = []

step_eval = 200
for i in range(step_eval):
    t_new = t0 + i * h
    x_next, A, B = expl_rk4(
        f,
        t_new,
        x[-1],
        control,
        A0,
        B0,
        h,
        1,
    )
    x.append(x_next.toarray())
    t.append(t_new)
    A_history.append(np.array(A))
    B_history.append(np.array(B))

# %%

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(
    np.array(t),
    np.array([x[i][0] for i in range(step_eval + 1)]),
    label="x[0]",
)
ax.plot(
    t,
    [x[i][1] for i in range(step_eval + 1)],
    label="x[1]",
)
ax.set_xlabel("Time")
ax.set_ylabel("State")
ax.set_title("Explicit Euler Integration")
ax.legend()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(
    t[1:],
    [A_history[i][0, 0] for i in range(step_eval)],
    label="A[0, 0]",
)
ax.plot(
    t[1:],
    [A_history[i][0, 1] for i in range(step_eval)],
    label="A[0, 1]",
)
ax.plot(
    t[1:],
    [A_history[i][1, 0] for i in range(step_eval)],
    label="A[1, 0]",
)
ax.plot(
    t[1:],
    [A_history[i][1, 1] for i in range(step_eval)],
    label="A[1, 1]",
)
ax.set_xlabel("Time")
ax.set_ylabel("A Matrix Elements")
ax.set_title("A Matrix Evolution")
ax.legend()

# %%
plt.show()


# %%

# Implicit Euler
c = np.array([1.0])
a = np.array([[1.0]])
b = np.array([1.0])

# Implicit rk4
c = np.array([1 / 2 - np.sqrt(3) / 6, 1 / 2 + np.sqrt(3) / 6])
a = np.array([[1 / 4, 1 / 4 - np.sqrt(3) / 6], [1 / 4 + np.sqrt(3) / 6, 1 / 4]])
b = np.array([1 / 2, 1 / 2])

# %% Check that the Butcher tableau is consistent
s = len(c)
assert a.shape == (s, s)
assert b.shape == (s,)


def implicit_scheme(
    f: ca.Function,
    tableau: dict,
):

    a_tab = tableau["a"]
    b_tab = tableau["b"]
    c_tab = tableau["c"]

    # Check that the Butcher tableau is consistent
    s = len(c_tab)
    assert a_tab.shape == (s, s)
    assert b_tab.shape == (s,)

    # Compose all stage equations at once with matrix operations.
    nx = int(f.size_out(0)[0])
    K = ca.SX.sym("K", nx, s)

    a_dm = ca.DM(a_tab)
    c_dm = ca.DM(c_tab)

    f_map = f.map(s)
    t_stages = ca.repmat(t, 1, s) + h * c_dm.T
    x_stages = ca.repmat(x, 1, s) + h * ca.mtimes(K, a_dm.T)
    u_stages = ca.repmat(u, 1, s)

    residual = K - f_map(t_stages, x_stages, u_stages)

    # Solve the nonlinear stage system with a Newton rootfinder.
    rf = ca.rootfinder(
        "rf",
        "newton",
        ca.Function("stage_residual", [K], [ca.reshape(residual, -1, 1)]),
    )
    kq_sol = ca.reshape(rf(ca.DM.zeros(nx * s, 1)), nx, s)

    return kq_sol
