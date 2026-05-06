import numpy as np


def exact_hessian(f, x):

    # Compute the Jacobian of f
    J = f.jacobian()

    # Compute the Hessian of f
    H = J.jacobian()

    # Evaluate the Hessian at x
    fx = f(x).toarray()
    Jx = J(x, fx).toarray().T  # transpose to get jacobian column-wise
    Hx = H(x, fx, Jx)[0].toarray()

    return Hx, Jx


def delta_j(epsilon: float, Lambda: float | None = None):

    if Lambda is None:
        sp = epsilon
    else:
        sp = max(epsilon, -Lambda)

    return sp


def hessian_regularization(Hx, Z, epsilon: float = 1e-6):

    # Compute the eigenvalues of the Hessian
    Lambda, E = np.linalg.eigh(Z.T @ Hx @ Z)

    # Compute the regularization parameter lambda
    Lambda_bar = Lambda.copy()
    for i in range(len(Lambda)):
        if Lambda[i] < epsilon:
            Lambda_bar[i] = delta_j(epsilon, Lambda=Lambda[i])

    # Regularize the Hessian by adding lambda * I
    H_reg = Hx + Z @ E @ np.diag(Lambda_bar - Lambda) @ E.T @ Z.T

    return H_reg, Lambda_bar


# def gauss_newton_hessian(f, x):
#     # Compute the Jacobian of f
#     J = f.jacobian()

#     # Evaluate the Jacobian at x
#     fx = f(x).toarray()
#     Jx = J(x, fx).toarray().T

#     # Compute the Gauss-Newton Hessian approximation
#     H_gn = Jx @ Jx.T

#     # Compute the exact Hessian
#     H_exact = exact_hessian(f, x)

#     return H_gn, H_exact


if __name__ == "__main__":
    import casadi as ca
    import numpy as np

    x = ca.SX.sym("x", 2)

    # f = x^2 + 3*x*y + 2*y^2 + x
    # df/dx = 2*x + 3*y + 1
    # df/dy = 3*x + 4*y
    # d^2f/dx^2 = 2
    # d^2f/dxdy = 3
    # d^2f/dydx = 3
    # d^2f/dy^2 = 4
    eval_point = np.array([[1.5], [-0.5]])
    f = ca.Function(
        "f",
        [x],
        [x[0] ** 2 + 3 * x[0] * x[1] + 2 * x[1] ** 2 + x[0]],
    )
    np.testing.assert_allclose(
        exact_hessian(f, eval_point)[1],
        np.array(
            [
                [2.0 * eval_point[0, 0] + 3.0 * eval_point[1, 0] + 1.0],
                [3.0 * eval_point[0, 0] + 4.0 * eval_point[1, 0]],
            ]
        ),
    )
    np.testing.assert_allclose(
        exact_hessian(f, eval_point)[0],
        np.array([[2.0, 3.0], [3.0, 4.0]]),
    )

    # f = x^3 + 2*x^2*y + 3*x*y^2 + 4*y^3
    # df/dx = 3*x^2 + 4*x*y + 3*y^2
    # df/dy = 2*x^2 + 6*x*y + 12*y^2
    # d^2f/dx^2 = 6*x + 4*y
    # d^2f/dxdy = 4*x + 6*y
    # d^2f/dydx = 4*x + 6*y
    # d^2f/dy^2 = 6*x + 24*y
    f = ca.Function(
        "f",
        [x],
        [x[0] ** 3 + 2 * x[0] ** 2 * x[1] + 3 * x[0] * x[1] ** 2 + 4 * x[1] ** 3],
    )
    np.testing.assert_allclose(
        exact_hessian(f, eval_point)[1],
        np.array(
            [
                [
                    3 * eval_point[0, 0] ** 2
                    + 4 * eval_point[0, 0] * eval_point[1, 0]
                    + 3 * eval_point[1, 0] ** 2
                ],
                [
                    2 * eval_point[0, 0] ** 2
                    + 6 * eval_point[0, 0] * eval_point[1, 0]
                    + 12 * eval_point[1, 0] ** 2
                ],
            ]
        ),
    )
    np.testing.assert_allclose(
        exact_hessian(f, eval_point)[0],
        np.array(
            [
                [
                    6 * eval_point[0, 0] + 4 * eval_point[1, 0],
                    4 * eval_point[0, 0] + 6 * eval_point[1, 0],
                ],
                [
                    4 * eval_point[0, 0] + 6 * eval_point[1, 0],
                    6 * eval_point[0, 0] + 24 * eval_point[1, 0],
                ],
            ]
        ),
    )

    print("hessian helpers smoke test passed")
