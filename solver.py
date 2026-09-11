"""
Assembles and solves the discrete system (S1)-(S4) for the coupled
Hilfer-Hadamard system (2.5)-(2.6), using the Haar operational-matrix
machinery in haar_core.py.

Two boundary-condition modes are supported:
  - "point": u(e) = lam * v(theta), v(e) = mu * u(eta)   (three-point, Carpathian-style)
  - "integral": u(e) = sum_i lam_i * I^{delta_i} v(theta_i),
                v(e) = sum_j mu_j  * I^{sigma_j} u(eta_j)  (multi-point, AIMS-style)
"""
import numpy as np
from scipy.optimize import root
from scipy.special import gamma as Gamma

from haar_core import (build_haar_matrix, discrete_ops, boundary_functional_row,
                        point_eval_row_linear)


class CoupledProblem:
    def __init__(self, alpha1, beta1, k1, alpha2, beta2, k2, f, g, bc,
                 f_grad=None, g_grad=None):
        """
        alpha_i in (1,2], beta_i in [0,1], k_i > 0.
        f, g: callables f(t, u, v), g(t, u, v)  (vectorized over t,u,v arrays)
        f_grad, g_grad: optional callables returning (df/du, df/dv) and
            (dg/du, dg/dv) pointwise (vectorized). If supplied, the solver
            uses the explicit block Jacobian (4.3) instead of a
            finite-difference approximation.
        bc: dict describing boundary conditions, either
            {"mode": "point", "lam": lam, "theta": theta, "mu": mu, "eta": eta}
            {"mode": "integral",
             "lam": [lam_i,...], "delta": [delta_i,...], "theta": [theta_i,...],
             "mu":  [mu_j,...],  "sigma": [sigma_j,...],  "eta":  [eta_j,...]}
        """
        self.alpha1, self.beta1, self.k1 = alpha1, beta1, k1
        self.alpha2, self.beta2, self.k2 = alpha2, beta2, k2
        self.f, self.g = f, g
        self.f_grad, self.g_grad = f_grad, g_grad
        self.bc = bc
        self.gamma1 = alpha1 + 2 * beta1 - alpha1 * beta1
        self.gamma2 = alpha2 + 2 * beta2 - alpha2 * beta2


def solve_problem(problem: CoupledProblem, J, x0=None, tol=1e-11, maxiter=200):
    """
    Solve the discrete system at resolution M = 2^J (N = 2M nodes per
    unknown). Returns dict with U, V, c0, d0, nodes, t_nodes, residual_norm,
    n_iter, converged.
    """
    Htilde, nodes, idx = build_haar_matrix(J)
    N = len(nodes)
    t_nodes = np.exp(nodes)  # t = e^x

    A1, r1 = discrete_ops(problem.alpha1, J, Htilde)
    A2, r2 = discrete_ops(problem.alpha2, J, Htilde)
    A1c, r1c = discrete_ops(1.0, J, Htilde)  # for the k_i * I^1 term
    # NOTE: I^1 is the same regardless of which equation, but alpha may differ
    # from 1; we need I^1 specifically for the k_i correction term.

    e1 = nodes ** (problem.gamma1 - 1)
    e2 = nodes ** (problem.gamma2 - 1)

    bc = problem.bc
    if bc["mode"] == "point":
        # linear interpolation, not nearest-cell: the naive nearest-cell
        # pick has an O(h) floor at a theta not aligned with the mesh,
        # which swamps the interior O(1/M^2) node-superconvergence (see
        # superconvergence_test.py). Linear interpolation restores a clean
        # O(1/M) global / ~O(1/M^2)-at-nodes pattern.
        rho_v = point_eval_row_linear(bc["theta"], J, Htilde) * bc["lam"]
        rho_u = point_eval_row_linear(bc["eta"], J, Htilde) * bc["mu"]
    elif bc["mode"] == "integral":
        rho_v = np.zeros(N)
        for lam_i, delta_i, theta_i in zip(bc["lam"], bc["delta"], bc["theta"]):
            rho_v += lam_i * boundary_functional_row(delta_i, theta_i, J, Htilde)
        rho_u = np.zeros(N)
        for mu_j, sigma_j, eta_j in zip(bc["mu"], bc["sigma"], bc["eta"]):
            rho_u += mu_j * boundary_functional_row(sigma_j, eta_j, J, Htilde)
    else:
        raise ValueError("unknown bc mode")

    def unpack(z):
        U = z[0:N]
        V = z[N:2 * N]
        c0 = z[2 * N]
        d0 = z[2 * N + 1]
        return U, V, c0, d0

    def residual(z):
        U, V, c0, d0 = unpack(z)
        F = problem.f(t_nodes, U, V)
        G = problem.g(t_nodes, U, V)
        R1 = U - (c0 * e1 - problem.k1 * (A1c @ U) + A1 @ F)
        R2 = V - (d0 * e2 - problem.k2 * (A1c @ V) + A2 @ G)
        R3 = c0 - problem.k1 * (r1c @ U) + (r1 @ F) - (rho_v @ V)
        R4 = d0 - problem.k2 * (r1c @ V) + (r2 @ G) - (rho_u @ U)
        return np.concatenate([R1, R2, [R3], [R4]])

    def analytic_jacobian(z):
        """Explicit block Jacobian, manuscript (4.3)."""
        U, V, c0, d0 = unpack(z)
        f_u, f_v = problem.f_grad(t_nodes, U, V)
        g_u, g_v = problem.g_grad(t_nodes, U, V)
        I_N = np.eye(N)
        J = np.zeros((2 * N + 2, 2 * N + 2))
        # Row block 1 (dR1/d.)
        J[0:N, 0:N] = I_N + problem.k1 * A1c - A1 * f_u[np.newaxis, :]
        J[0:N, N:2 * N] = -A1 * f_v[np.newaxis, :]
        J[0:N, 2 * N] = -e1
        # Row block 2 (dR2/d.)
        J[N:2 * N, 0:N] = -A2 * g_u[np.newaxis, :]
        J[N:2 * N, N:2 * N] = I_N + problem.k2 * A1c - A2 * g_v[np.newaxis, :]
        J[N:2 * N, 2 * N + 1] = -e2
        # Row 3 (dR3/d.)
        J[2 * N, 0:N] = -problem.k1 * r1c + r1 * f_u
        J[2 * N, N:2 * N] = r1 * f_v - rho_v
        J[2 * N, 2 * N] = 1.0
        # Row 4 (dR4/d.)
        J[2 * N + 1, 0:N] = r2 * g_u - rho_u
        J[2 * N + 1, N:2 * N] = -problem.k2 * r1c + r2 * g_v
        J[2 * N + 1, 2 * N + 1] = 1.0
        return J

    use_analytic = (problem.f_grad is not None and problem.g_grad is not None)

    if x0 is None:
        z0 = np.zeros(2 * N + 2)
    else:
        z0 = x0

    if use_analytic:
        sol = root(residual, z0, jac=analytic_jacobian, method="hybr", tol=tol,
                   options={"maxfev": maxiter * (2 * N + 2)})
    else:
        sol = root(residual, z0, method="hybr", tol=tol,
                   options={"maxfev": maxiter * (2 * N + 2)})

    U, V, c0, d0 = unpack(sol.x)
    res_norm = np.max(np.abs(residual(sol.x)))
    return {
        "U": U, "V": V, "c0": c0, "d0": d0,
        "nodes": nodes, "t_nodes": t_nodes,
        "residual_norm": res_norm,
        "converged": sol.success,
        "message": sol.message,
        "z": sol.x,
        "N": N,
    }


def reconstruct_at(nodes_src, values_src, x_query):
    """Piecewise-constant reconstruction: for each x_query in [0,1], return
    the value of the cell (from a uniform mesh of width 1/len) containing it."""
    N = len(values_src)
    ell = np.floor(x_query * N).astype(int)
    ell = np.clip(ell, 0, N - 1)
    return values_src[ell]
