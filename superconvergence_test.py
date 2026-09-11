"""
Manufactured-solution test for Corollary 5.9 (superconvergence at nodes).

There is no closed-form solution for Examples A/B, so the global BC-error
check in convergence_study.py cannot distinguish "O(1/M) everywhere" from
"O(1/M) globally but O(1/M^2) exactly at the collocation nodes". To test
the latter we build our OWN exact solution by working backwards:

Pick u_exact(x) = x^p, v_exact(x) = x^q  (x = log t in [0,1], smooth C^infty,
comfortably C^2). Using the closed-form I^alpha[x^m](x) = Gamma(m+1)/
Gamma(m+1+alpha) x^{m+alpha}, invert it term-by-term to build a forcing
F(x) [i.e. f(t,u,v) := F(log t), ignoring u,v] such that

    x^p = 0*x^{gamma1-1} - k1*I^1[x^p](x) + I^{alpha1}[F](x)

holds as an IDENTITY in x (design constant c0 = 0, chosen to avoid a mild
singularity at x=0 that a nonzero c0*x^{gamma1-1} term's inverse would
introduce into F). Same construction for v with exponent q.

Since F is now independent of u,v, the discrete system is exactly linear
(no genuine coupling nonlinearity), but it still exercises the SAME
discrete operators (A^alpha, r^alpha, boundary functionals) as the real
examples, so it is a valid and standard way (method of manufactured
solutions) to isolate the DISCRETIZATION error and check its behavior at
nodes vs elsewhere -- which is exactly what Corollary 5.9 claims.
"""
import numpy as np
from scipy.special import gamma as Gamma

from haar_core import build_haar_matrix, discrete_ops, point_eval_row
from solver import CoupledProblem, solve_problem


def make_manufactured_problem(p=3.0, q=4.0,
                               alpha1=1.5, beta1=0.5, k1=0.2,
                               alpha2=2.0, beta2=1.0 / 3.0, k2=0.1,
                               theta=1.5, eta=2.0, bc_mode="point", delta=2.0, sigma=2.0,
                               nonlin_kappa=0.0):
    """
    nonlin_kappa=0 (default) reproduces the original LINEAR manufactured
    problem (f, g independent of u, v) used for the base superconvergence
    check. nonlin_kappa != 0 adds kappa*(u^2 - u_exact(x)^2) to f and
    kappa*(v^2 - v_exact(x)^2) to g: this term is EXACTLY ZERO whenever
    u=u_exact, v=v_exact (so the manufactured solution is unchanged and
    still exact), but away from the target it is genuinely nonlinear with
    f_u = 2*kappa*u != 0, so Newton must actually do nonlinear work to find
    it rather than solving a linear system in disguise -- addressing the
    gap that the pure t-only forcing sidesteps genuine coupling entirely.
    """
    gamma1 = alpha1 + 2 * beta1 - alpha1 * beta1
    gamma2 = alpha2 + 2 * beta2 - alpha2 * beta2

    def u_exact_x(x):
        return x ** p

    def v_exact_x(x):
        return x ** q

    # F(x) such that I^{alpha1}[F](x) = x^p + (k1/(p+1)) x^{p+1}  (c0=0 design)
    # Inverting I^alpha[x^s](x) = Gamma(s+1)/Gamma(s+1+alpha) x^{s+alpha}:
    # to get I^alpha[A x^{m-alpha}](x) = x^m, need A = Gamma(m+1)/Gamma(m-alpha+1).
    c1 = Gamma(p + 1) / Gamma(p - alpha1 + 1)
    c2 = Gamma(p + 2) / Gamma(p - alpha1 + 2)

    def F_of_x(x):
        return c1 * x ** (p - alpha1) + (k1 / (p + 1)) * c2 * x ** (p + 1 - alpha1)

    d1 = Gamma(q + 1) / Gamma(q - alpha2 + 1)
    d2 = Gamma(q + 2) / Gamma(q - alpha2 + 2)

    def G_of_x(x):
        return d1 * x ** (q - alpha2) + (k2 / (q + 1)) * d2 * x ** (q + 1 - alpha2)

    def f(t, u, v):
        x = np.log(t)
        base = F_of_x(x)
        if nonlin_kappa == 0.0:
            return base
        return base + nonlin_kappa * (u ** 2 - u_exact_x(x) ** 2)

    def g(t, u, v):
        x = np.log(t)
        base = G_of_x(x)
        if nonlin_kappa == 0.0:
            return base
        return base + nonlin_kappa * (v ** 2 - v_exact_x(x) ** 2)

    def f_grad(t, u, v):
        if nonlin_kappa == 0.0:
            return np.zeros_like(t), np.zeros_like(t)
        return 2 * nonlin_kappa * u, np.zeros_like(t)

    def g_grad(t, u, v):
        if nonlin_kappa == 0.0:
            return np.zeros_like(t), np.zeros_like(t)
        return np.zeros_like(t), 2 * nonlin_kappa * v


    if bc_mode == "point":
        lam = 1.0 / (np.log(theta)) ** q
        mu = 1.0 / (np.log(eta)) ** p
        bc = {"mode": "point", "lam": lam, "theta": theta, "mu": mu, "eta": eta}
    elif bc_mode == "integral":
        # u(e) = lam * I^delta v(theta), where I^delta is an RL integral in
        # the ORIGINAL t-variable s (Lemma 3.4) -- NOT a simple power-function
        # closed form in x=log s. Must integrate v(s)=v_exact(log s) directly.
        from scipy.integrate import quad as _quad
        Ivq, _ = _quad(lambda s: (theta - s) ** (delta - 1) * v_exact_x(np.log(s)),
                        1.0, theta, limit=200)
        Ivq /= Gamma(delta)
        Iup, _ = _quad(lambda s: (eta - s) ** (sigma - 1) * u_exact_x(np.log(s)),
                        1.0, eta, limit=200)
        Iup /= Gamma(sigma)
        lam = 1.0 / Ivq
        mu = 1.0 / Iup
        bc = {"mode": "integral", "lam": [lam], "delta": [delta], "theta": [theta],
              "mu": [mu], "sigma": [sigma], "eta": [eta]}
    else:
        raise ValueError("unknown bc_mode")

    problem = CoupledProblem(alpha1=alpha1, beta1=beta1, k1=k1,
                              alpha2=alpha2, beta2=beta2, k2=k2,
                              f=f, g=g, bc=bc, f_grad=f_grad, g_grad=g_grad)
    return problem, u_exact_x, v_exact_x


def run_superconvergence_check(Js=(2, 3, 4, 5, 6, 7), bc_mode="point", nonlin_kappa=0.0):
    problem, u_exact_x, v_exact_x = make_manufactured_problem(bc_mode=bc_mode, nonlin_kappa=nonlin_kappa)

    print(f"{'M':>5} {'nodal err (U)':>14} {'offnode err (U)':>16} "
          f"{'rate(node)':>11} {'rate(off)':>10}")
    prev_node, prev_off = None, None
    rows = []
    for J in Js:
        res = solve_problem(problem, J)
        U, nodes, N = res["U"], res["nodes"], res["N"]
        h = 1.0 / N

        # nodal error: |U_h[l] - u_exact(x_l)|
        node_err = np.max(np.abs(U - u_exact_x(nodes)))

        # off-node error: sample several points inside each cell (avoiding
        # the exact midpoint), take the worst deviation of u_exact from the
        # cell's CONSTANT value U_h[l]
        offsets = np.array([-0.45, -0.25, 0.25, 0.45]) * h
        off_err = 0.0
        for l in range(N):
            xl = nodes[l]
            for off in offsets:
                x = xl + off
                if 0 <= x <= 1:
                    off_err = max(off_err, abs(U[l] - u_exact_x(x)))

        rate_node = np.log2(prev_node / node_err) if prev_node else float("nan")
        rate_off = np.log2(prev_off / off_err) if prev_off else float("nan")
        print(f"{2**J:5d} {node_err:14.4e} {off_err:16.4e} {rate_node:11.3f} {rate_off:10.3f}")
        rows.append((2 ** J, node_err, off_err, rate_node, rate_off))
        prev_node, prev_off = node_err, off_err
    return rows


if __name__ == "__main__":
    print("Manufactured solution: u_exact(x)=x^3, v_exact(x)=x^4 (C^infty)")
    print("Testing Corollary 5.9: nodal error should show rate~2, off-node rate~1")
    print()
    print("=== bc_mode = 'point' (Carpathian-style: u(e)=lam*v(theta)) ===")
    run_superconvergence_check(bc_mode="point")
    print()
    print("=== bc_mode = 'integral' (AIMS-style: u(e)=lam*I^delta v(theta)) ===")
    run_superconvergence_check(bc_mode="integral")
    print()
    print("--- Robustness check: does the rate survive GENUINE nonlinear ---")
    print("--- coupling, or only the linear (t-only forcing) proxy above? ---")
    print("=== integral BC, nonlin_kappa=0.5 (f_u = u, g_v = v away from target) ===")
    run_superconvergence_check(bc_mode="integral", nonlin_kappa=0.5)
    print()
    print("=== integral BC, nonlin_kappa=5.0 (10x stronger nonlinear term) ===")
    run_superconvergence_check(bc_mode="integral", nonlin_kappa=5.0)
