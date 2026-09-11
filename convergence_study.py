import numpy as np
from scipy.integrate import quad
from scipy.special import gamma as Gamma

from examples import make_example_A, make_example_B
from solver import solve_problem, reconstruct_at


def true_bc_error(res, problem):
    """
    Independent accuracy check: numerically integrate the TRUE continuous
    boundary condition against the piecewise-constant reconstruction of the
    computed solution (via scipy.integrate.quad, NOT the Haar operational
    matrix), and compare to u(e), v(e) as reconstructed from the solution
    itself. This is a genuine consistency check distinct from the Newton
    residual (which is ~0 by construction of (S3)-(S4) using the discrete
    boundary operator).
    """
    U, V, N = res["U"], res["V"], res["N"]
    bc = problem.bc

    def u_of_t(t):
        x = np.log(t)
        return reconstruct_at(None, U, np.array([x]))[0]

    def v_of_t(t):
        x = np.log(t)
        return reconstruct_at(None, V, np.array([x]))[0]

    # u(e) as reconstructed directly (top cell of U)
    u_e = U[-1]
    v_e = V[-1]

    if bc["mode"] == "point":
        rhs_u = bc["lam"] * v_of_t(bc["theta"])
        rhs_v = bc["mu"] * u_of_t(bc["eta"])
    else:  # integral
        rhs_u = 0.0
        for lam_i, delta_i, theta_i in zip(bc["lam"], bc["delta"], bc["theta"]):
            val, _ = quad(lambda s: (theta_i - s) ** (delta_i - 1) * v_of_t(s),
                           1.0, theta_i, limit=200)
            rhs_u += lam_i * val / Gamma(delta_i)
        rhs_v = 0.0
        for mu_j, sigma_j, eta_j in zip(bc["mu"], bc["sigma"], bc["eta"]):
            val, _ = quad(lambda s: (eta_j - s) ** (sigma_j - 1) * u_of_t(s),
                           1.0, eta_j, limit=200)
            rhs_v += mu_j * val / Gamma(sigma_j)

    err_u = abs(u_e - rhs_u)
    err_v = abs(v_e - rhs_v)
    return max(err_u, err_v)


def run_study(problem, Js, label):
    results = {}
    for J in Js:
        res = solve_problem(problem, J)
        results[J] = res
        print(f"[{label}] J={J:2d} M={2**J:3d} N={res['N']:4d} "
              f"converged={res['converged']} newton_res={res['residual_norm']:.2e} "
              f"maxU={np.max(np.abs(res['U'])):.6f} maxV={np.max(np.abs(res['V'])):.6f}")

    Jmax = max(Js)
    Uref, Vref = results[Jmax]["U"], results[Jmax]["V"]

    print(f"\n{'M':>6} {'BC error':>12} {'selfConvErr':>14} {'rate':>8}")
    prev_err = None
    rows = []
    for J in Js:
        res = results[J]
        bc_err = true_bc_error(res, problem)
        # self-convergence: reconstruct this level's solution at Jmax's
        # collocation nodes, compare to the Jmax solution there
        Nref = results[Jmax]["N"]
        x_ref = (np.arange(1, Nref + 1) - 0.5) / Nref
        U_at_ref = reconstruct_at(None, res["U"], x_ref)
        V_at_ref = reconstruct_at(None, res["V"], x_ref)
        err = np.max(np.abs(U_at_ref - Uref)) + np.max(np.abs(V_at_ref - Vref))
        rate = np.log2(prev_err / err) if (prev_err is not None and err > 0) else float("nan")
        prev_err = err
        print(f"{2**J:6d} {bc_err:12.3e} {err:14.3e} {rate:8.2f}")
        rows.append({"M": 2 ** J, "bc_error": bc_err, "self_conv_err": err,
                     "rate": rate, "newton_res": res["residual_norm"]})
    return results, rows


if __name__ == "__main__":
    print("=" * 70)
    print("EXAMPLE A (Carpathian 2024, Example 4.1, three-point BCs)")
    print("=" * 70)
    probA = make_example_A()
    resA, rowsA = run_study(probA, [2, 3, 4, 5, 6], "A")

    print()
    print("=" * 70)
    print("EXAMPLE B (AIMS 2024, Example 5.2, multi-point integral BCs,")
    print("           forcing=0 -- confirming the trivial-solution finding)")
    print("=" * 70)
    probB0 = make_example_B(forcing=0.0)
    resB0, rowsB0 = run_study(probB0, [2, 3, 4, 5, 6], "B0")

    print()
    print("=" * 70)
    print("EXAMPLE B' (same as B, with small additive forcing=0.01 so the")
    print("            solution is non-trivial; Xi unchanged at ~0.704)")
    print("=" * 70)
    probBf = make_example_B(forcing=0.01)
    resBf, rowsBf = run_study(probBf, [2, 3, 4, 5, 6], "B'")
