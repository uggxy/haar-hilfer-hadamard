"""
Core Haar wavelet / operational-matrix machinery for the coupled
Hilfer-Hadamard collocation scheme (manuscript Sections 3-4).

Implements:
  - normalized Haar basis on [0,1] (Prop 3.1)
  - operational matrix of RL fractional integration (Lemma 3.3)
  - discrete integration operator A^alpha, boundary row r^alpha (Sec 4.1)
  - boundary functional block for point-evaluation (Carpathian, "delta->0")
    and for genuine RL integral boundary terms (Lemma 3.4, AIMS case)
"""
import numpy as np
from scipy.special import gamma as Gamma


def haar_indices(J):
    """Return list of (i, m, kappa) for i=1..2M, M=2^J, matching (3.1)."""
    M = 2 ** J
    idx = [(1, 0, 0)]  # i=1 is the scaling function h_1 = 1
    i = 2
    for j in range(0, J + 1):
        m = 2 ** j
        for kappa in range(m):
            idx.append((i, m, kappa))
            i += 1
    assert i - 1 == 2 * M, (i - 1, 2 * M)
    return idx


def haar_breakpoints(m, kappa):
    z1 = kappa / m
    z2 = (kappa + 0.5) / m
    z3 = (kappa + 1) / m
    return z1, z2, z3


def build_haar_matrix(J):
    """
    Build the normalized Haar matrix Htilde (2M x 2M): rows = basis functions
    h_tilde_i, columns = collocation nodes x_l = (l-0.5)/(2M).
    Returns (Htilde, nodes, idx) where idx[k] = (i, m, kappa) for row k.
    """
    idx = haar_indices(J)
    N = len(idx)  # = 2M
    nodes = (np.arange(1, N + 1) - 0.5) / N
    Htilde = np.zeros((N, N))
    for row, (i, m, kappa) in enumerate(idx):
        if i == 1:
            Htilde[row, :] = 1.0
        else:
            z1, z2, z3 = haar_breakpoints(m, kappa)
            vals = np.where((nodes >= z1) & (nodes < z2), 1.0,
                    np.where((nodes >= z2) & (nodes < z3), -1.0, 0.0))
            Htilde[row, :] = np.sqrt(m) * vals
    return Htilde, nodes, idx


def frac_int_indicator(alpha, c, d, x):
    """(I^alpha 1_[c,d))(x), Lemma 3.2, vectorized over x."""
    xc = np.clip(x - c, 0, None) ** alpha
    xd = np.clip(x - d, 0, None) ** alpha
    return (xc - xd) / Gamma(alpha + 1)


def op_matrix_P(alpha, J, eval_points=None):
    """
    Operational matrix (P^alpha)_{i,l} = (I^alpha htilde_i)(x_l), Lemma 3.3.
    If eval_points is None, evaluate at the collocation nodes (2M x 2M matrix).
    Otherwise evaluate at the given points (2M x len(eval_points)).
    """
    idx = haar_indices(J)
    N = len(idx)
    if eval_points is None:
        eval_points = (np.arange(1, N + 1) - 0.5) / N
    eval_points = np.asarray(eval_points, dtype=float)
    P = np.zeros((N, len(eval_points)))
    for row, (i, m, kappa) in enumerate(idx):
        if i == 1:
            P[row, :] = eval_points ** alpha / Gamma(alpha + 1)
        else:
            z1, z2, z3 = haar_breakpoints(m, kappa)
            term = (frac_int_indicator(alpha, z1, z2, eval_points)
                    - frac_int_indicator(alpha, z2, z3, eval_points))
            # matches boxed (3.3): (x-z1)^a -2(x-z2)^a+(x-z3)^a, all /Gamma(a+1)
            P[row, :] = np.sqrt(m) * term
    return P


def discrete_ops(alpha, J, Htilde=None):
    """
    Build A^alpha (2M x 2M) and r^alpha (row, length 2M): manuscript (4.1).
    A^alpha = (1/N) P^{alpha,T} Htilde   [N = 2M]
    r^alpha = (1/N) (p^alpha)^T Htilde,  p^alpha_i = (I^alpha htilde_i)(1)
    """
    idx = haar_indices(J)
    N = len(idx)
    if Htilde is None:
        Htilde, nodes, idx = build_haar_matrix(J)
    P = op_matrix_P(alpha, J)  # N x N, columns = nodes
    A = (1.0 / N) * (P.T @ Htilde)
    p_at_1 = op_matrix_P(alpha, J, eval_points=np.array([1.0])).flatten()  # length N
    r = (1.0 / N) * (p_at_1 @ Htilde)  # length N row vector
    return A, r


def boundary_functional_row(delta, theta, J, Htilde=None):
    """
    Row vector rho such that rho @ V approximates I^delta v(theta) (Lemma 3.4),
    for a genuine RL integral boundary term (delta > 0).
    q^delta(theta)_i = (I^delta [h_i(log .)])(theta), theta given in the
    ORIGINAL t-variable (t in [1,e]); a_r = exp(zeta_r) are the breakpoints
    mapped back to t-space and clipped at theta.
    """
    idx = haar_indices(J)
    N = len(idx)
    if Htilde is None:
        Htilde, nodes, idx = build_haar_matrix(J)
    q = np.zeros(N)
    for row, (i, m, kappa) in enumerate(idx):
        if i == 1:
            z1, z2, z3 = 0.0, 1.0, 1.0  # h_1 = 1 on all of [0,1) -> a1=1,a2=e
            a1, a2 = 1.0, np.e
            ah1, ah2 = min(a1, theta), min(a2, theta)
            q[row] = ((theta - ah1) ** delta - (theta - ah2) ** delta) / Gamma(delta + 1)
        else:
            z1, z2, z3 = haar_breakpoints(m, kappa)
            a1, a2, a3 = np.exp(z1), np.exp(z2), np.exp(z3)
            ah1, ah2, ah3 = min(a1, theta), min(a2, theta), min(a3, theta)
            val = ((theta - ah1) ** delta - 2 * (theta - ah2) ** delta
                   + (theta - ah3) ** delta) / Gamma(delta + 1)
            q[row] = np.sqrt(m) * val
    rho = (1.0 / N) * (q @ Htilde)
    return rho


def point_eval_row(theta, J, Htilde=None):
    """
    Row vector rho such that rho @ V gives the exact piecewise-constant
    reconstruction of v evaluated at t=theta (the delta->0 / pure point
    evaluation case used in the three-point Carpathian-style boundary
    conditions). x0 = log(theta) in [0,1]; picks the Haar cell containing x0.
    """
    idx = haar_indices(J)
    N = len(idx)
    if Htilde is None:
        Htilde, nodes, idx = build_haar_matrix(J)
    x0 = np.log(theta)
    # cell width 1/N; cell ell (1-indexed) = [(ell-1)/N, ell/N)
    ell = int(np.floor(x0 * N))
    ell = min(max(ell, 0), N - 1)
    e_ell = np.zeros(N)
    e_ell[ell] = 1.0
    # V_ell = sum_i c_i htilde_i(x_ell) = e_ell . Htilde^T c ; and V = Htilde^T c
    # so picking nodal value V_ell is just selecting the ell-th nodal entry.
    return e_ell  # rho @ V = V[ell] directly (V already IS the nodal vector)


def point_eval_row_linear(theta, J, Htilde=None):
    """
    Improved point-evaluation functional: linear interpolation between the
    two nearest collocation nodes, rather than a nearest-cell pick. The
    plain nearest-cell version (point_eval_row) is only O(h) accurate at a
    generic theta not aligned with the mesh, which can swamp the O(1/M^2)
    node-superconvergence elsewhere in the scheme (see manuscript Remark on
    Corollary 5.9 / superconvergence_test.py). Linear interpolation of a
    piecewise-constant reconstruction's node values is a cheap fix.
    """
    idx = haar_indices(J)
    N = len(idx)
    if Htilde is None:
        Htilde, nodes, idx = build_haar_matrix(J)
    else:
        nodes = (np.arange(1, N + 1) - 0.5) / N
    x0 = np.log(theta)
    row = np.zeros(N)
    if x0 <= nodes[0]:
        row[0] = 1.0
    elif x0 >= nodes[-1]:
        row[-1] = 1.0
    else:
        ell = np.searchsorted(nodes, x0) - 1
        ell = min(max(ell, 0), N - 2)
        x_lo, x_hi = nodes[ell], nodes[ell + 1]
        w = (x0 - x_lo) / (x_hi - x_lo)
        row[ell] = 1.0 - w
        row[ell + 1] = w
    return row


# ---------------------------------------------------------------------------
# Self-tests: verify against closed-form fractional integrals of known
# functions, e.g. I^alpha[x^p](x) = Gamma(p+1)/Gamma(p+1+alpha) * x^{p+alpha}
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    J = 6  # M = 64, N = 128
    Htilde, nodes, idx = build_haar_matrix(J)
    N = len(nodes)
    print(f"J={J}, N=2M={N}")

    # Check orthonormality: Htilde @ Htilde.T should be N * I
    gram = Htilde @ Htilde.T
    err_orthonorm = np.max(np.abs(gram - N * np.eye(N)))
    print("max |HH^T - N I| =", err_orthonorm)

    alpha = 0.7
    A, r = discrete_ops(alpha, J, Htilde)

    # Test function y(x) = x^2 (smooth, well inside [0,1])
    p = 2.0
    y = nodes ** p
    exact = Gamma(p + 1) / Gamma(p + 1 + alpha) * nodes ** (p + alpha)
    approx = A @ y
    err = np.max(np.abs(exact - approx))
    print(f"max error I^{alpha}[x^{p}] via A^alpha, at nodes:", err)

    # test r^alpha : (I^alpha y)(1) should be Gamma(p+1)/Gamma(p+1+alpha)*1^{p+alpha}
    exact_at_1 = Gamma(p + 1) / Gamma(p + 1 + alpha)
    approx_at_1 = r @ y
    print(f"I^{alpha}[x^{p}](1): exact={exact_at_1:.6f} approx={approx_at_1:.6f}")

    # test with a non-polynomial smooth function: y = exp(x)
    y2 = np.exp(nodes)
    # no closed form; just check refinement converges by comparing J and J+1
    print("Smoke test complete.")
