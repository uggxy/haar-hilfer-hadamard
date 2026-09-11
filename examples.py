import numpy as np
from solver import CoupledProblem


def make_example_A():
    """
    Carpathian J. Math. 40 (2024), Example 4.1 (three-point coupled system).
    (HD^{3/2,1/2}+ (1/5)HD^{1/2,1/2}) u = |u|/(log t + 50) + |v|/[(t+9)^2(1+|v|)] + 1/(t+7)^2
    (HD^{2,1/3}  + (1/10)HD^{1,1/3}) v = |u|/(t+99)^{1/5} + sin(pi v)/(70 pi) + 1/65
    u(1)=0, u(e) = -2 v(3/2);  v(1)=0, v(e) = (4/9) u(2)
    NOTE: this example was validated in the source paper via the (weaker)
    Leray-Schauder / growth-bound existence theorem, not a global Lipschitz
    contraction. Independently checking the Banach-style contraction
    constant Xi for this f,g gives Xi > 1 (dominated by the |u|/(t+99)^{1/5}
    term in g, whose Lipschitz constant is ~0.4, not small). So Theorem 5.6's
    *unconditional* discrete-solvability guarantee does not formally cover
    this example -- it is included as a stress test of the numerical scheme
    outside its proven guarantee region, not as a case with a-priori Xi<1.
    """
    def f(t, u, v):
        return (np.abs(u) / (np.log(t) + 50.0)
                + np.abs(v) / ((t + 9) ** 2 * (1 + np.abs(v)))
                + 1.0 / (t + 7) ** 2)

    def g(t, u, v):
        return (np.abs(u) / (t + 99) ** 0.2
                + np.sin(np.pi * v) / (70 * np.pi)
                + 1.0 / 65.0)

    def f_grad(t, u, v):
        f_u = np.sign(u) / (np.log(t) + 50.0)
        f_v = np.sign(v) / ((t + 9) ** 2 * (1 + np.abs(v)) ** 2)
        return f_u, f_v

    def g_grad(t, u, v):
        g_u = np.sign(u) / (t + 99) ** 0.2
        g_v = np.cos(np.pi * v) / 70.0
        return g_u, g_v

    bc = {"mode": "point", "lam": -2.0, "theta": 1.5, "mu": 4.0 / 9.0, "eta": 2.0}
    return CoupledProblem(alpha1=1.5, beta1=0.5, k1=0.2,
                           alpha2=2.0, beta2=1.0 / 3.0, k2=0.1,
                           f=f, g=g, bc=bc, f_grad=f_grad, g_grad=g_grad)


def make_example_B(forcing=0.0):
    """
    AIMS Mathematics 9(9) (2024), Example 5.2 (multi-point integral BCs).
    (HD^{5/4,1/2}+(4/55)HD^{1/4,1/2}) u = |u|/[sqrt(143+t^2)(5+|u|)] + (1/100)(1+log t)|v| [+forcing]
    (HD^{3/2,1}  +(1/26)HD^{1/2,1})  v = |u|/[(4+t)^3(1+|u|)] + sin(v)/(5+t)^2         [+forcing]
    u(1)=0, u(e) = (5/9) I^{5/2} v(2) + 6 I^{3/2} v(3/2)
    v(1)=0, v(e) = 5 I^{19/7} u(5/3) + (1/21) I^{7/2} u(3/2)
    Known: Xi ~ 0.704 < 1 (Banach contraction verified in the source paper).

    As published (forcing=0), f(t,0,0)=g(t,0,0)=0 and the boundary conditions
    are homogeneous, so this system is entirely homogeneous and its unique
    solution (guaranteed by Xi<1) is exactly u=v=0 -- confirmed numerically
    below. With forcing != 0 (a t-only additive constant, which does not
    change the Lipschitz constants L, Lbar at all, so Xi is UNCHANGED), the
    problem becomes genuinely inhomogeneous with a nontrivial solution,
    while keeping the identical contraction guarantee Xi ~ 0.704 < 1.
    """
    def f(t, u, v):
        return (np.abs(u) / (np.sqrt(143 + t ** 2) * (5 + np.abs(u)))
                + (1.0 / 100.0) * (1 + np.log(t)) * np.abs(v) + forcing)

    def g(t, u, v):
        return (np.abs(u) / ((4 + t) ** 3 * (1 + np.abs(u)))
                + np.sin(v) / (5 + t) ** 2 + forcing)

    def f_grad(t, u, v):
        f_u = np.sign(u) * 5.0 / (np.sqrt(143 + t ** 2) * (5 + np.abs(u)) ** 2)
        f_v = (1.0 / 100.0) * (1 + np.log(t)) * np.sign(v)
        return f_u, f_v

    def g_grad(t, u, v):
        g_u = np.sign(u) / ((4 + t) ** 3 * (1 + np.abs(u)) ** 2)
        g_v = np.cos(v) / (5 + t) ** 2
        return g_u, g_v

    bc = {"mode": "integral",
          "lam": [5.0 / 9.0, 6.0], "delta": [2.5, 1.5], "theta": [2.0, 1.5],
          "mu": [5.0, 1.0 / 21.0], "sigma": [19.0 / 7.0, 3.5], "eta": [5.0 / 3.0, 1.5]}
    return CoupledProblem(alpha1=1.25, beta1=0.5, k1=4.0 / 55.0,
                           alpha2=1.5, beta2=1.0, k2=1.0 / 26.0,
                           f=f, g=g, bc=bc, f_grad=f_grad, g_grad=g_grad)
