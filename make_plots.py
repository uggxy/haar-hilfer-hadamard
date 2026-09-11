import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from examples import make_example_A, make_example_B
from solver import solve_problem

plt.rcParams.update({"font.size": 11, "font.family": "serif"})


def profile_plot(res, title, fname):
    t = res["t_nodes"]
    U, V = res["U"], res["V"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ax, y, lbl, color in zip(axes, [U, V], ["$u_M(t)$", "$v_M(t)$"], ["#2b6cb0", "#c05621"]):
        ax.step(t, y, where="mid", color=color, lw=1.6)
        ax.set_xlabel("$t$")
        ax.set_ylabel(lbl)
        ax.grid(alpha=0.3)
        ax.set_xlim(1, np.e)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print("wrote", fname)


if __name__ == "__main__":
    probA = make_example_A()
    resA = solve_problem(probA, J=5)  # M=32
    profile_plot(resA, "Example A (Carpathian 4.1), $M=32$", "profile_A_M32.png")

    probBf = make_example_B(forcing=0.01)
    resBf = solve_problem(probBf, J=5)
    profile_plot(resBf, "Example B$'$ (AIMS 5.2, forcing$=0.01$), $M=32$", "profile_B_M32.png")
