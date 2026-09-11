# Haar wavelet collocation — reference implementation

Companion code to `Paper2_Haar_Coupled_HilferHadamard_v2.docx`, §6.

## Files
- `haar_core.py` — normalized Haar basis, operational matrix of fractional
  integration (Lemma 3.3), and the boundary-functional evaluators:
  `boundary_functional_row` for genuine RL integral BCs (Lemma 3.4),
  `point_eval_row` (nearest-cell, kept for reference/comparison) and
  `point_eval_row_linear` (linear interpolation, now the default — see
  `superconvergence_test.py` for why) for pure point-evaluation BCs.
  Run directly for self-tests against closed-form fractional integrals.
- `solver.py` — `CoupledProblem` (now accepts optional `f_grad`/`g_grad` for
  the analytic block Jacobian, manuscript (4.3)) + `solve_problem(problem, J)`.
- `examples.py` — `make_example_A()` (Carpathian 2024, Ex. 4.1, three-point)
  and `make_example_B(forcing=0.0)` (AIMS 2024, Ex. 5.2, multi-point integral;
  `forcing` is an additive constant added to both f and g — see manuscript
  §6.2 for why this is needed to get a nontrivial solution). Both now supply
  analytic gradients.
- `convergence_study.py` — runs both examples at M=4..64, reports the Newton
  residual, an independent BC-error check (via `scipy.integrate.quad` on the
  continuous boundary integral, not the discrete operator), and the
  estimated convergence rate. This is what produced Tables 1-2.
- `superconvergence_test.py` — method-of-manufactured-solutions check of
  Corollary 5.9 (node superconvergence). Produced Table 3 and the finding
  that a naive point-evaluation boundary functional has a mesh-alignment
  floor that swamps node superconvergence, fixed by linear interpolation.
- `make_plots.py` — produces the M=32 solution-profile figures.

## Reproduce
```bash
pip install numpy scipy matplotlib
python convergence_study.py       # Tables 1-2
python superconvergence_test.py   # Table 3
python make_plots.py              # Figures 1-2
```

## Key numerical findings
1. AIMS 2024 Example 5.2, as published, is homogeneous (f(t,0,0)=g(t,0,0)=0,
   homogeneous BCs), so its unique solution (guaranteed by Xi<1) is exactly
   u=v=0 — confirmed to the last bit at every M. `make_example_B(forcing=0.01)`
   adds a small additive constant that leaves the Lipschitz constants (hence
   Xi~0.704) exactly unchanged while making the solution nontrivial.
2. Carpathian 2024 Example 4.1's own Lipschitz constants give Xi~2.47>1, so
   the manuscript's unconditional-solvability theorem does not formally cover
   it; Newton converges anyway (see finding 3).
3. With the explicit analytic Jacobian (4.3), Newton converges in a constant
   9 function evaluations regardless of resolution M — vs. growing with M
   when scipy estimates the Jacobian by finite differences.
4. Corollary 5.9 (O(1/M^2) at nodes) holds exactly for genuine integral
   boundary conditions (Lemma 3.4), confirmed via manufactured solution. For
   pure point-evaluation boundary conditions, a naive nearest-cell pick has
   its own O(h) floor that swamps this; linear interpolation fixes it.
