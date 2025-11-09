Heat Equation Solver
Implementation and Comparison of FTCS vs Crank–Nicolson

Complete the notebook by fixing issues and implementing the required functionality below. Use only numpy and scipy (no plotting required).

PDE
Solve the 1D heat equation on the domain x ∈ [0, 1], t ∈ [0, 0.1]:
u_t = α u_xx, with α = 1.
Boundary conditions: u(0, t) = 0, u(1, t) = 0.
Initial condition: u(x, 0) = sin(π x).
The exact solution is u(x, t) = exp(−π² t) sin(π x).

Use spatial grid with N interior points (so Δx = 1/(N+1)) and time step Δt (choose Δt per instructions below). Evolve from t = 0 to T = 0.1 and report the solution at T.

1) Fix the FTCS scheme (explicit)
The FTCS update should be:
r = α Δt / Δx²
u^{n+1}i = u^n_i + r (u^n{i−1} − 2 u^n_i + u^n_{i+1}) for i=1..N
Enforce Dirichlet boundaries at each time step.

What to do:
Implement correct FTCS using r (many drafts forget the r factor or the neighbor terms).
Choose Δt = 0.4 * Δx² / α (so r = 0.4 is stable).
Build the FTCS amplification matrix G (N×N) with tridiagonal structure:
main diag = (1 − 2r), off-diags = r

Compute its spectral radius (largest |eigenvalue|) and store in amplification_spectral_radius.
Run FTCS to time T and store the final L2 error vs exact in ftcs_error and the final solution vector (length N) in ftcs_solution.

2) Implement Crank–Nicolson (implicit, second order in time)
The CN scheme can be written as:
r = α Δt / Δx²
(I − (r/2) T) u^{n+1} = (I + (r/2) T) u^n where T is the standard 1D Laplacian stencil on interior points (tridiagonal with −2 on diag, +1 on off-diags).

What to do:
Construct matrices A = (I − (r/2) T) and B = (I + (r/2) T).
At each time step, solve A u^{n+1} = B u^n (use scipy.linalg.solve).
Store final solution as cn_solution (length N).
Compute final L2 error vs exact as cn_error.
Compute and store the 2-norm condition number of A as A_CN_condition_number via np.linalg.cond(A).

3) Convergence analysis
Run both methods for N in [20, 40, 80].
For each N, set Δt = 0.4 * Δx² / α (same Δt for both methods).
Evolve to T = 0.1.

Store results in a dict convergence_data with:
keys = N
values = tuple (ftcs_error, cn_error).
Compute observed convergence orders for CN between successive grid refinements:
order_CN(N→2N) = log2( error_CN(N) / error_CN(2N) )
Store in a dict observed_orders with keys (20,40) and (40,80).

4) Stability report
Set r = α Δt / Δx² for the last run (N=80).
FTCS is stable if r ≤ 0.5.
Store stability_ok as True if r ≤ 0.5 else False.
Build dict stability_report with keys:
"r" → r (float)
"ftcs_stable" → stability_ok (bool)
"amplification_spectral_radius" → amplification_spectral_radius (float) from the last run
"cn_unconditionally_stable" → True (bool)
"A_CN_condition_number" → A_CN_condition_number (float) from the last run

5) Best method
Set best_method = "crank_nicolson" if cn_error < ftcs_error at N=80, else "ftcs".