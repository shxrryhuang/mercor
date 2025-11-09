mplementation and Comparison of Root-Finding Algorithms

Complete the notebook by fixing and implementing the following:

1.Fix the Bisection Method:
Currently, it does not update the midpoint or stop correctly.
Correct logic:
Compute midpoint c = (a + b)/2
If |f(c)| < tolerance, stop.
If f(a)*f(c) < 0, set b = c, else a = c.
Save the final root as bisection_root.

2.Implement the Newton-Raphson Method:
Formula: x_{n+1} = x_n - f(x_n)/f'(x_n)
with derivative f'(x) = -sin(x) - 0.5.
Iterate until |f(x)| < tolerance or until max iterations reached.
Save result as newton_root.

3.Compute Errors:
Find the exact root using scipy.optimize.fsolve.
Compute: bisection_error = abs(exact_root - bisection_root), newton_error = abs(exact_root - newton_root)

4.Convergence Analysis:
For tolerances [1e-1, 1e-3, 1e-6, 1e-9], run both methods and store results as:
convergence_data[tol] = (bisection_root, newton_root, bisection_error, newton_error)

5.Best Method:
If newton_error < bisection_error, set best_method = "newton", else "bisection".

Your notebook should define these variables:
bisection_root
newton_root
bisection_error
newton_error
convergence_data
best_method

Function: f(x) = cos(x) - x/2, interval [0, 2].