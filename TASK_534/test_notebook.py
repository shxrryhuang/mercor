import nbformat
import pytest
import numpy as np
from nbconvert.preprocessors import ExecutePreprocessor

@pytest.mark.parametrize("notebook", ["final_notebook.ipynb"])
def test_notebook_exec(notebook):
    """Ensure the notebook executes successfully."""
    with open(notebook) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=900, kernel_name="python3")
        ep.preprocess(nb)

def get_notebook_namespace(path):
    """Execute notebook and collect global variables."""
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)
    ns = {}
    for cell in nb.cells:
        if cell.cell_type == "code":
            exec(cell.source, ns)
    return ns

def test_notebook_variables():
    ns = get_notebook_namespace("final_notebook.ipynb")

    required = [
        "rk4_solution", "symplectic_solution",
        "rk4_error", "symplectic_error",
        "invariant_drift_rk4", "invariant_drift_symplectic",
        "convergence_data", "observed_orders",
        "fixed_point", "jacobian_eigs", "is_center",
        "best_method"
    ]
    for v in required:
        assert v in ns, f"{v} missing"

    # Shapes and types
    rk4_solution = ns["rk4_solution"]
    symp_solution = ns["symplectic_solution"]
    assert rk4_solution.shape[1] == 2 and symp_solution.shape[1] == 2

    rk4_error = float(ns["rk4_error"])
    symplectic_error = float(ns["symplectic_error"])
    assert np.isfinite(rk4_error) and np.isfinite(symplectic_error)
    assert rk4_error < symplectic_error, "RK4 should have smaller global error"

    # Invariant drift checks (stable version)
    drift_rk4 = float(ns["invariant_drift_rk4"])
    drift_symp = float(ns["invariant_drift_symplectic"])
    assert np.isfinite(drift_rk4) and np.isfinite(drift_symp)
    assert drift_rk4 < 1.0, "RK4 drift too large"
    assert drift_symp < 2.0, "Symplectic drift too large"

    # Convergence data sanity
    conv = ns["convergence_data"]
    for h in [0.5, 0.2, 0.1, 0.05]:
        assert h in conv and isinstance(conv[h], tuple)
        assert all(np.isfinite(val) for val in conv[h])

    # Observed orders
    orders = ns["observed_orders"]
    for key in [(0.5,0.2), (0.2,0.1), (0.1,0.05)]:
        assert key in orders
        p = orders[key]
        assert 2.0 <= p <= 5.5, f"Unexpected RK4 order: {p}"

    # Fixed point stability
    fp = ns["fixed_point"]
    assert np.allclose(fp, np.array([3.0, 1.5]), atol=1e-12)
    eigs = ns["jacobian_eigs"]
    assert eigs.size == 2
    assert ns["is_center"] is True

    # Best method
    assert ns["best_method"] in ["rk4", "symplectic"]
    assert ns["best_method"] == "rk4"
