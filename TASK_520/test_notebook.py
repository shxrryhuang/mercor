import nbformat
import pytest
import numpy as np
from nbconvert.preprocessors import ExecutePreprocessor


@pytest.mark.parametrize("notebook", ["final_notebook.ipynb"])
def test_notebook_exec(notebook):
    """Test that notebooks execute without errors."""
    with open(notebook) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        try:
            assert ep.preprocess(nb) is not None, f"Got empty notebook for {notebook}"
        except Exception:
            assert False, f"Failed executing {notebook}"


def get_notebook_namespace(notebook_path):
    """Execute notebook cells and return namespace with all variables."""
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    namespace = {}
    for cell in nb.cells:
        if cell.cell_type == "code":
            exec(cell.source, namespace)

    return namespace


def test_notebook_variables():
    """Test that notebook variables have expected values."""
    ns = get_notebook_namespace("final_notebook.ipynb")

    # Required variables must exist
    required = [
        "ftcs_solution", "cn_solution",
        "ftcs_error", "cn_error",
        "amplification_spectral_radius",
        "A_CN_condition_number",
        "convergence_data", "observed_orders",
        "stability_ok", "stability_report",
        "best_method"
    ]
    for v in required:
        assert v in ns, f"Missing variable: {v}"

    # Shapes and types (last run is N=80)
    ftcs_solution = ns["ftcs_solution"]
    cn_solution = ns["cn_solution"]
    assert isinstance(ftcs_solution, np.ndarray) and ftcs_solution.ndim == 1 and ftcs_solution.size == 80
    assert isinstance(cn_solution, np.ndarray) and cn_solution.ndim == 1 and cn_solution.size == 80

    # Errors should be finite and small; CN should be better than FTCS
    ftcs_error = float(ns["ftcs_error"])
    cn_error = float(ns["cn_error"])
    assert np.isfinite(ftcs_error) and np.isfinite(cn_error)
    assert cn_error < ftcs_error, "Crank–Nicolson should be more accurate than FTCS at N=80"

    # Amplification spectral radius exists and reasonable (≤ 1 when r <= 0.5)
    amp_rad = float(ns["amplification_spectral_radius"])
    assert np.isfinite(amp_rad)
    assert amp_rad <= 1.0 + 1e-10

    # Condition number should be finite and >= 1
    condA = float(ns["A_CN_condition_number"])
    assert np.isfinite(condA) and condA >= 1.0

    # Convergence data: dict with keys 20, 40, 80 -> tuple(ftcs_err, cn_err)
    conv = ns["convergence_data"]
    assert isinstance(conv, dict)
    for N in [20, 40, 80]:
        assert N in conv, f"Missing N={N} in convergence_data"
        vals = conv[N]
        assert isinstance(vals, tuple) and len(vals) == 2
        assert all(isinstance(x, float) for x in vals)
        assert all(np.isfinite(x) for x in vals)

    # Observed orders should be around 2 for CN (second-order in time with this setup),
    # allow a generous tolerance due to short final time and coarse grids.
    orders = ns["observed_orders"]
    assert isinstance(orders, dict)
    for k in [(20,40), (40,80)]:
        assert k in orders
        p = float(orders[k])
        assert np.isfinite(p)
        assert 1.0 <= p <= 3.0, f"Observed CN order {p} out of expected range [1,3]"

    # Stability report
    st = ns["stability_report"]
    assert isinstance(st, dict)
    for key in ["r", "ftcs_stable", "amplification_spectral_radius",
                "cn_unconditionally_stable", "A_CN_condition_number"]:
        assert key in st
    assert isinstance(ns["stability_ok"], (bool, np.bool_))
    assert st["ftcs_stable"] == ns["stability_ok"]
    assert st["cn_unconditionally_stable"] is True
    assert np.isclose(st["amplification_spectral_radius"], amp_rad, rtol=1e-10, atol=1e-12)
    assert np.isclose(st["A_CN_condition_number"], condA, rtol=1e-10, atol=1e-12)
    assert st["r"] <= 0.5 + 1e-12

    # Best method selection
    assert ns["best_method"] in ["crank_nicolson", "ftcs"]
    assert ns["best_method"] == "crank_nicolson"
