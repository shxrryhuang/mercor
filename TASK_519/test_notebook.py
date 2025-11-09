import nbformat
import pytest
import numpy as np
from nbconvert.preprocessors import ExecutePreprocessor
from scipy.optimize import fsolve


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

    # Check variables exist
    for var in [
        "bisection_root", "newton_root",
        "bisection_error", "newton_error",
        "convergence_data", "best_method"
    ]:
        assert var in ns, f"{var} not found in notebook"

    f = lambda x: np.cos(x) - x/2
    exact_root = fsolve(f, 1)[0]

    # Check approximate root values
    assert 0 < ns["bisection_root"] < 2
    assert 0 < ns["newton_root"] < 2

    # Check error correctness
    bis_err_expected = abs(exact_root - ns["bisection_root"])
    new_err_expected = abs(exact_root - ns["newton_root"])
    assert np.isclose(ns["bisection_error"], bis_err_expected, rtol=1e-9)
    assert np.isclose(ns["newton_error"], new_err_expected, rtol=1e-9)

    # Check convergence data
    conv = ns["convergence_data"]
    assert isinstance(conv, dict)
    for tol, vals in conv.items():
        assert isinstance(vals, tuple) and len(vals) == 4
        assert all(isinstance(v, (float, np.floating)) for v in vals)

    # Check best method
    assert ns["best_method"] in ["bisection", "newton"]
    assert ns["newton_error"] < ns["bisection_error"]
