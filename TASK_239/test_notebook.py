
import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor
import numpy as np

@pytest.mark.parametrize("notebook", ["final_notebook.ipynb"])
def test_notebook_exec(notebook):
    """Ensure the notebook runs without errors."""
    with open(notebook, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    try:
        ep.preprocess(nb)
    except Exception as e:
        assert False, f"Failed executing {notebook}: {e}"

def get_notebook_namespace(notebook_path):
    """Execute notebook cells and return global namespace."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    ns = {}
    for cell in nb.cells:
        if cell.cell_type == "code":
            exec(cell.source, ns)
    return ns

def test_function_exists_and_returns_dict():
    ns = get_notebook_namespace("final_notebook.ipynb")
    assert "simulate_oscillator" in ns, "simulate_oscillator() not found"
    result = ns["simulate_oscillator"]()
    assert isinstance(result, dict), "simulate_oscillator must return a dict"

def test_keys_and_shapes():
    ns = get_notebook_namespace("final_notebook.ipynb")
    result = ns["simulate_oscillator"]()
    # Required keys
    for k in ["t", "x", "v", "max_abs_error", "energy_ratio_end", "peak_freq"]:
        assert k in result, f"Missing key: {k}"
    t, x, v = result["t"], result["x"], result["v"]
    assert isinstance(t, np.ndarray) and isinstance(x, np.ndarray) and isinstance(v, np.ndarray), "t, x, v must be numpy arrays"
    assert len(t) == len(x) == len(v) and len(t) > 1000, "t, x, v must have the same length > 1000"
    # time monotonic
    assert np.all(np.diff(t) > 0), "t must be strictly increasing"

def test_accuracy_vs_analytic_and_energy_decay_and_peak_freq():
    ns = get_notebook_namespace("final_notebook.ipynb")
    result = ns["simulate_oscillator"]()
    # Accuracy check
    assert result["max_abs_error"] < 2e-4, f"max_abs_error too large: {result['max_abs_error']}"

    # Energy decay ratio ~ exp(-c/m * T) with c=0.2, m=1, T=10 -> ~ exp(-2) ≈ 0.1353
    expected_energy_ratio = np.exp(-2.0)
    assert np.isfinite(result["energy_ratio_end"]) and result["energy_ratio_end"] > 0
    assert np.isclose(result["energy_ratio_end"], expected_energy_ratio, rtol=0.2), (
        f"energy_ratio_end {result['energy_ratio_end']} not close to expected ~{expected_energy_ratio}"
    )

