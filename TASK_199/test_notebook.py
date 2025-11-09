import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor
import numpy as np


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
    ns = get_notebook_namespace("final_notebook.ipynb")

    data = [
        ("A", 80, 2),
        ("B", 70, 3),
        ("A", 90, 1),
        ("C", 85, 2),
        ("A", 75, 4)
    ]
    category = "A"
    filtered = [(score, weight) for cat, score, weight in data if cat == category]
    expected_avg = sum(score * weight for score, weight in filtered) / sum(weight for _, weight in filtered)

    assert np.isclose(ns["weighted_avg"], expected_avg), \
        f"Expected weighted average {expected_avg}, got {ns['weighted_avg']}"
