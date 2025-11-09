import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor
import numpy as np


@pytest.mark.parametrize("notebook", ["final_notebook.ipynb"])
def test_notebook_exec(notebook):
    """Ensure inventory notebook runs fully without failure."""
    with open(notebook) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        assert ep.preprocess(nb) is not None


def get_notebook_namespace(path):
    """Load notebook and execute all code."""
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)
    ns = {}
    for cell in nb.cells:
        if cell.cell_type == "code":
            exec(cell.source, ns)
    return ns


def test_predict_turnover_function_exists():
    ns = get_notebook_namespace("final_notebook.ipynb")
    assert "predict_turnover" in ns
    assert callable(ns["predict_turnover"])


def test_predict_turnover_output_structure():
    ns = get_notebook_namespace("final_notebook.ipynb")
    result = ns["predict_turnover"]()
    assert isinstance(result, dict)
    assert set(result.keys()) == {"avg_turnover", "predicted_turnover"}


def test_predict_turnover_computation_accuracy():
    ns = get_notebook_namespace("final_notebook.ipynb")
    df = ns["df"]
    result = ns["predict_turnover"]()

    q2_total = df["q2_units"].sum()
    q3_total = df["q3_units"].sum()
    q4_total = df["q4_units"].sum()
    expected_avg = (q2_total + q3_total + q4_total) / 3

    quarters = np.array([1, 2, 3])
    totals = np.array([q2_total, q3_total, q4_total])
    coeffs = np.polyfit(quarters, totals, 1)
    expected_pred = np.polyval(coeffs, 4)

    assert np.isclose(result["avg_turnover"], round(expected_avg, 2), rtol=0.001)
    assert np.isclose(result["predicted_turnover"], round(expected_pred, 2), rtol=0.001)


def test_growth_rate_exists_and_valid():
    ns = get_notebook_namespace("final_notebook.ipynb")
    df = ns["df"]
    assert "growth_pct" in df.columns
    assert np.all(np.isfinite(df["growth_pct"]))
    assert df["growth_pct"].mean() > 0
