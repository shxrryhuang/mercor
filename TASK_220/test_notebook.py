import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor
import numpy as np
import pandas as pd


@pytest.mark.parametrize("notebook", ["final_notebook.ipynb"])
def test_notebook_exec(notebook):
    """Ensure notebook executes without errors"""
    with open(notebook) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        try:
            ep.preprocess(nb)
        except Exception as e:
            assert False, f"Failed executing {notebook}: {e}"


def get_notebook_namespace(notebook_path):
    """Execute notebook cells and return namespace"""
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    namespace = {}
    for cell in nb.cells:
        if cell.cell_type == "code":
            exec(cell.source, namespace)
    return namespace


def test_customer_trends_function_exists():
    ns = get_notebook_namespace("final_notebook.ipynb")
    assert "customer_trends" in ns, "customer_trends function not found"
    assert callable(ns["customer_trends"]), "customer_trends should be callable"


def test_customer_trends_returns_dict():
    ns = get_notebook_namespace("final_notebook.ipynb")
    result = ns["customer_trends"]()
    assert isinstance(result, dict), "customer_trends should return a dictionary"
    assert "moving_average" in result, "Missing key: 'moving_average'"
    assert "prediction" in result, "Missing key: 'prediction'"


def test_customer_trends_moving_average():
    ns = get_notebook_namespace("final_notebook.ipynb")
    df = ns["df"]
    expected_avg = df["revenue"].rolling(window=3).mean().iloc[-1]
    result = ns["customer_trends"]()
    assert np.isclose(result["moving_average"], expected_avg, rtol=0.01), \
        f"Expected {expected_avg}, got {result['moving_average']}"


def test_customer_trends_prediction():
    ns = get_notebook_namespace("final_notebook.ipynb")
    df = ns["df"]
    x = np.arange(len(df))
    y = df["revenue"].values
    coeffs = np.polyfit(x, y, 1)
    expected_pred = np.polyval(coeffs, len(df))
    result = ns["customer_trends"]()
    assert np.isclose(result["prediction"], expected_pred, rtol=0.01), \
        f"Expected {expected_pred}, got {result['prediction']}"


def test_customer_trends_values_reasonable():
    ns = get_notebook_namespace("final_notebook.ipynb")
    result = ns["customer_trends"]()
    assert result["moving_average"] > 0, "moving_average should be positive"
    assert result["prediction"] > result["moving_average"], \
        "prediction should indicate growth beyond recent average"
