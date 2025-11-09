import nbformat
import pytest
import pandas as pd
from nbconvert.preprocessors import ExecutePreprocessor


@pytest.mark.parametrize("notebook", ["final_notebook.ipynb"])
def test_notebook_exec(notebook):
    """Test that the final notebook executes without errors."""
    with open(notebook) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        try:
            ep.preprocess(nb)
        except Exception as e:
            pytest.fail(f"Notebook execution failed: {e}")


def get_notebook_namespace(notebook_path):
    """Execute all code cells and return the namespace."""
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    ns = {}
    for cell in nb.cells:
        if cell.cell_type == "code":
            exec(cell.source, ns)
    return ns


def test_subscription_report_output():
    """Verify that subscription_report() and subscription_summary meet requirements."""
    ns = get_notebook_namespace("final_notebook.ipynb")

    # Function existence
    assert "subscription_report" in ns, "Function subscription_report() is missing."
    assert callable(ns["subscription_report"]), "subscription_report must be callable."

    # Variable existence
    assert "subscription_summary" in ns, "Variable subscription_summary not found."
    subscription_summary = ns["subscription_summary"]

    # Type checks
    assert isinstance(subscription_summary, pd.DataFrame), "subscription_summary must be a DataFrame."

    # Structure checks
    expected_cols = {"total_customers", "churn_rate"}
    assert set(subscription_summary.columns) == expected_cols, (
        f"Expected columns {expected_cols}, got {set(subscription_summary.columns)}."
    )

    # Index checks (subscription types)
    expected_index = {"Basic", "Standard", "Premium"}
    assert set(subscription_summary.index) == expected_index, (
        f"Expected index {expected_index}, got {set(subscription_summary.index)}."
    )

    # Value checks
    for val in subscription_summary["churn_rate"]:
        assert isinstance(val, float), "churn_rate values must be floats."
        assert 0 <= val <= 1, f"Churn rate {val} out of expected [0,1] range."
        assert round(val, 2) == val, f"Churn rate {val} must be rounded to 2 decimals."

    for val in subscription_summary["total_customers"]:
        assert isinstance(val, (int, float)), "total_customers must be numeric."
        assert val > 0, "Each subscription type must have at least one customer."
