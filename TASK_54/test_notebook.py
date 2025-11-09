import nbformat
import pytest
import pandas as pd
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
    """Test that notebook variables have expected values for churn analysis."""
    ns = get_notebook_namespace("final_notebook.ipynb")

    # Verify required variables and functions exist
    assert "df" in ns, "DataFrame 'df' not found"
    assert isinstance(ns["df"], pd.DataFrame), "'df' must be a pandas DataFrame"

    assert "churn_summary" in ns, "Function 'churn_summary' not found"
    assert callable(ns["churn_summary"]), "'churn_summary' should be callable"

    assert "churn_pivot" in ns, "Variable 'churn_pivot' not found"
    assert isinstance(ns["churn_pivot"], pd.DataFrame), "'churn_pivot' must be a pandas DataFrame"

    # Validate churn_pivot structure
    churn_pivot = ns["churn_pivot"]
    expected_index = {"Basic", "Standard", "Premium"}
    expected_columns = {"Female", "Male"}

    assert set(churn_pivot.index) == expected_index, f"Expected subscription types {expected_index}, got {set(churn_pivot.index)}"
    assert set(churn_pivot.columns) == expected_columns, f"Expected genders {expected_columns}, got {set(churn_pivot.columns)}"

    # Validate numeric contents and range
    for value in churn_pivot.to_numpy().flatten():
        assert isinstance(value, float), "All churn values should be floats"
        assert 0 <= value <= 1, f"Churn rate {value} out of expected range [0,1]"
        assert round(value, 2) == value, f"Churn value {value} should be rounded to 2 decimals"
