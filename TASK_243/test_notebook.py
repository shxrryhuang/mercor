import nbformat
import pytest
import numpy as np
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOK = "main_final.ipynb"

@pytest.mark.parametrize("notebook", ["final_notebook.ipynb"])
def test_notebook_exec(notebook):
    """Notebook executes without errors."""
    with open(notebook, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    try:
        ep.preprocess(nb)
    except Exception as e:
        assert False, f"Failed executing {notebook}: {e}"

def get_notebook_namespace(notebook_path):
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    ns = {}
    for cell in nb.cells:
        if cell.cell_type == "code":
            exec(cell.source, ns)
    return ns

def test_required_variables_and_types():
    ns = get_notebook_namespace("final_notebook.ipynb")
    # Required dict of metrics
    assert "model_metrics" in ns, "model_metrics not found"
    mm = ns["model_metrics"]
    expected_keys = {"accuracy","precision","recall","f1_score"}
    assert set(mm.keys()) == expected_keys, f"model_metrics keys mismatch: {set(mm.keys())}"
    for k in expected_keys:
        assert isinstance(mm[k], (float, np.floating)), f"{k} must be float"
        assert 0.0 <= mm[k] <= 1.0, f"{k} out of [0,1] range"

    # Reasonable accuracy
    assert mm["accuracy"] > 0.6, f"Accuracy too low: {mm['accuracy']}"

    # Feature importance dict
    assert "feature_importance_dict" in ns, "feature_importance_dict not found"
    fid = ns["feature_importance_dict"]
    assert isinstance(fid, dict) and len(fid) > 0, "feature_importance_dict invalid"
    total_importance = sum(fid.values())
    assert 0.95 <= total_importance <= 1.05, f"Importances should sum ~1.0, got {total_importance}"

    # Probabilities
    assert "default_probabilities" in ns, "default_probabilities not found"
    probs = ns["default_probabilities"]
    assert isinstance(probs, np.ndarray), "default_probabilities must be numpy array"
    assert np.all(probs >= 0) and np.all(probs <= 1), "Probabilities must be in [0,1]"

    # Shapes: test set should be 20% of df
    assert "df" in ns, "df not found"
    n = len(ns["df"])
    expected_test = int(0.20 * n)
    assert len(probs) == expected_test, f"Expected {expected_test} probs, got {len(probs)}"

    # Encoded feature dimensionality check
    assert "X_train" in ns, "X_train not found"
    X_train = ns["X_train"]
    # allow either DataFrame or ndarray
    num_features = X_train.shape[1]
    assert num_features >= 10, f"Expected >=10 features after encoding, got {num_features}"
