import nbformat
import pytest
import numpy as np
from nbconvert.preprocessors import ExecutePreprocessor

@pytest.mark.parametrize("notebook", ["final_notebook.ipynb"])
def test_notebook_exec(notebook):
    """Ensure notebook executes cleanly."""
    with open(notebook) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        try:
            ep.preprocess(nb)
        except Exception as e:
            assert False, f"Failed executing {notebook}: {e}"

def get_notebook_namespace(notebook_path):
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    ns = {}
    for cell in nb.cells:
        if cell.cell_type == "code":
            exec(cell.source, ns)
    return ns

def test_cross_validation_results():
    ns = get_notebook_namespace("final_notebook.ipynb")

    # Basic CV
    assert "cv_scores" in ns and len(ns["cv_scores"]) == 5
    assert np.isclose(ns["cv_mean_score"], np.mean(ns["cv_scores"]))
    assert np.isclose(ns["cv_std_score"], np.std(ns["cv_scores"]))

    # Stratified CV
    assert "stratified_cv_scores" in ns and len(ns["stratified_cv_scores"]) == 5
    assert np.isclose(ns["stratified_cv_mean"], np.mean(ns["stratified_cv_scores"]))
    assert np.isclose(ns["stratified_cv_std"], np.std(ns["stratified_cv_scores"]))

    # F1 + ROC AUC
    assert "cv_f1_scores" in ns and len(ns["cv_f1_scores"]) == 5
    assert "cv_roc_auc_scores" in ns and len(ns["cv_roc_auc_scores"]) == 5
    assert 0 <= ns["cv_f1_mean"] <= 1
    assert 0 <= ns["cv_roc_auc_mean"] <= 1

    print("\nAll tests passed successfully!")
