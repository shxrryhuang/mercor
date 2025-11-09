import nbformat
import pytest
import numpy as np
from nbconvert.preprocessors import ExecutePreprocessor

@pytest.mark.parametrize("notebook", ["final_notebook.ipynb"])
def test_notebook_exec(notebook):
    """Ensure notebook executes without errors."""
    with open(notebook) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        np.random.seed(7)
        ep.preprocess(nb)

def get_notebook_namespace(path):
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)
    ns = {}
    for cell in nb.cells:
        if cell.cell_type == "code":
            exec(cell.source, ns)
    return ns


def test_notebook_variables():
    """Ensure required variables exist and basic consistency checks pass."""
    ns = get_notebook_namespace("final_notebook.ipynb")

    required = [
        "cv_scores", "cv_mean_score", "cv_std_score",
        "stratified_cv_scores", "stratified_cv_mean", "stratified_cv_std",
        "cv_f1_scores", "cv_f1_mean",
        "cv_roc_auc_scores", "cv_roc_auc_mean"
    ]

    # Check all expected variables exist
    for var in required:
        assert var in ns, f"{var} missing"

    # Cross-validation scores should be numpy arrays
    assert isinstance(ns["cv_scores"], np.ndarray)
    assert isinstance(ns["stratified_cv_scores"], np.ndarray)
    assert isinstance(ns["cv_f1_scores"], np.ndarray)
    assert isinstance(ns["cv_roc_auc_scores"], np.ndarray)

    # Each CV array should contain 5 folds
    for name in ["cv_scores", "stratified_cv_scores", "cv_f1_scores", "cv_roc_auc_scores"]:
        assert len(ns[name]) == 5, f"{name} should have 5 folds (cv=5)"

    # Mean and std must be consistent with the actual data
    assert np.isclose(ns["cv_mean_score"], ns["cv_scores"].mean()), "cv_mean_score mismatch"
    assert np.isclose(ns["cv_std_score"], ns["cv_scores"].std()), "cv_std_score mismatch"
    assert np.isclose(ns["stratified_cv_mean"], ns["stratified_cv_scores"].mean()), "stratified_cv_mean mismatch"
    assert np.isclose(ns["stratified_cv_std"], ns["stratified_cv_scores"].std()), "stratified_cv_std mismatch"

    # Check mean F1 and ROC AUC calculations
    assert np.isclose(ns["cv_f1_mean"], ns["cv_f1_scores"].mean()), "cv_f1_mean mismatch"
    assert np.isclose(ns["cv_roc_auc_mean"], ns["cv_roc_auc_scores"].mean()), "cv_roc_auc_mean mismatch"


def test_cv_scores_reasonable():
    """Validate that metrics are within expected bounds."""
    ns = get_notebook_namespace("final_notebook.ipynb")

    # Accuracy, F1, ROC AUC values should all be between 0 and 1
    for metric in ["cv_scores", "stratified_cv_scores", "cv_f1_scores", "cv_roc_auc_scores"]:
        arr = ns[metric]
        assert np.all((arr >= 0) & (arr <= 1)), f"{metric} contains invalid values"

    # ROC AUC should typically be >= F1 for classification
    assert ns["cv_roc_auc_mean"] >= ns["cv_f1_mean"] * 0.8, \
        "ROC AUC mean seems inconsistent with F1 mean"

    # Accuracy mean should generally be above 0.7 for this dataset
    assert ns["cv_mean_score"] > 0.7, "Model accuracy too low; may indicate training issue"
