import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor
import numpy as np


@pytest.mark.parametrize("notebook", ["final_notebook.ipynb"])
def test_notebook_exec(notebook):
    """Ensure the notebook executes completely without errors."""
    with open(notebook) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=800, kernel_name="python3")
        assert ep.preprocess(nb) is not None, f"Failed executing {notebook}"


def get_namespace(path):
    """Execute notebook cells and return global variable namespace."""
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)
    ns = {}
    for cell in nb.cells:
        if cell.cell_type == "code":
            exec(cell.source, ns)
    return ns


def test_cross_validation_and_stratified():
    """Validate all CV and model comparison variables from the churn prompt."""
    ns = get_namespace("final_notebook.ipynb")

    # Logistic Regression CV results
    for var in ["cv_accuracy_scores", "cv_accuracy_mean", "cv_accuracy_std"]:
        assert var in ns, f"{var} missing from notebook namespace"

    scores = ns["cv_accuracy_scores"]
    assert isinstance(scores, np.ndarray)
    assert len(scores) == 5, "Must use 5 folds for CV"
    assert np.isclose(ns["cv_accuracy_mean"], scores.mean(), rtol=1e-5)
    assert np.isclose(ns["cv_accuracy_std"], scores.std(), rtol=1e-5)

    # Random Forest CV results
    for var in ["rf_cv_scores", "rf_cv_mean", "rf_cv_std"]:
        assert var in ns, f"{var} missing"
    rf_scores = ns["rf_cv_scores"]
    assert isinstance(rf_scores, np.ndarray)
    assert len(rf_scores) == 5
    assert np.isclose(ns["rf_cv_mean"], rf_scores.mean(), rtol=1e-5)
    assert np.isclose(ns["rf_cv_std"], rf_scores.std(), rtol=1e-5)

    # Stratified validation
    for var in ["stratified_lr_scores", "stratified_rf_scores"]:
        assert var in ns, f"{var} missing"
        val = ns[var]
        assert isinstance(val, np.ndarray)
        assert len(val) == 5
        assert np.all((val >= 0) & (val <= 1)), f"{var} must contain accuracy values in [0,1]"

    # Model existence and fit confirmation
    assert "lr" in ns and "rf" in ns, "Both LogisticRegression and RandomForestClassifier must exist"
    assert hasattr(ns["lr"], "predict")
    assert hasattr(ns["rf"], "feature_importances_")

    # Sanity checks for metric ranges
    all_scores = np.concatenate([
        ns["cv_accuracy_scores"], ns["rf_cv_scores"],
        ns["stratified_lr_scores"], ns["stratified_rf_scores"]
    ])
    assert np.all((all_scores >= 0) & (all_scores <= 1)), "All accuracy scores must be in [0,1]"

    for metric in ["cv_accuracy_mean", "cv_accuracy_std", "rf_cv_mean", "rf_cv_std"]:
        assert np.isfinite(ns[metric]), f"{metric} must be finite"


def test_score_shapes_and_dtypes():
    """Ensure all arrays are 1D float arrays of shape (5,)."""
    ns = get_namespace("final_notebook.ipynb")

    vars_to_check = [
        "cv_accuracy_scores", "rf_cv_scores",
        "stratified_lr_scores", "stratified_rf_scores"
    ]
    for var in vars_to_check:
        arr = ns[var]
        assert arr.ndim == 1, f"{var} must be 1D"
        assert arr.shape == (5,), f"{var} must have shape (5,)"
        assert np.issubdtype(arr.dtype, np.floating), f"{var} must contain floats"


def test_stratified_consistency():
    """Check stratified scores are roughly similar to normal CV results."""
    ns = get_namespace("final_notebook.ipynb")

    lr_diff = abs(ns["cv_accuracy_mean"] - ns["stratified_lr_scores"].mean())
    rf_diff = abs(ns["rf_cv_mean"] - ns["stratified_rf_scores"].mean())

    assert lr_diff < 0.05, f"Stratified vs CV LR diff too high: {lr_diff}"
    assert rf_diff < 0.05, f"Stratified vs CV RF diff too high: {rf_diff}"


def test_rf_vs_lr_performance():
    """Check that RandomForest performs as well or better on average than Logistic Regression."""
    ns = get_namespace("final_notebook.ipynb")
    assert ns["rf_cv_mean"] >= ns["cv_accuracy_mean"] - 0.05, (
        f"Random Forest mean ({ns['rf_cv_mean']}) should not be worse than Logistic Regression by more than 0.05"
    )


def test_random_forest_feature_importances():
    """Validate Random Forest feature importance vector."""
    ns = get_namespace("final_notebook.ipynb")
    rf = ns["rf"]
    fi = rf.feature_importances_
    assert np.isclose(fi.sum(), 1.0, rtol=1e-3), "Feature importances must sum to ~1"
    assert len(fi) == ns["X_train"].shape[1], "Feature importance length must match number of features"


def test_result_print_ranges():
    """Check that printed mean accuracies fall in expected realistic range (0.4–1.0)."""
    ns = get_namespace("final_notebook.ipynb")
    for var in ["cv_accuracy_mean", "rf_cv_mean"]:
        assert 0.4 <= ns[var] <= 1.0, f"{var} = {ns[var]} seems unrealistic"
