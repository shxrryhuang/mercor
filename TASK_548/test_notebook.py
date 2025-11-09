import nbformat
import pytest
import numpy as np
from nbconvert.preprocessors import ExecutePreprocessor

@pytest.mark.parametrize("notebook", ["final_notebook.ipynb"])
def test_notebook_exec(notebook):
    with open(notebook) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        assert ep.preprocess(nb) is not None

def test_notebook_variables(path):
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)
    ns = {}
    for cell in nb.cells:
        if cell.cell_type == "code":
            exec(cell.source, ns)
    return ns

def test_notebook_variables():
    ns = test_notebook_variables("final_notebook.ipynb")
    for var in ["cv_scores","cv_mean_score","cv_std_score","stratified_cv_scores",
                "stratified_cv_mean","stratified_cv_std","cv_f1_scores","cv_f1_mean",
                "cv_roc_auc_scores","cv_roc_auc_mean"]:
        assert var in ns, f"{var} missing"
    assert isinstance(ns["cv_scores"], np.ndarray)
    assert len(ns["cv_scores"]) == 5
    assert np.isclose(ns["cv_mean_score"], ns["cv_scores"].mean())
    assert np.isclose(ns["cv_std_score"], ns["cv_scores"].std())
    assert len(ns["stratified_cv_scores"]) == 5
    assert len(ns["cv_f1_scores"]) == 5
    assert len(ns["cv_roc_auc_scores"]) == 5
