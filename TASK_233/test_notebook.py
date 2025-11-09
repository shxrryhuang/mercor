import nbformat, pytest, numpy as np, pandas as pd
from nbconvert.preprocessors import ExecutePreprocessor

@pytest.mark.parametrize("notebook", ["final_notebook.ipynb"])
def test_notebook_exec(notebook):
    with open(notebook) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        ep.preprocess(nb)

def get_notebook_namespace(path):
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)
    ns = {}
    for cell in nb.cells:
        if cell.cell_type == "code":
            exec(cell.source, ns)
    return ns

def test_std_factory():
    ns = get_notebook_namespace("final_notebook.ipynb")
    df = pd.DataFrame({
        "factory": ["A","A","A","B","B","B","C","C","C"],
        "quality_score": [80,82,85,78,77,79,90,92,91]
    })
    expected_stats = df.groupby("factory")["quality_score"].std()
    assert np.allclose(ns["factory_stats"].values, expected_stats.values)
    assert ns["highest_std_factory"] == expected_stats.idxmax()
