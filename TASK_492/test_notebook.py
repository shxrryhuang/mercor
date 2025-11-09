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

def test_values_correct():
    ns = get_notebook_namespace("final_notebook.ipynb")
    df = pd.DataFrame({
        "temperature": [30, 32, 35, 37, 40, 42, 43, 45, 100, 47, 49, 50],
        "energy_usage": [200, 210, 215, 220, 230, 240, 245, 250, 800, 255, 260, 265]
    })
    Q1, Q3 = df["temperature"].quantile(0.25), df["temperature"].quantile(0.75)
    IQR = Q3 - Q1
    filtered = df[(df["temperature"] >= Q1 - 1.5*IQR) & (df["temperature"] <= Q3 + 1.5*IQR)]
    expected_corr = filtered["temperature"].corr(filtered["energy_usage"])
    expected_mean = filtered["energy_usage"].mean()
    assert np.isclose(ns["correlation"], expected_corr)
    assert np.isclose(ns["average_energy"], expected_mean)
