import nbformat
import pytest
import numpy as np
import scipy.linalg as la
from nbconvert.preprocessors import ExecutePreprocessor


@pytest.mark.parametrize("notebook", ["final_notebook.ipynb"])
def test_notebook_exec(notebook):
    """Ensure notebook runs without errors."""
    with open(notebook) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        try:
            assert ep.preprocess(nb) is not None, f"Got empty notebook for {notebook}"
        except Exception as e:
            assert False, f"Failed executing {notebook}: {e}"


def get_notebook_namespace(path):
    """Executes notebook cells and returns variables."""
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)
    ns = {}
    for cell in nb.cells:
        if cell.cell_type == "code":
            exec(cell.source, ns)
    return ns


def test_hamiltonian_analysis():
    ns = get_notebook_namespace("final_notebook.ipynb")

    # Variable existence
    for v in ["result", "analyze_quantum_hamiltonian"]:
        assert v in ns, f"{v} missing"
    result = ns["result"]

    # Validate structure
    keys = {"H", "eigenvalues", "eigenvectors", "eigen_reconstruction_valid",
            "ground_state_energy", "trace_conserved", "matrix_norms",
            "is_positive_definite", "is_orthogonal", "stability_report"}
    assert set(result.keys()) == keys, "Incorrect return structure"

    H = result["H"]
    assert H.shape == (5, 5)
    assert np.allclose(H, H.T, atol=1e-12), "Matrix must be symmetric"

    eigenvalues = result["eigenvalues"]
    eigenvectors = result["eigenvectors"]
    assert np.all(np.diff(eigenvalues) >= 0), "Eigenvalues not ascending"
    assert np.allclose(eigenvectors.T @ eigenvectors, np.eye(5), atol=1e-10)

    assert result["eigen_reconstruction_valid"] is True
    assert result["trace_conserved"] is True
    assert bool(result["is_positive_definite"]) is True
    assert result["is_orthogonal"] is True

    # Matrix norms and stability
    mn = result["matrix_norms"]
    assert set(mn.keys()) == {"frobenius", "spectral", "1-norm", "inf-norm"}
    assert all(v > 0 for v in mn.values())

    sr = result["stability_report"]
    assert sr["well_conditioned"] and sr["is_symmetric"] and sr["full_rank"]
    print("All Hamiltonian checks passed.")
