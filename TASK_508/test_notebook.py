import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor
import numpy as np
import pandas as pd
import warnings


# Silence sklearn and runtime warnings for clean output
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@pytest.mark.parametrize("notebook", ["final_notebook.ipynb"])
def test_notebook_executes(notebook):
    """Ensure the notebook executes fully without errors."""
    with open(notebook) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=800, kernel_name="python3")
        assert ep.preprocess(nb) is not None, f"Failed executing {notebook}"


def get_namespace(path):
    """Execute notebook cells and collect namespace."""
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)
    ns = {}
    for cell in nb.cells:
        if cell.cell_type == "code":
            exec(cell.source, ns)
    return ns


def test_kmeans_variables_and_metrics():
    """Validate all required KMeans clustering variables and metrics."""
    ns = get_namespace("final_notebook.ipynb")

    required_vars = [
        "kmeans_model", "inertia_value",
        "silhouette_avg", "davies_bouldin", "cluster_summary"
    ]
    for var in required_vars:
        assert var in ns, f"{var} missing from notebook output"

    km = ns["kmeans_model"]
    assert hasattr(km, "inertia_"), "kmeans_model not fitted properly"
    assert km.n_clusters == 4, "KMeans must use n_clusters=4"
    assert getattr(km, "random_state", None) == 42, "random_state must be 42 for reproducibility"

    inertia = ns["inertia_value"]
    silhouette = ns["silhouette_avg"]
    db_index = ns["davies_bouldin"]

    assert isinstance(inertia, float) and inertia > 0, "inertia_value must be positive float"
    assert isinstance(silhouette, float), "silhouette_avg must be float"
    assert 0 <= silhouette <= 1, "silhouette_avg must be in [0, 1]"
    assert isinstance(db_index, float) and db_index >= 0, "davies_bouldin must be non-negative float"


def test_cluster_summary_structure_and_sorting():
    """Ensure cluster_summary DataFrame structure, shape, and sorting correctness."""
    ns = get_namespace("final_notebook.ipynb")
    df = ns["df"]
    summary = ns["cluster_summary"]
    km = ns["kmeans_model"]

    assert isinstance(summary, pd.DataFrame), "cluster_summary must be a DataFrame"
    assert not summary.empty, "cluster_summary should not be empty"

    for col in ["age", "income", "spending_score"]:
        assert col in summary.columns, f"{col} missing in cluster_summary"

    assert len(summary) == km.n_clusters, "cluster_summary must have one row per cluster"

    assert summary["spending_score"].is_monotonic_decreasing, \
        "cluster_summary must be sorted by spending_score descending"

    assert "cluster" in df.columns, "df must include cluster labels"
    assert df["cluster"].nunique() == km.n_clusters, "Cluster label count mismatch"


def test_cluster_value_ranges_and_means():
    """Check numerical consistency of cluster means and value ranges."""
    ns = get_namespace("final_notebook.ipynb")
    df = ns["df"]
    summary = ns["cluster_summary"]

    for col in ["age", "income", "spending_score"]:
        min_val, max_val = df[col].min(), df[col].max()
        assert summary[col].between(min_val, max_val).all(), \
            f"cluster_summary {col} values out of range [{min_val}, {max_val}]"

    sorted_scores = summary["spending_score"].values
    assert np.all(sorted_scores[:-1] >= sorted_scores[1:]), \
        "spending_score means should be sorted in descending order"


def test_cluster_summary_matches_groupby_mean():
    """Ensure cluster_summary correctly represents per-cluster means."""
    ns = get_namespace("final_notebook.ipynb")
    df = ns["df"]
    summary = ns["cluster_summary"]

    recomputed = (
        df.groupby("cluster")[["age", "income", "spending_score"]]
        .mean()
        .sort_values(by="spending_score", ascending=False)
    ).round(2)

    # Allow small numerical rounding difference
    pd.testing.assert_frame_equal(
        summary.round(2).reset_index(drop=True),
        recomputed.round(2).reset_index(drop=True),
        check_dtype=False,
        atol=1e-2
    )


def test_reproducibility_with_random_state():
    """Verify model reproducibility using same random_state."""
    ns = get_namespace("final_notebook.ipynb")
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    df = ns["df"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[["age", "income", "spending_score"]])

    # Re-run clustering with same config
    km_re = KMeans(n_clusters=4, random_state=42)
    km_re.fit(scaled)

    assert np.isclose(km_re.inertia_, ns["kmeans_model"].inertia_, rtol=1e-5), \
        "Recomputed inertia mismatch — model not reproducible"
