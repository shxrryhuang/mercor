import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor
import pandas as pd
import json
import os

@pytest.mark.parametrize("notebook", ["final_notebook.ipynb"])
def test_notebook_exec(notebook):
    with open(notebook) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    assert ep.preprocess(nb) is not None

def get_ns(path):
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)
    ns = {}
    for cell in nb.cells:
        if cell.cell_type == "code":
            exec(cell.source, ns)
    return ns

def test_returns_and_files():
    ns = get_ns("final_notebook.ipynb")
    assert "validate_and_export_inventory" in ns and callable(ns["validate_and_export_inventory"])
    assert "result" in ns
    result = ns["result"]

    # structure
    assert set(result.keys()) == {"validation_passed","validation_issues","exports_generated","export_data"}
    assert isinstance(result["validation_passed"], bool)
    assert isinstance(result["validation_issues"], list)
    for issue in result["validation_issues"]:
        assert set(issue.keys()) == {"check","severity","message","affected_rows"}
        assert issue["severity"] in {"critical","warning"}
        assert isinstance(issue["affected_rows"], list)

    # files
    files = result["exports_generated"]
    assert set(files) == {"inventory_summary.json","inventory_detailed.csv","inventory_report.txt"}
    for f in files:
        assert os.path.exists(f)

    # json summary
    js = result["export_data"]["json_summary"]
    assert "metadata" in js and "category_snapshot" in js and "at_risk_items" in js
    md = js["metadata"]
    assert "export_date" in md and "total_skus" in md and "total_inventory_value_usd" in md
    assert isinstance(md["total_skus"], int) and md["total_skus"] == 8

    # csv data
    csv_data = result["export_data"]["csv_data"]
    assert isinstance(csv_data, list) and len(csv_data) == 8
    assert "restock_tier" in csv_data[0]

    # txt
    txt = result["export_data"]["txt_report"]
    for cat in ["Electronics","Home","Beauty","Grocery"]:
        assert cat in txt

    # validation expectations: should pass (no criticals in provided data)
    assert result["validation_passed"] is True
    checks = {i["check"] for i in result["validation_issues"]}
    assert "low_stock" in checks or "high_return_rate" in checks or "suspicious_pricing" in checks
