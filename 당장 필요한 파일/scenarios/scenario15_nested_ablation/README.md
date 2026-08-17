# Scenario 15 nested ablation + Scenario 8 SHAP

`01_nested_ablation_shap.py` trains strictly nested XGBoost feature sets A--E.
For each pitcher role, Model C selects hyperparameters on validation PR-AUC once;
the same parameters are then reused for every stage. Class weights use train data,
the classification threshold uses validation data, and test remains evaluation-only.

Model E test data is used for SHAP role comparison. One-hot categorical levels are
aggregated back to their source variable, and importance is also summarized across
the six conceptual feature groups.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe "당장 필요한 파일/scenarios/scenario15_nested_ablation/01_nested_ablation_shap.py"
```

Outputs are saved under `results/` as CSV, Excel, model JSON, feature manifests,
and PNG charts.
