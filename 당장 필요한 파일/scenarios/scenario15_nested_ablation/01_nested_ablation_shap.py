"""Scenario 15 nested ablation and Scenario 8 starter/reliever SHAP comparison.

The feature sets are strictly nested. Hyperparameters are selected once per role
with Model C and then held fixed for Models A--E, so feature additions are not
confounded with a separate tuning search.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight


SEED = 42
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
WORKSPACE = HERE.parents[2]
OUT = HERE / "results"
MODEL_DIR = OUT / "models"
FIGURE_DIR = OUT / "figures"
RUNNER_DIR = PROJECT / "scenarios" / "scenario5_runner_gap" / "data"
EPISODES_PATH = WORKSPACE / "필요없는거" / "mlb_injury_project" / "data" / "interim" / "injury_episodes" / "injury_episodes.parquet"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


C = load_module("scenario1_common", PROJECT / "scenarios" / "scenario1_case_control" / "02_xgboost.py")
HISTORY_MODULE = load_module("scenario13_history", PROJECT / "scenarios" / "scenario13_injury_history" / "01_history_xgboost.py")

DEMOGRAPHICS = ["p_throws", "age", "height_inches", "weight_lb", "birth_country"]
WORKLOAD = [
    "days_since_prev_game", "n_pitches_window", "n_batters_faced_window",
    "n_appearances_window", "innings_pitched_window", "complete_games_window",
]
HISTORY = list(HISTORY_MODULE.H)
RUNNER_GAP = [
    "runner_gap_velocity", "runner_gap_spin", "runner_gap_extension",
    "runner_gap_pfx_x", "runner_gap_pfx_z",
]
META = [
    "player_id", "window_end_date", "il_start_date", "injury_class_strict",
    "days_to_injury", "split", "label",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", nargs="+", choices=["bullpen", "starter"], default=["bullpen", "starter"])
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--shap-sample", type=int, default=5000)
    return parser.parse_args()


def add_train_delta(splits: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    absolute = [c for c in splits["train"].columns if c.startswith("w_")]
    player_means = splits["train"].groupby("player_id", dropna=False)[absolute].mean()
    league_means = splits["train"][absolute].mean()
    result: dict[str, pd.DataFrame] = {}
    for split, frame in splits.items():
        d = frame.copy()
        baseline = d[["player_id"]].join(player_means, on="player_id")[absolute]
        baseline = baseline.fillna(league_means)
        for col in absolute:
            d[f"delta_{col}"] = d[col].to_numpy() - baseline[col].to_numpy()
        result[split] = d
    return result, [f"delta_{c}" for c in absolute]


def load_enriched_splits(role: str, episodes: pd.DataFrame):
    path = RUNNER_DIR / f"{role}_runner_gap.parquet"
    frame = pd.read_parquet(path)
    frame = frame.loc[frame["label"] != 3].copy()
    frame["label"] = (frame["label"] > 0).astype("int8")
    for col in ("window_end_date", "il_start_date"):
        frame[col] = pd.to_datetime(frame[col], errors="coerce")
    splits = {s: frame.loc[frame["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    splits = {s: HISTORY_MODULE.add_history(d, episodes) for s, d in splits.items()}
    return add_train_delta(splits)


def feature_sets(splits: dict[str, pd.DataFrame], delta_cols: list[str]):
    absolute = [c for c in splits["train"].columns if c.startswith("w_")]
    sets = {
        "A_demographics_workload": DEMOGRAPHICS + WORKLOAD,
        "B_plus_injury_history": DEMOGRAPHICS + WORKLOAD + HISTORY,
        "C_plus_absolute_statcast": DEMOGRAPHICS + WORKLOAD + HISTORY + absolute,
        "D_plus_personal_delta": DEMOGRAPHICS + WORKLOAD + HISTORY + absolute + delta_cols,
        "E_plus_runner_gap": DEMOGRAPHICS + WORKLOAD + HISTORY + absolute + delta_cols + RUNNER_GAP,
    }
    missing = {name: [c for c in cols if c not in splits["train"]] for name, cols in sets.items()}
    missing = {name: cols for name, cols in missing.items() if cols}
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    return sets


def subset_splits(splits: dict[str, pd.DataFrame], features: list[str]):
    keep = META + features
    return {s: d[keep].copy() for s, d in splits.items()}


def feature_group(feature: str) -> str:
    if feature.startswith("runner_gap_"):
        return "runner-context gap"
    if feature.startswith("delta_"):
        return "personal delta"
    if feature.startswith("w_"):
        return "absolute Statcast"
    if feature in HISTORY:
        return "injury history"
    if feature in WORKLOAD:
        return "workload"
    return "demographics"


def original_feature(encoded_name: str) -> str:
    if encoded_name.startswith("p_throws_"):
        return "p_throws"
    if encoded_name.startswith("birth_country_"):
        return "birth_country"
    return encoded_name


def shap_table(model, x_test: pd.DataFrame, role: str, sample_size: int) -> pd.DataFrame:
    sample = x_test.sample(min(sample_size, len(x_test)), random_state=SEED)
    values = shap.TreeExplainer(model).shap_values(sample, check_additivity=False)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    encoded = pd.DataFrame({
        "encoded_feature": sample.columns,
        "feature": [original_feature(c) for c in sample.columns],
        "mean_abs_shap": np.abs(values).mean(axis=0),
        "mean_shap": values.mean(axis=0),
    })
    # Combine one-hot levels into their source feature for a stable role comparison.
    grouped = encoded.groupby("feature", as_index=False).agg(
        mean_abs_shap=("mean_abs_shap", "sum"), mean_shap=("mean_shap", "sum")
    )
    grouped["group"] = grouped["feature"].map(feature_group)
    grouped = grouped.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    grouped.insert(0, "role", role)
    grouped.insert(2, "rank", np.arange(1, len(grouped) + 1))
    grouped["shap_share"] = grouped["mean_abs_shap"] / grouped["mean_abs_shap"].sum()
    return grouped


def save_shap_plots(shap_tables: dict[str, pd.DataFrame], group_df: pd.DataFrame) -> None:
    for role, table in shap_tables.items():
        top = table.head(15).sort_values("mean_abs_shap")
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh(top["feature"], top["mean_abs_shap"], color="#377eb8" if role == "starter" else "#e41a1c")
        ax.set(xlabel="Mean |SHAP value|", title=f"Model E top features: {role}")
        ax.grid(axis="x", alpha=.25)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"scenario8_shap_top15_{role}.png", dpi=180)
        plt.close(fig)
    pivot = group_df.pivot(index="group", columns="role", values="shap_share").fillna(0)
    ax = pivot.plot.barh(figsize=(9, 5), color=["#e41a1c", "#377eb8"])
    ax.set(xlabel="Share of total mean |SHAP|", title="SHAP importance by feature group")
    ax.grid(axis="x", alpha=.25)
    ax.figure.tight_layout()
    ax.figure.savefig(FIGURE_DIR / "scenario8_shap_group_role_comparison.png", dpi=180)
    plt.close(ax.figure)


def run_role(role: str, episodes: pd.DataFrame, trials: int, n_jobs: int, shap_sample: int):
    print(f"\n[{role}] enriching data")
    splits, delta_cols = load_enriched_splits(role, episodes)
    sets = feature_sets(splits, delta_cols)

    c_splits = subset_splits(splits, sets["C_plus_absolute_statcast"])
    cx, cy, _ = C.preprocess(c_splits)
    weights = compute_sample_weight("balanced", cy["train"])
    print("  tuning once on Model C")
    params, tuning = C.tune(cx["train"], cy["train"], cx["val"], cy["val"], weights, trials, n_jobs)
    tuning.insert(0, "role", role)

    metric_rows, models = [], {}
    for order, (stage, features) in enumerate(sets.items(), start=1):
        current = subset_splits(splits, features)
        x, y, encoded_features = C.preprocess(current)
        train_weight = compute_sample_weight("balanced", y["train"])
        model = xgb.XGBClassifier(**C.base_params(n_jobs, **params))
        model.fit(x["train"], y["train"], sample_weight=train_weight, eval_set=[(x["val"], y["val"])], verbose=False)
        val_prob = model.predict_proba(x["val"])[:, 1]
        threshold = C.choose_threshold(y["val"], val_prob)
        for split, prob in (("val", val_prob), ("test", model.predict_proba(x["test"])[:, 1])):
            for threshold_type, cutoff in (("0.5", .5), ("validation_selected", threshold)):
                metric_rows.append({
                    "role": role, "stage_order": order, "stage": stage, "split": split,
                    "threshold_type": threshold_type, "threshold": cutoff,
                    "n_features_raw": len(features), "n_features_encoded": len(encoded_features),
                    "best_iteration": model.best_iteration, **C.metrics(y[split], prob, cutoff),
                })
        model.save_model(MODEL_DIR / f"{role}_{stage}.json")
        (MODEL_DIR / f"{role}_{stage}_features.json").write_text(
            json.dumps(encoded_features, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        models[stage] = (model, x)
        print(f"  {stage}: test PR-AUC={metric_rows[-1]['pr_auc']:.5f}, encoded features={len(encoded_features)}")

    e_model, e_x = models["E_plus_runner_gap"]
    return pd.DataFrame(metric_rows), tuning, shap_table(e_model, e_x["test"], role, shap_sample), sets, params


def main() -> None:
    args = parse_args()
    for directory in (OUT, MODEL_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    episodes = pd.read_parquet(EPISODES_PATH)
    episodes["il_start_date"] = pd.to_datetime(episodes["il_start_date"], errors="coerce")
    episodes["il_end_date"] = pd.to_datetime(episodes["il_end_date"], errors="coerce")

    metrics, tunings, shap_tables, manifests, selected = [], [], {}, [], []
    for role in args.roles:
        m, t, s, sets, params = run_role(role, episodes, args.trials, args.n_jobs, args.shap_sample)
        metrics.append(m); tunings.append(t); shap_tables[role] = s
        selected.append({"role": role, **params})
        for stage, features in sets.items():
            manifests.extend({"stage": stage, "feature_order": i, "feature": f} for i, f in enumerate(features, 1))

    metrics_df = pd.concat(metrics, ignore_index=True)
    tuning_df = pd.concat(tunings, ignore_index=True)
    shap_df = pd.concat(shap_tables.values(), ignore_index=True)
    group_df = shap_df.groupby(["role", "group"], as_index=False)["mean_abs_shap"].sum()
    group_df["shap_share"] = group_df["mean_abs_shap"] / group_df.groupby("role")["mean_abs_shap"].transform("sum")
    comparison = shap_tables.get("starter", pd.DataFrame()).merge(
        shap_tables.get("bullpen", pd.DataFrame()), on="feature", how="outer", suffixes=("_starter", "_reliever")
    ) if {"starter", "bullpen"}.issubset(shap_tables) else pd.DataFrame()
    manifest_df = pd.DataFrame(manifests).drop_duplicates()
    params_df = pd.DataFrame(selected)

    metrics_df.to_csv(OUT / "scenario15_nested_ablation_metrics.csv", index=False)
    shap_df.to_csv(OUT / "scenario8_shap_feature_importance.csv", index=False)
    group_df.to_csv(OUT / "scenario8_shap_group_importance.csv", index=False)
    with pd.ExcelWriter(OUT / "scenario15_scenario8_results.xlsx") as writer:
        metrics_df.to_excel(writer, sheet_name="01_nested_metrics", index=False)
        tuning_df.to_excel(writer, sheet_name="02_model_c_tuning", index=False)
        params_df.to_excel(writer, sheet_name="03_selected_params", index=False)
        manifest_df.to_excel(writer, sheet_name="04_feature_manifest", index=False)
        shap_tables.get("starter", pd.DataFrame()).to_excel(writer, sheet_name="05_shap_starter", index=False)
        shap_tables.get("bullpen", pd.DataFrame()).to_excel(writer, sheet_name="06_shap_reliever", index=False)
        group_df.to_excel(writer, sheet_name="07_group_shap", index=False)
        comparison.to_excel(writer, sheet_name="08_role_comparison", index=False)
    save_shap_plots(shap_tables, group_df)
    print("\nTest PR-AUC by nested stage")
    print(metrics_df.query("split == 'test' and threshold_type == 'validation_selected'")[["role", "stage", "pr_auc", "roc_auc", "f1", "recall", "precision"]].to_string(index=False))


if __name__ == "__main__":
    main()
