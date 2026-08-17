# Kang et al. (2025) TJS baseline reproduction

This directory is independent from the existing short-term arm-IL experiments. It
reconstructs the player-level Tommy John Surgery (TJS) classification task described
in Kang et al., *Journal of Big Data* 12, 87 (2025), and cross-checks details against
the authors' public repository (`dxlabskku/TJS_Prediction`).

## What this baseline predicts

One sample is one pitcher, not one game window. The label is whether that pitcher is
in the TJS case group. Each sample contains 224 five-day bins from 100 through 1,215
days before the pitcher-specific reference date and 102 features (17 Statcast metrics
times six pitch types: CH, CU, FC, FF, SI, SL).

## Reproduction choices

- MLB regular-season Statcast, 2016--2023.
- TJS cases come from the public surgery list used by the paper.
- Left-handed horizontal variables are reflected to right-handed orientation.
- Missing values are interpolated within pitcher; wholly missing pitcher-features use
  the cohort mean, matching the public code.
- Values outside median +/- 4.7 standard deviations are treated as missing.
- Random stratified train/validation/test split of 60/20/20 for each seed.
- MinMaxScaler is fit on train only.
- Weighted BCE uses positive weight 5.
- Seeds are 100, 200, ..., 1000, matching the public code.

The surgery list is a live file and no longer represents the exact publication-time
snapshot. The builder therefore matches the published 101/519 class counts and saves
the selected pitcher IDs. With the locally available Statcast files, the exact Table 2
season-count composition cannot be recovered (the authors' `final_df.csv` is only a
placeholder in the public repository); the observed composition is recorded in
`data/cohort_manifest.json`.

The paper text says differencing uses the first-season mean, while the released code
uses the mean over the full selected period. `--delta-mode official_full_period` is the
default for numerical reproduction; `--delta-mode paper_first_season` implements the
written method. This discrepancy is preserved in the manifest instead of hidden.

## Run order

```powershell
.\.venv\Scripts\python.exe "베이스라인 파일/01_build_tjs_cohort.py"
.\.venv\Scripts\python.exe "베이스라인 파일/02_build_sequences.py" --delta-mode official_full_period
.\.venv\Scripts\python.exe "베이스라인 파일/03_train_models.py" --model logistic
```

GPU models:

```powershell
.\.venv\Scripts\python.exe "베이스라인 파일/03_train_models.py" --model lstm
.\.venv\Scripts\python.exe "베이스라인 파일/03_train_models.py" --model vit
```

The CPU logistic model is a pipeline/data sanity baseline. LSTM and ViT are the
paper-comparable models. CSV, Excel, plots, predictions, and checkpoints are written
under `results/`. Large intermediate data and checkpoints are excluded by the project
gitignore.

## Important interpretation

This reproduces a case-level random-split experiment. It must not be presented as a
prospective rolling 100-day alarm without a separate temporal/player-external
validation. After reproducing the published baseline, the recommended extension is a
second, stricter experiment with temporal splitting and train-only cohort statistics.
