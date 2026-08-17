"""Create the three runnable experiment notebooks."""
from pathlib import Path
import nbformat as nbf

HERE=Path(__file__).resolve().parent

def make(name,title,description,cells):
    nb=nbf.v4.new_notebook()
    nb["cells"]=[nbf.v4.new_markdown_cell(f"# {title}\n\n{description}")]+[nbf.v4.new_code_cell(c) for c in cells]
    nb["metadata"]["kernelspec"]={"display_name":"Python 3","language":"python","name":"python3"}
    nbf.write(nb,HERE/name)

make("01_XGBoost_CPU.ipynb","Scenario 1 — Event-aware weighted XGBoost (CPU)",
     "Main interpretable baseline. Train-only imputation, event weighting, class balance, Platt calibration, and Validation-selected F2 threshold.",[
     "%run 03_train_xgboost_cpu.py",
     "import pandas as pd\ndisplay(pd.read_csv('results/xgboost_weighted_metrics.csv'))"])
make("02_LSTM_GPU.ipynb","Scenario 2 — 2025-paper LSTM extension (GPU)",
     "20 x 5-day sequence, event-aware weighted BCE, and auxiliary days-to-IL regression. Requires CUDA.",[
     "!pip install -r requirements-gpu.txt",
     "%run 04_train_gpu_models.py --model lstm --seed 42 --epochs 100",
     "%run 04_train_gpu_models.py --model lstm --seed 52 --epochs 100",
     "%run 04_train_gpu_models.py --model lstm --seed 62 --epochs 100",
     "%run 05_combine_all_results.py"])
make("03_ViT_GPU.ipynb","Scenario 3 — 2025-paper ViT extension (GPU)",
     "The 20 x 33 temporal matrix is resized to an image for pretrained ViT. Weighted classification plus days-to-IL regression. Requires CUDA and timm.",[
     "!pip install -r requirements-gpu.txt",
     "%run 04_train_gpu_models.py --model vit --seed 42 --epochs 80",
     "%run 04_train_gpu_models.py --model vit --seed 52 --epochs 80",
     "%run 04_train_gpu_models.py --model vit --seed 62 --epochs 80",
     "%run 05_combine_all_results.py"])
make("04_ResNet_GPU.ipynb","Scenario 4 — Residual TCN improvement (GPU)",
     "Compact 1D ResNet with dilated temporal convolutions, squeeze-and-excitation, personal deltas, prior injury history, label smoothing, and Validation Platt calibration.",[
     "!pip install -r requirements-gpu.txt",
     "%run 04_train_gpu_models.py --model resnet --seed 42 --epochs 120",
     "%run 04_train_gpu_models.py --model resnet --seed 52 --epochs 120",
     "%run 04_train_gpu_models.py --model resnet --seed 62 --epochs 120",
     "%run 05_combine_all_results.py"])
make("05_GBM_Engineered.ipynb","Scenario 5 — Domain-engineered workload features + LightGBM (CPU)",
     "Acute:chronic workload ratios, rest-day patterns, velocity/spin/extension trend deltas, and season/career cumulative load replace the flattened 20x33 raw sequence. Train-only event weighting, Validation Platt calibration, and Validation-selected F1 threshold.",[
     "%run 05_build_engineered_features.py",
     "%run 05_train_gbm.py --seed 42",
     "%run 05_train_gbm.py --seed 52",
     "%run 05_train_gbm.py --seed 62",
     "%run 05_combine_all_results.py"])
print("Created XGBoost, LSTM, ViT, ResNet, and GBM notebooks")
