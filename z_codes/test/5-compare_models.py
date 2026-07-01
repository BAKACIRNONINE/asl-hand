"""
5-compare_models.py

Compare PyTorch MLP and scikit-learn RandomForest on the same ASL MediaPipe CSV dataset.

对应 MD 扩展 C：
- 使用同一份 CSV
- 比较 PyTorch MLP 和 RandomForestClassifier
- 输出准确率、训练时间、推断时间、模型大小、对比报告

输入：
runs/model_xxx_sample0.xx/
    csv_landmark_dataset/
        train.csv
        val.csv
    model_training/
        best_model.pth

输出：
runs/model_xxx_sample0.xx/model_evaluation/model_comparison/
    comparison_table.csv
    comparison_report.md
    mlp_metrics.json
    random_forest_metrics.json
    random_forest_model.joblib
    random_forest_classification_report.csv
    random_forest_confusion_matrix.csv
    random_forest_wrong_predictions.csv

默认：
- MLP 评估使用 GPU：--device cuda
- RandomForest 使用 CPU：n_jobs=-1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


FEATURE_LENGTH = 2 * 21 * 3
NUM_CLASSES = 26
CLASS_NAMES = [chr(ord("A") + index) for index in range(NUM_CLASSES)]


class HandLandmarkDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]


class HandLandmarkClassifier(nn.Module):
    def __init__(self, input_size: int = FEATURE_LENGTH, num_classes: int = NUM_CLASSES):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        return self.fc3(x)


def find_project_root() -> Path:
    current = Path(__file__).resolve()

    for parent in current.parents:
        if (parent / "runs").exists() and (parent / "z_codes").exists():
            return parent

    return current.parents[2]


def choose_model_run(runs_dir: Path) -> Path:
    candidates = sorted([
        path for path in runs_dir.iterdir()
        if path.is_dir() and path.name.startswith("model_")
    ])

    if not candidates:
        raise SystemExit(f"[错误] runs 里没有 model_xxx 文件夹：{runs_dir}")

    print("=" * 80)
    print("请选择要进行模型对比的 model run：")
    print("=" * 80)

    for index, folder in enumerate(candidates, start=1):
        train_csv = folder / "csv_landmark_dataset" / "train.csv"
        val_csv = folder / "csv_landmark_dataset" / "val.csv"
        mlp_model = folder / "model_training" / "best_model.pth"
        comparison_dir = folder / "model_evaluation" / "model_comparison"

        status = []
        status.append("train: yes" if train_csv.exists() else "train: no")
        status.append("val: yes" if val_csv.exists() else "val: no")
        status.append("mlp: yes" if mlp_model.exists() else "mlp: no")
        status.append("compared: yes" if comparison_dir.exists() else "compared: no")

        print(f"[{index}] {folder.name:30s} | {' | '.join(status)}")

    print("[0] 退出")
    print("=" * 80)

    while True:
        choice = input("请输入编号：").strip()

        if choice == "0":
            raise SystemExit("已退出")

        if choice.isdigit():
            number = int(choice)
            if 1 <= number <= len(candidates):
                return candidates[number - 1]

        print("[错误] 请输入有效编号")


def resolve_device(device_text: str) -> torch.device:
    if device_text == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "[错误] 默认要求使用 GPU/CUDA，但当前 PyTorch 检测不到 CUDA。\n"
                "临时 CPU 调试可运行：python \"z_codes/test/5-compare_models.py\" --device cpu"
            )
        return torch.device("cuda")

    if device_text == "cpu":
        return torch.device("cpu")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not csv_path.exists():
        raise SystemExit(f"[错误] 找不到 CSV：{csv_path}")

    data = pd.read_csv(csv_path)

    if data.shape[1] < 2 + FEATURE_LENGTH:
        raise SystemExit(
            f"[错误] CSV 列数不对，至少需要 {2 + FEATURE_LENGTH} 列，当前 {data.shape[1]} 列"
        )

    image_paths = data.iloc[:, 0].astype(str).tolist()
    features = data.iloc[:, 2:2 + FEATURE_LENGTH].to_numpy(dtype=np.float32)

    if features.shape[1] != FEATURE_LENGTH:
        raise SystemExit(f"[错误] Expected 126 features, got {features.shape[1]}")

    if "class_index" in data.columns:
        labels = data["class_index"].to_numpy(dtype=np.int64)
    elif "Class Index" in data.columns:
        labels = data["Class Index"].to_numpy(dtype=np.int64)
    else:
        labels = data.iloc[:, 1].to_numpy(dtype=np.int64)

    if labels.min() < 0 or labels.max() >= NUM_CLASSES:
        raise SystemExit(f"[错误] class_index 应该在 0-25，当前范围：{labels.min()}-{labels.max()}")

    return features, labels, image_paths


def torch_load_any(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_mlp_model(model_path: Path, device: torch.device):
    if not model_path.exists():
        raise SystemExit(f"[错误] 找不到 MLP 模型：{model_path}")

    loaded = torch_load_any(model_path, device)

    if isinstance(loaded, nn.Module):
        model = loaded
        class_names = CLASS_NAMES
    elif isinstance(loaded, dict) and "model_state_dict" in loaded:
        input_size = int(loaded.get("input_size", FEATURE_LENGTH))
        num_classes = int(loaded.get("num_classes", NUM_CLASSES))
        class_names = loaded.get("class_names", CLASS_NAMES)
        model = HandLandmarkClassifier(input_size=input_size, num_classes=num_classes)
        model.load_state_dict(loaded["model_state_dict"])
    elif isinstance(loaded, dict):
        model = HandLandmarkClassifier()
        model.load_state_dict(loaded)
        class_names = CLASS_NAMES
    else:
        raise SystemExit("[错误] MLP 模型格式不支持")

    model.to(device)
    model.eval()
    return model, class_names


def evaluate_mlp(
    model,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict:
    dataset = HandLandmarkDataset(x_val, y_val)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    y_pred = []

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        for features, _labels in tqdm(loader, desc="Evaluating MLP"):
            features = features.to(device, non_blocking=True)
            logits = model(features)
            predicted = logits.argmax(dim=1)
            y_pred.extend(predicted.cpu().numpy().tolist())

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    y_pred = np.array(y_pred, dtype=np.int64)
    accuracy = accuracy_score(y_val, y_pred)

    return {
        "model_name": "PyTorch MLP",
        "accuracy": float(accuracy),
        "accuracy_percent": float(accuracy * 100),
        "inference_time_seconds": float(elapsed),
        "samples": int(len(y_val)),
        "samples_per_second": float(len(y_val) / elapsed) if elapsed > 0 else None,
        "predictions": y_pred,
    }


def train_and_evaluate_random_forest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    args,
) -> dict:
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        class_weight=args.class_weight,
    )

    print("=" * 80)
    print("Training RandomForestClassifier")
    print(f"n_estimators: {args.n_estimators}")
    print(f"max_depth: {args.max_depth}")
    print(f"min_samples_leaf: {args.min_samples_leaf}")
    print(f"n_jobs: {args.n_jobs}")
    print("=" * 80)

    train_start = time.perf_counter()
    rf.fit(x_train, y_train)
    train_elapsed = time.perf_counter() - train_start

    infer_start = time.perf_counter()
    y_pred = rf.predict(x_val)
    infer_elapsed = time.perf_counter() - infer_start

    accuracy = accuracy_score(y_val, y_pred)

    return {
        "model_name": "RandomForestClassifier",
        "model": rf,
        "accuracy": float(accuracy),
        "accuracy_percent": float(accuracy * 100),
        "training_time_seconds": float(train_elapsed),
        "inference_time_seconds": float(infer_elapsed),
        "samples": int(len(y_val)),
        "samples_per_second": float(len(y_val) / infer_elapsed) if infer_elapsed > 0 else None,
        "predictions": y_pred.astype(np.int64),
        "params": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "random_state": args.seed,
            "n_jobs": args.n_jobs,
            "class_weight": args.class_weight,
        },
    }


def save_json(path: Path, data: dict) -> None:
    clean_data = {
        key: value
        for key, value in data.items()
        if key not in {"model", "predictions"}
    }

    path.write_text(json.dumps(clean_data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_classification_report(path: Path, y_true, y_pred, class_names) -> None:
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        labels=list(range(len(class_names))),
        output_dict=True,
        zero_division=0,
    )

    rows = []
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):
            rows.append({
                "label": label,
                "precision": metrics.get("precision", ""),
                "recall": metrics.get("recall", ""),
                "f1_score": metrics.get("f1-score", ""),
                "support": metrics.get("support", ""),
            })
        else:
            rows.append({
                "label": label,
                "precision": "",
                "recall": "",
                "f1_score": metrics,
                "support": "",
            })

    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def save_confusion_matrix_csv(path: Path, y_true, y_pred, class_names) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    matrix_df = pd.DataFrame(matrix, index=class_names, columns=class_names)
    matrix_df.index.name = "true_label"
    matrix_df.to_csv(path, encoding="utf-8")


def save_wrong_predictions(
    path: Path,
    y_true,
    y_pred,
    image_paths: list[str],
    class_names: list[str],
) -> None:
    rows = []

    for true_idx, pred_idx, image_path in zip(y_true, y_pred, image_paths):
        if int(true_idx) != int(pred_idx):
            rows.append({
                "image_path": image_path,
                "true_index": int(true_idx),
                "true_label": class_names[int(true_idx)],
                "pred_index": int(pred_idx),
                "pred_label": class_names[int(pred_idx)],
            })

    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def file_size_mb(path: Path) -> float | None:
    if not path.exists():
        return None
    return path.stat().st_size / (1024 * 1024)


def save_comparison_table(path: Path, mlp_metrics: dict, rf_metrics: dict, mlp_model_path: Path, rf_model_path: Path) -> None:
    rows = [
        {
            "model": "PyTorch MLP",
            "accuracy_percent": mlp_metrics["accuracy_percent"],
            "training_time_seconds": "already trained in 2-train_model.py",
            "inference_time_seconds": mlp_metrics["inference_time_seconds"],
            "samples_per_second": mlp_metrics["samples_per_second"],
            "model_size_mb": file_size_mb(mlp_model_path),
            "device": "GPU/CPU depends on --device",
        },
        {
            "model": "RandomForestClassifier",
            "accuracy_percent": rf_metrics["accuracy_percent"],
            "training_time_seconds": rf_metrics["training_time_seconds"],
            "inference_time_seconds": rf_metrics["inference_time_seconds"],
            "samples_per_second": rf_metrics["samples_per_second"],
            "model_size_mb": file_size_mb(rf_model_path),
            "device": "CPU",
        },
    ]

    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def write_comparison_report(
    path: Path,
    model_run_dir: Path,
    mlp_metrics: dict,
    rf_metrics: dict,
    mlp_model_path: Path,
    rf_model_path: Path,
) -> None:
    mlp_acc = mlp_metrics["accuracy_percent"]
    rf_acc = rf_metrics["accuracy_percent"]
    diff = mlp_acc - rf_acc

    if diff > 0:
        winner = "PyTorch MLP"
        reason = f"MLP is higher by {diff:.2f} percentage points."
    elif diff < 0:
        winner = "RandomForestClassifier"
        reason = f"RandomForest is higher by {-diff:.2f} percentage points."
    else:
        winner = "Tie"
        reason = "Both models have the same validation accuracy."

    lines = [
        "# ASL Model Comparison Report",
        "",
        "## Purpose",
        "",
        "This report corresponds to tutorial extension C: compare different models using the same MediaPipe landmark CSV dataset.",
        "",
        "Both models use the same 126 landmark features:",
        "",
        "```text",
        "2 hands × 21 landmarks × 3 coordinates = 126 features",
        "```",
        "",
        "## Compared Models",
        "",
        "| Model | Type | Library | Training Device | Notes |",
        "|---|---|---|---|---|",
        "| PyTorch MLP | Neural network | PyTorch | GPU by default | Already trained by `2-train_model.py` |",
        "| RandomForestClassifier | Traditional ML | scikit-learn | CPU | Trained inside `5-compare_models.py` |",
        "",
        "## Results",
        "",
        "| Metric | PyTorch MLP | RandomForestClassifier |",
        "|---|---:|---:|",
        f"| Validation accuracy | {mlp_acc:.2f}% | {rf_acc:.2f}% |",
        f"| Inference time | {mlp_metrics['inference_time_seconds']:.4f}s | {rf_metrics['inference_time_seconds']:.4f}s |",
        f"| Samples / second | {mlp_metrics['samples_per_second']:.2f} | {rf_metrics['samples_per_second']:.2f} |",
        f"| Model size | {file_size_mb(mlp_model_path):.2f} MB | {file_size_mb(rf_model_path):.2f} MB |",
        "",
        f"**Winner by validation accuracy:** `{winner}`",
        "",
        reason,
        "",
        "## Interpretation",
        "",
        "- If RandomForest performs close to MLP, it means MediaPipe landmarks already provide strong structured features.",
        "- If MLP performs better, the neural network is learning smoother nonlinear relationships between landmarks.",
        "- If RandomForest performs better, this task may not require a neural network for this dataset size.",
        "- RandomForest is a useful baseline, but the MLP integrates more naturally with the PyTorch realtime inference pipeline.",
        "",
        "## Saved Files",
        "",
        "- `comparison_table.csv`",
        "- `mlp_metrics.json`",
        "- `random_forest_metrics.json`",
        "- `random_forest_model.joblib`",
        "- `random_forest_classification_report.csv`",
        "- `random_forest_confusion_matrix.csv`",
        "- `random_forest_wrong_predictions.csv`",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    root = find_project_root()

    parser = argparse.ArgumentParser(description="Compare PyTorch MLP and RandomForestClassifier on ASL landmark CSV.")
    parser.add_argument("--runs_dir", type=Path, default=root / "runs")
    parser.add_argument("--model_run", type=Path, default=None)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--batch_size", type=int, default=512)

    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--class_weight", type=str, default=None, choices=[None, "balanced", "balanced_subsample"])
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    model_run_dir = choose_model_run(args.runs_dir) if args.model_run is None else args.model_run.resolve()

    if not model_run_dir.exists():
        raise SystemExit(f"[错误] model run 不存在：{model_run_dir}")

    csv_dir = model_run_dir / "csv_landmark_dataset"
    train_csv_path = csv_dir / "train.csv"
    val_csv_path = csv_dir / "val.csv"
    mlp_model_path = model_run_dir / "model_training" / "best_model.pth"

    comparison_dir = model_run_dir / "model_evaluation" / "model_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ASL model comparison")
    print(f"Model run: {model_run_dir}")
    print(f"Train CSV: {train_csv_path}")
    print(f"Val CSV:   {val_csv_path}")
    print(f"MLP model: {mlp_model_path}")
    print("=" * 80)

    x_train, y_train, _train_paths = load_csv(train_csv_path)
    x_val, y_val, val_paths = load_csv(val_csv_path)

    device = resolve_device(args.device)
    print(f"Using device for MLP evaluation: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    mlp_model, class_names = load_mlp_model(mlp_model_path, device)
    mlp_metrics = evaluate_mlp(
        model=mlp_model,
        x_val=x_val,
        y_val=y_val,
        device=device,
        batch_size=args.batch_size,
    )

    rf_metrics = train_and_evaluate_random_forest(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        args=args,
    )

    rf_model_path = comparison_dir / "random_forest_model.joblib"
    joblib.dump(rf_metrics["model"], rf_model_path)

    save_json(comparison_dir / "mlp_metrics.json", mlp_metrics)
    save_json(comparison_dir / "random_forest_metrics.json", rf_metrics)

    save_classification_report(
        comparison_dir / "random_forest_classification_report.csv",
        y_val,
        rf_metrics["predictions"],
        class_names,
    )
    save_confusion_matrix_csv(
        comparison_dir / "random_forest_confusion_matrix.csv",
        y_val,
        rf_metrics["predictions"],
        class_names,
    )
    save_wrong_predictions(
        comparison_dir / "random_forest_wrong_predictions.csv",
        y_val,
        rf_metrics["predictions"],
        val_paths,
        class_names,
    )

    save_comparison_table(
        comparison_dir / "comparison_table.csv",
        mlp_metrics,
        rf_metrics,
        mlp_model_path,
        rf_model_path,
    )

    write_comparison_report(
        comparison_dir / "comparison_report.md",
        model_run_dir,
        mlp_metrics,
        rf_metrics,
        mlp_model_path,
        rf_model_path,
    )

    print("=" * 80)
    print("[完成] Model comparison complete")
    print(f"MLP accuracy:          {mlp_metrics['accuracy_percent']:.2f}%")
    print(f"RandomForest accuracy: {rf_metrics['accuracy_percent']:.2f}%")
    print(f"Output folder:         {comparison_dir}")
    print(f"Comparison report:     {comparison_dir / 'comparison_report.md'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
