"""
3-evaluate_model.py

Evaluate trained ASL landmark model.

对应教程扩展：
A. 可视化训练过程：training_curve.png
B. 混淆矩阵分析：confusion_matrix.png、top_confusions.csv

输入：
runs/model_xxx_sample0.xx/
    csv_landmark_dataset/val.csv
    model_training/best_model.pth
    model_training/training_log.csv

输出：
runs/model_xxx_sample0.xx/model_evaluation/
    training_curve.png
    confusion_matrix.png
    classification_report.csv
    wrong_predictions.csv
    top_confusions.csv
    evaluation_metrics.json
    evaluation_report.md

默认：
- 使用 GPU：--device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


FEATURE_LENGTH = 2 * 21 * 3
NUM_CLASSES = 26
CLASS_NAMES = [chr(ord("A") + index) for index in range(NUM_CLASSES)]


class HandLandmarkDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, image_paths: list[str]):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.image_paths = image_paths

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index], self.image_paths[index]


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
    print("请选择要评估的 model run：")
    print("=" * 80)

    for index, folder in enumerate(candidates, start=1):
        model_path = folder / "model_training" / "best_model.pth"
        val_csv = folder / "csv_landmark_dataset" / "val.csv"
        status = []
        status.append("model: yes" if model_path.exists() else "model: no")
        status.append("val: yes" if val_csv.exists() else "val: no")
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
                "临时 CPU 调试可运行：python \"z_codes/test/3-evaluate_model.py\" --device cpu"
            )
        return torch.device("cuda")

    if device_text == "cpu":
        return torch.device("cpu")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def torch_load_any(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(model_path: Path, device):
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
        raise SystemExit("[错误] 模型格式不支持")

    model.to(device)
    model.eval()
    return model, class_names


def load_val_data(csv_path: Path):
    data = pd.read_csv(csv_path)

    if data.shape[1] < 2 + FEATURE_LENGTH:
        raise SystemExit(f"[错误] CSV 列数不足，当前 {data.shape[1]}")

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

    return features, labels, image_paths


def run_predictions(model, loader, device):
    y_true = []
    y_pred = []
    confidences = []
    image_paths = []

    with torch.no_grad():
        for features, labels, paths in tqdm(loader, desc="Evaluating"):
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(features)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predicted.cpu().numpy().tolist())
            confidences.extend(confidence.cpu().numpy().tolist())
            image_paths.extend(list(paths))

    return np.array(y_true), np.array(y_pred), np.array(confidences), image_paths


def save_classification_report(path: Path, y_true, y_pred, class_names):
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


def save_wrong_predictions(path: Path, y_true, y_pred, confidences, image_paths, class_names):
    rows = []

    for true_idx, pred_idx, conf, image_path in zip(y_true, y_pred, confidences, image_paths):
        if int(true_idx) != int(pred_idx):
            rows.append({
                "image_path": image_path,
                "true_index": int(true_idx),
                "true_label": class_names[int(true_idx)],
                "pred_index": int(pred_idx),
                "pred_label": class_names[int(pred_idx)],
                "confidence": float(conf),
            })

    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def save_top_confusions(path: Path, matrix: np.ndarray, class_names: list[str], top_k: int = 20):
    rows = []

    for true_idx in range(matrix.shape[0]):
        for pred_idx in range(matrix.shape[1]):
            if true_idx == pred_idx:
                continue

            count = int(matrix[true_idx, pred_idx])
            if count > 0:
                rows.append({
                    "true_label": class_names[true_idx],
                    "predicted_label": class_names[pred_idx],
                    "count": count,
                })

    rows = sorted(rows, key=lambda row: row["count"], reverse=True)[:top_k]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    return rows


def plot_training_curve(training_log_path: Path, output_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not training_log_path.exists():
        return False

    log = pd.read_csv(training_log_path)

    if log.empty:
        return False

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(log["epoch"], log["train_loss"], label="Train Loss")
    ax1.plot(log["epoch"], log["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(log["epoch"], log["val_accuracy_percent"], label="Val Accuracy (%)")
    ax2.set_ylabel("Validation Accuracy (%)")
    ax2.legend(loc="upper right")

    plt.title("Training Curve")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return True


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], output_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix)

    ax.set_title("ASL Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.colorbar(im, ax=ax)

    max_value = matrix.max() if matrix.size else 0
    threshold = max_value / 2 if max_value else 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
            if value:
                ax.text(j, i, str(value), ha="center", va="center")

    fig.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_evaluation_report(
    path: Path,
    model_run_dir: Path,
    accuracy: float,
    total: int,
    wrong_count: int,
    top_confusions: list[dict],
    device: torch.device,
):
    lines = [
        "# ASL Model Evaluation Report",
        "",
        "## Tutorial Extension Coverage",
        "",
        "- A. Training curve visualization: `training_curve.png`",
        "- B. Confusion matrix analysis: `confusion_matrix.png`",
        "- Classification report: `classification_report.csv`",
        "- Wrong predictions: `wrong_predictions.csv`",
        "- Top confusions: `top_confusions.csv`",
        "",
        "## Summary",
        "",
        f"- Model run: `{model_run_dir.name}`",
        f"- Device: `{device}`",
        f"- Evaluation samples: `{total}`",
        f"- Accuracy: `{accuracy * 100:.2f}%`",
        f"- Wrong predictions: `{wrong_count}`",
        "",
        "## Top Confusions",
        "",
    ]

    if top_confusions:
        lines.extend([
            "| True Label | Predicted Label | Count |",
            "|---|---|---:|",
        ])
        for row in top_confusions:
            lines.append(f"| {row['true_label']} | {row['predicted_label']} | {row['count']} |")
    else:
        lines.append("No off-diagonal confusions found.")

    lines.extend([
        "",
        "## Limitations and Discussion",
        "",
        "- The model recognizes ASL alphabet classes A-Z, not full natural sign language.",
        "- MediaPipe detection quality affects the generated CSV quality.",
        "- Lighting, camera angle, hand size, background, and skin tone may affect generalization.",
        "- Sampling ratio affects dataset coverage; compare sample0.05 and sample0.10 runs before drawing conclusions.",
        "- High validation accuracy does not guarantee robust real-world webcam performance.",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    root = find_project_root()

    parser = argparse.ArgumentParser(description="Evaluate ASL landmark model. GPU-first.")
    parser.add_argument("--runs_dir", type=Path, default=root / "runs")
    parser.add_argument("--model_run", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    model_run_dir = choose_model_run(args.runs_dir) if args.model_run is None else args.model_run.resolve()

    if not model_run_dir.exists():
        raise SystemExit(f"[错误] model run 不存在：{model_run_dir}")

    csv_path = model_run_dir / "csv_landmark_dataset" / "val.csv"
    model_path = model_run_dir / "model_training" / "best_model.pth"
    training_log_path = model_run_dir / "model_training" / "training_log.csv"
    eval_dir = model_run_dir / "model_evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise SystemExit(f"[错误] 找不到 val.csv：{csv_path}")

    if not model_path.exists():
        raise SystemExit(f"[错误] 找不到 best_model.pth：{model_path}")

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model, class_names = load_model(model_path, device)

    features, labels, image_paths = load_val_data(csv_path)
    dataset = HandLandmarkDataset(features, labels, image_paths)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    y_true, y_pred, confidences, paths = run_predictions(model, loader, device)

    accuracy = accuracy_score(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    wrong_count = int((y_true != y_pred).sum())

    training_curve_path = eval_dir / "training_curve.png"
    confusion_matrix_path = eval_dir / "confusion_matrix.png"
    classification_report_path = eval_dir / "classification_report.csv"
    wrong_predictions_path = eval_dir / "wrong_predictions.csv"
    top_confusions_path = eval_dir / "top_confusions.csv"
    metrics_path = eval_dir / "evaluation_metrics.json"
    report_path = eval_dir / "evaluation_report.md"

    try:
        plot_training_curve(training_log_path, training_curve_path)
    except ImportError as exc:
        raise SystemExit(
            "[错误] 需要 matplotlib 生成训练曲线。请运行：python -m pip install matplotlib"
        ) from exc

    plot_confusion_matrix(matrix, class_names, confusion_matrix_path)
    save_classification_report(classification_report_path, y_true, y_pred, class_names)
    save_wrong_predictions(wrong_predictions_path, y_true, y_pred, confidences, paths, class_names)
    top_confusions = save_top_confusions(top_confusions_path, matrix, class_names)

    metrics = {
        "model_run_name": model_run_dir.name,
        "device": str(device),
        "accuracy": float(accuracy),
        "accuracy_percent": float(accuracy * 100),
        "samples": int(len(y_true)),
        "wrong_predictions": wrong_count,
        "model_path": str(model_path),
        "val_csv": str(csv_path),
    }

    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    write_evaluation_report(
        path=report_path,
        model_run_dir=model_run_dir,
        accuracy=accuracy,
        total=len(y_true),
        wrong_count=wrong_count,
        top_confusions=top_confusions,
        device=device,
    )

    print("=" * 80)
    print("[完成] Model evaluation complete")
    print(f"Accuracy:              {accuracy * 100:.2f}%")
    print(f"Training curve:        {training_curve_path}")
    print(f"Confusion matrix:      {confusion_matrix_path}")
    print(f"Classification report: {classification_report_path}")
    print(f"Wrong predictions:     {wrong_predictions_path}")
    print(f"Top confusions:        {top_confusions_path}")
    print(f"Evaluation report:     {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
