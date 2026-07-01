"""
2-train_model.py

GPU-first ASL landmark classifier training.

对应教程：
4.2 检查 CSV 特征列 126
4.3 确保 model.eval() 只在评估阶段使用
4.4 保存 best_model.pth，不用 last_model.pth 做默认推断
5.5 早停、最佳模型保存、ReduceLROnPlateau、梯度累积

默认：
- 使用 GPU：--device cuda
- 如果没有 CUDA，直接报错提醒，不自动切 CPU
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


FEATURE_LENGTH = 2 * 21 * 3
NUM_CLASSES = 26
CLASS_NAMES = [chr(ord("A") + index) for index in range(NUM_CLASSES)]


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
    print("请选择要训练的 model run：")
    print("=" * 80)

    for index, folder in enumerate(candidates, start=1):
        dataset_csv = folder / "csv_landmark_dataset" / "dataset_info.csv"
        train_csv = folder / "csv_landmark_dataset" / "train.csv"
        best_model = folder / "model_training" / "best_model.pth"
        dataset_status = "csv: yes" if dataset_csv.exists() or train_csv.exists() else "csv: no"
        model_status = "model: yes" if best_model.exists() else "model: no"
        print(f"[{index}] {folder.name:30s} | {dataset_status:8s} | {model_status}")

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


class HandLandmarkDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
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


def load_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not csv_path.exists():
        raise SystemExit(f"[错误] 找不到 CSV：{csv_path}")

    data = pd.read_csv(csv_path)

    if data.shape[1] < 2 + FEATURE_LENGTH:
        raise SystemExit(
            f"[错误] CSV 列数不对，至少需要 {2 + FEATURE_LENGTH} 列，当前 {data.shape[1]} 列"
        )

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

    return features, labels


def resolve_device(device_text: str) -> torch.device:
    if device_text == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "[错误] 默认要求使用 GPU/CUDA，但当前 PyTorch 检测不到 CUDA。\n"
                "请检查：\n"
                "1. 是否安装 NVIDIA 驱动\n"
                "2. 当前环境里的 torch 是否为 CUDA 版本\n"
                "3. 可运行：python -c \"import torch; print(torch.cuda.is_available())\"\n"
                "如果只是临时想用 CPU，可运行：python \"z_codes/test/2-train_model.py\" --device cpu"
            )
        return torch.device("cuda")

    if device_text == "cpu":
        return torch.device("cpu")

    if device_text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raise SystemExit(f"[错误] unknown device: {device_text}")


def print_gpu_info(device: torch.device):
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch CUDA: {torch.version.cuda}")


def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            predicted = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    mean_loss = running_loss / total if total else 0.0
    accuracy = correct / total if total else 0.0

    return mean_loss, accuracy


def save_checkpoint(path: Path, model, epoch: int, best_val_acc: float, args, model_run_dir: Path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_class": "HandLandmarkClassifier",
        "input_size": FEATURE_LENGTH,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "model_run_dir": str(model_run_dir),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_args": vars(args),
    }
    torch.save(checkpoint, path)


def write_training_report(
    report_path: Path,
    model_run_dir: Path,
    args,
    train_count: int,
    val_count: int,
    best_val_acc: float,
    best_epoch: int,
    total_time: float,
    device,
):
    lines = [
        "# ASL Model Training Report",
        "",
        "## Tutorial Checks",
        "",
        "- Device default: `cuda`",
        "- Optimizer: `Adam`",
        "- Scheduler: `ReduceLROnPlateau`",
        "- Early stopping: enabled",
        "- Best model checkpoint: `best_model.pth`",
        "- Final model checkpoint: `last_model.pth`",
        "- Gradient accumulation: enabled",
        "",
        "## Run Summary",
        "",
        f"- Model run: `{model_run_dir.name}`",
        f"- Device: `{device}`",
        f"- Train samples: `{train_count}`",
        f"- Validation samples: `{val_count}`",
        f"- Batch size: `{args.batch_size}`",
        f"- Accumulation steps: `{args.accumulation_steps}`",
        f"- Learning rate: `{args.lr}`",
        f"- Max epochs: `{args.max_epochs}`",
        f"- Early stopping patience: `{args.patience}`",
        f"- Best validation accuracy: `{best_val_acc * 100:.2f}%`",
        f"- Best epoch: `{best_epoch}`",
        f"- Total training time: `{total_time:.2f}s`",
        "",
        "## Saved files",
        "",
        "- `best_model.pth`",
        "- `last_model.pth`",
        "- `training_log.csv`",
        "- `training_metrics.json`",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    root = find_project_root()

    parser = argparse.ArgumentParser(description="Train ASL landmark classifier. GPU-first.")
    parser.add_argument("--runs_dir", type=Path, default=root / "runs")
    parser.add_argument(
        "--model_run",
        type=Path,
        default=None,
        help="Optional model run folder. If omitted, an interactive menu is shown.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=5000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="cuda",
        help="Default is cuda. Use --device cpu only for debugging.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.model_run is None:
        model_run_dir = choose_model_run(args.runs_dir)
    else:
        model_run_dir = args.model_run.resolve()

    if not model_run_dir.exists():
        raise SystemExit(f"[错误] model run 不存在：{model_run_dir}")

    csv_dir = model_run_dir / "csv_landmark_dataset"
    training_dir = model_run_dir / "model_training"
    training_dir.mkdir(parents=True, exist_ok=True)

    train_csv_path = csv_dir / "train.csv"
    val_csv_path = csv_dir / "val.csv"

    if not train_csv_path.exists() or not val_csv_path.exists():
        raise SystemExit(f"[错误] 找不到 train.csv / val.csv，请先运行 1-prepare_dataset.py：{csv_dir}")

    x_train, y_train = load_csv(train_csv_path)
    x_val, y_val = load_csv(val_csv_path)

    train_dataset = HandLandmarkDataset(x_train, y_train)
    val_dataset = HandLandmarkDataset(x_val, y_val)

    effective_batch_size = min(args.batch_size, max(2, len(train_dataset)))
    pin_memory = args.device == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        drop_last=len(train_dataset) > effective_batch_size,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    device = resolve_device(args.device)
    print_gpu_info(device)

    model = HandLandmarkClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=10)

    best_val_acc = 0.0
    best_epoch = 0
    early_stopping_counter = 0

    best_model_path = training_dir / "best_model.pth"
    last_model_path = training_dir / "last_model.pth"
    log_path = training_dir / "training_log.csv"
    report_path = training_dir / "training_report.md"
    metrics_path = training_dir / "training_metrics.json"

    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "val_accuracy",
            "val_accuracy_percent",
            "learning_rate",
            "epoch_time_seconds",
        ])

    print("=" * 80)
    print("ASL landmark model training")
    print(f"Model run: {model_run_dir}")
    print(f"Train CSV: {train_csv_path}")
    print(f"Val CSV:   {val_csv_path}")
    print(f"Train rows: {len(train_dataset)}")
    print(f"Val rows:   {len(val_dataset)}")
    print("=" * 80)

    total_start = time.time()
    progress = tqdm(range(1, args.max_epochs + 1), desc="Training epochs")

    for epoch in progress:
        epoch_start = time.time()

        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, (inputs, labels) in enumerate(train_loader, start=1):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels) / args.accumulation_steps
            loss.backward()

            running_loss += loss.item() * args.accumulation_steps * labels.size(0)

            if step % args.accumulation_steps == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

        train_loss = running_loss / len(train_dataset)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        progress.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "val_acc": f"{val_acc * 100:.2f}%",
            "best": f"{best_val_acc * 100:.2f}%",
        })

        with log_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch,
                train_loss,
                val_loss,
                val_acc,
                val_acc * 100,
                current_lr,
                epoch_time,
            ])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            early_stopping_counter = 0
            save_checkpoint(best_model_path, model, epoch, best_val_acc, args, model_run_dir)
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    save_checkpoint(last_model_path, model, best_epoch, best_val_acc, args, model_run_dir)
    total_time = time.time() - total_start

    metrics = {
        "model_run_name": model_run_dir.name,
        "best_val_accuracy": best_val_acc,
        "best_val_accuracy_percent": best_val_acc * 100,
        "best_epoch": best_epoch,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "total_training_time_seconds": total_time,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "pytorch_cuda": torch.version.cuda if device.type == "cuda" else None,
    }

    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    write_training_report(
        report_path=report_path,
        model_run_dir=model_run_dir,
        args=args,
        train_count=len(train_dataset),
        val_count=len(val_dataset),
        best_val_acc=best_val_acc,
        best_epoch=best_epoch,
        total_time=total_time,
        device=device,
    )

    print("=" * 80)
    print("[完成] Model training complete")
    print(f"Best accuracy: {best_val_acc * 100:.2f}% at epoch {best_epoch}")
    print(f"Best model:    {best_model_path}")
    print(f"Last model:    {last_model_path}")
    print(f"Training log:  {log_path}")
    print(f"Report:        {report_path}")
    print(f"Metrics:       {metrics_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
