"""
asl_eda_one_click_debug.py

One-click ASL EDA exporter with verbose progress, step-level error handling,
and an error log. Designed for Windows / VS Code terminal.

Default project path:
    C:\\Users\\haha2\\Desktop\\CS\\ASL Hand

Run:
    python asl_eda_one_click_debug.py

Optional:
    python asl_eda_one_click_debug.py --project "C:\\Users\\haha2\\Desktop\\CS\\ASL Hand"
    python asl_eda_one_click_debug.py --dataset "C:\\...\\your_dataset.csv"

Output:
    <project>\\eda_export_debug\\
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable, Optional

try:
    import pandas as pd
except Exception as e:
    print("ERROR: pandas is not installed. Run: python -m pip install pandas matplotlib")
    raise

try:
    import matplotlib.pyplot as plt
except Exception as e:
    print("ERROR: matplotlib is not installed. Run: python -m pip install pandas matplotlib")
    raise

DEFAULT_PROJECT = r"C:\Users\haha2\Desktop\CS\ASL Hand"
OUT_FOLDER_NAME = "eda_export_debug"
MAX_COPY_FILE_MB = 80
SCAN_TIMEOUT_SECONDS = 180

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
TEXT_EXTS = {".txt", ".md", ".csv", ".json"}
EVAL_KEYWORDS = [
    "classification_report",
    "confusion_matrix",
    "evaluation_metrics",
    "evaluation_report",
    "training_curve",
    "top_confusions",
    "wrong_predictions",
    "model_comparison",
    "metrics",
]
SKIP_DIR_NAMES = {
    OUT_FOLDER_NAME.lower(),
    "eda_export",
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".idea",
    ".vscode",
    "node_modules",
}
# Avoid scanning raw image/video folders forever. EDA only needs CSV/evaluation outputs.
SKIP_HEAVY_DIR_NAMES = {
    "images", "image", "videos", "video", "frames", "raw", "raw_data",
    "dataset_images", "screenshots", "cache"
}


def now() -> str:
    return time.strftime("%H:%M:%S")


class Logger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("ASL EDA debug log\n", encoding="utf-8")

    def write(self, msg: str) -> None:
        line = f"[{now()}] {msg}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def error(self, msg: str) -> None:
        self.write("ERROR: " + msg)


class StepProgress:
    def __init__(self, total_steps: int, logger: Logger):
        self.total_steps = total_steps
        self.step = 0
        self.logger = logger

    def next(self, msg: str) -> None:
        self.step += 1
        pct = self.step / self.total_steps * 100
        self.logger.write(f"STEP {self.step}/{self.total_steps} ({pct:.1f}%): {msg}")


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")[:180]


def is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def should_skip_dir(dir_path: Path) -> bool:
    lname = dir_path.name.lower()
    if lname in SKIP_DIR_NAMES:
        return True
    if lname in SKIP_HEAVY_DIR_NAMES:
        return True
    return False


def walk_files_fast(root: Path, logger: Logger, timeout_seconds: int = SCAN_TIMEOUT_SECONDS):
    start = time.time()
    checked = 0
    for current_root, dirs, files in os.walk(root):
        current_path = Path(current_root)
        dirs[:] = [d for d in dirs if not should_skip_dir(current_path / d)]
        checked += len(files)
        if checked % 500 == 0:
            logger.write(f"Scanning... checked about {checked} files, current folder: {current_path}")
        if time.time() - start > timeout_seconds:
            raise TimeoutError(f"Scanning took over {timeout_seconds}s. Last folder: {current_path}")
        for fname in files:
            yield current_path / fname


def find_dataset_csv(project: Path, out: Path, logger: Logger) -> Optional[Path]:
    logger.write("Searching for dataset CSV files...")
    candidates = []
    for p in walk_files_fast(project, logger, timeout_seconds=SCAN_TIMEOUT_SECONDS):
        if out in p.parents:
            continue
        if p.suffix.lower() != ".csv":
            continue
        name = p.name.lower()
        if any(x in name for x in ["dataset", "landmark", "train", "val", "test", "asl", "features"]):
            try:
                candidates.append(p)
            except Exception:
                pass
    if not candidates:
        logger.write("No dataset CSV found automatically.")
        return None
    candidates.sort(key=lambda x: x.stat().st_size, reverse=True)
    logger.write("Dataset CSV candidates, largest first:")
    for c in candidates[:8]:
        logger.write(f"  {c} ({c.stat().st_size / 1024 / 1024:.2f} MB)")
    return candidates[0]


def detect_label_col(df: pd.DataFrame) -> Optional[str]:
    preferred = ["label", "class", "gesture", "letter", "Class", "Label", "Gesture", "Class Index", "class_index", "target"]
    for col in preferred:
        if col in df.columns:
            return col
    for col in df.columns:
        try:
            nunique = df[col].nunique(dropna=True)
            if 1 < nunique <= 60 and not pd.api.types.is_float_dtype(df[col]):
                return col
        except Exception:
            continue
    return None


def detect_split_col(df: pd.DataFrame) -> Optional[str]:
    for col in ["split", "Split", "subset", "Subset", "set", "Set", "phase"]:
        if col in df.columns:
            return col
    return None


def save_bar(series: pd.Series, title: str, xlabel: str, ylabel: str, out: Path, rotate: int = 45) -> None:
    plt.figure(figsize=(10, 5))
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotate, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def save_hist(series: pd.Series, title: str, xlabel: str, out: Path) -> None:
    plt.figure(figsize=(8, 5))
    pd.to_numeric(series, errors="coerce").dropna().plot(kind="hist", bins=30)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def dataset_eda(dataset_path: Optional[Path], out: Path, logger: Logger) -> dict:
    report = {"dataset_found": False}
    if not dataset_path or not dataset_path.exists():
        logger.write("Skipping dataset EDA: no dataset CSV path.")
        return report

    logger.write(f"Reading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    logger.write(f"Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")

    summary_dir = out / "dataset_summary"
    chart_dir = out / "charts"
    summary_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)

    label_col = detect_label_col(df)
    split_col = detect_split_col(df)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    report = {
        "dataset_found": True,
        "dataset_path": str(dataset_path),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "label_col": label_col,
        "split_col": split_col,
        "numeric_columns": len(numeric_cols),
        "duplicate_rows": int(df.duplicated().sum()),
        "total_missing_values": int(df.isna().sum().sum()),
    }

    column_dict = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing": [int(df[c].isna().sum()) for c in df.columns],
        "missing_rate": [float(df[c].isna().mean()) for c in df.columns],
        "unique_values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    column_dict.to_csv(summary_dir / "column_dictionary_auto.csv", index=False)

    if numeric_cols:
        df[numeric_cols].describe().T.to_csv(summary_dir / "numeric_descriptive_statistics.csv")

    if label_col:
        logger.write(f"Detected label column: {label_col}")
        label_counts = df[label_col].value_counts(dropna=False).sort_index()
        label_counts.to_csv(summary_dir / "class_distribution.csv", header=["count"])
        save_bar(label_counts, "ASL Class Distribution", label_col, "Sample Count", chart_dir / "class_distribution.png")
        report["classes"] = int(df[label_col].nunique(dropna=True))
        report["smallest_class_count"] = int(label_counts.min())
        report["largest_class_count"] = int(label_counts.max())
    else:
        logger.write("No label column detected.")

    if split_col:
        logger.write(f"Detected split column: {split_col}")
        split_counts = df[split_col].value_counts(dropna=False)
        split_counts.to_csv(summary_dir / "split_distribution.csv", header=["count"])
        save_bar(split_counts, "Train / Validation / Test Split Distribution", split_col, "Sample Count", chart_dir / "split_distribution.png")
        if label_col:
            ctab = pd.crosstab(df[label_col], df[split_col])
            ctab.to_csv(summary_dir / "class_by_split_table.csv")
            plt.figure(figsize=(12, 6))
            ctab.plot(kind="bar", stacked=False, ax=plt.gca())
            plt.title("Class Distribution by Split")
            plt.xlabel(label_col)
            plt.ylabel("Sample Count")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(chart_dir / "class_by_split.png", dpi=200)
            plt.close()

    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0].head(25)
    if len(missing) > 0:
        save_bar(missing, "Top Missing Values by Column", "Column", "Missing Count", chart_dir / "missing_values_top25.png")

    for col in numeric_cols[:6]:
        save_hist(df[col], f"Distribution of {col}", col, chart_dir / f"hist_{safe_name(col)}.png")

    xyz_cols = [c for c in numeric_cols if re.search(r"(^|_)(x|y|z)(_|$)|landmark", c, re.I)]
    if len(xyz_cols) >= 3:
        sample_cols = xyz_cols[:min(126, len(xyz_cols))]
        feature_abs_mean = df[sample_cols].abs().mean(axis=1)
        feature_abs_mean.to_frame("mean_abs_landmark_value").describe().to_csv(summary_dir / "landmark_magnitude_descriptive_statistics.csv")
        save_hist(feature_abs_mean, "Mean Absolute Landmark Feature Value", "Mean absolute value", chart_dir / "mean_abs_landmark_value_hist.png")
        if label_col:
            grouped = pd.DataFrame({label_col: df[label_col], "mean_abs_landmark_value": feature_abs_mean}).groupby(label_col)["mean_abs_landmark_value"].mean().sort_index()
            grouped.to_csv(summary_dir / "landmark_magnitude_by_class.csv", header=["mean_abs_landmark_value"])
            save_bar(grouped, "Mean Landmark Magnitude by ASL Class", label_col, "Mean absolute landmark value", chart_dir / "mean_landmark_magnitude_by_class.png")

    return report


def copy_eval_files(project: Path, out: Path, logger: Logger) -> list[dict]:
    eval_out = out / "model_evaluation_files"
    eval_out.mkdir(parents=True, exist_ok=True)
    copied: list[dict] = []
    skipped_large: list[str] = []
    start = time.time()
    logger.write("Copying evaluation files. This version prints every matched file and skips large files.")

    for p in walk_files_fast(project, logger, timeout_seconds=SCAN_TIMEOUT_SECONDS):
        if out in p.parents:
            continue
        lower = p.name.lower()
        if not any(k in lower for k in EVAL_KEYWORDS):
            continue
        if p.suffix.lower() not in (IMAGE_EXTS | TEXT_EXTS):
            continue
        try:
            size_mb = p.stat().st_size / 1024 / 1024
            if size_mb > MAX_COPY_FILE_MB:
                skipped_large.append(str(p))
                logger.write(f"SKIP large eval file: {p} ({size_mb:.1f} MB)")
                continue
            rel = p.relative_to(project)
            dest = eval_out / safe_name(str(rel).replace("\\", "_").replace("/", "_"))
            logger.write(f"Copying: {p} -> {dest}")
            shutil.copy2(p, dest)
            copied.append({"source": str(p), "copied_to": str(dest), "size_bytes": p.stat().st_size})
        except Exception as e:
            logger.error(f"Failed to copy {p}: {repr(e)}")

    (eval_out / "copied_evaluation_files_manifest.csv").write_text(
        "source,copied_to,size_bytes\n" + "".join(
            f'"{x["source"]}","{x["copied_to"]}",{x["size_bytes"]}\n' for x in copied
        ),
        encoding="utf-8",
    )
    if skipped_large:
        (eval_out / "skipped_large_files.txt").write_text("\n".join(skipped_large), encoding="utf-8")
    logger.write(f"Copied {len(copied)} evaluation files in {time.time() - start:.1f}s")
    return copied


def extract_metric_value(text: str, names: Iterable[str]) -> Optional[float]:
    for name in names:
        patterns = [rf"{re.escape(name)}\s*[:=]\s*([0-9]*\.?[0-9]+)", rf"{re.escape(name)}[^0-9]+([0-9]*\.?[0-9]+)"]
        for pat in patterns:
            m = re.search(pat, text, flags=re.I)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass
    return None


def model_comparison(project: Path, out: Path, logger: Logger) -> pd.DataFrame:
    comp_dir = out / "model_comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    logger.write("Searching model_* folders for comparison metrics...")

    model_dirs = []
    for p in walk_files_fast(project, logger, timeout_seconds=SCAN_TIMEOUT_SECONDS):
        # collect parent dirs whose name starts model_
        for parent in [p.parent] + list(p.parents):
            if parent == project.parent:
                break
            if parent.name.lower().startswith("model_") and out not in parent.parents:
                model_dirs.append(parent)
                break
    model_dirs = sorted(set(model_dirs))
    logger.write(f"Found {len(model_dirs)} model_* folders")

    for run_dir in model_dirs:
        metrics = {"run": run_dir.name, "path": str(run_dir)}
        logger.write(f"Reading model folder: {run_dir}")
        try:
            for current_root, dirs, files in os.walk(run_dir):
                dirs[:] = [d for d in dirs if not should_skip_dir(Path(current_root) / d)]
                for fname in files:
                    file = Path(current_root) / fname
                    lower = file.name.lower()
                    if file.suffix.lower() == ".csv" and any(k in lower for k in ["metrics", "report", "comparison", "result"]):
                        try:
                            df = pd.read_csv(file)
                            metrics[f"file_{file.name}"] = str(file)
                            for col in df.columns:
                                lc = col.lower()
                                if any(k in lc for k in ["acc", "accuracy"]):
                                    vals = pd.to_numeric(df[col], errors="coerce")
                                    if vals.notna().sum():
                                        metrics.setdefault("accuracy_from_csv", float(vals.max()))
                                if "f1" in lc:
                                    vals = pd.to_numeric(df[col], errors="coerce")
                                    if vals.notna().sum():
                                        metrics.setdefault("f1_from_csv", float(vals.max()))
                        except Exception as e:
                            logger.error(f"Metric CSV read failed {file}: {repr(e)}")
                    elif file.suffix.lower() in {".txt", ".md", ".json"} and any(k in lower for k in ["metrics", "report", "classification"]):
                        try:
                            text = file.read_text(encoding="utf-8", errors="ignore")
                            metrics.setdefault("accuracy_from_text", extract_metric_value(text, ["accuracy", "val_acc", "validation accuracy", "test accuracy"]))
                            metrics.setdefault("f1_from_text", extract_metric_value(text, ["macro avg", "weighted avg", "f1-score", "f1"]))
                            metrics[f"file_{file.name}"] = str(file)
                        except Exception as e:
                            logger.error(f"Metric text read failed {file}: {repr(e)}")
        except Exception as e:
            logger.error(f"Failed inside model folder {run_dir}: {repr(e)}")
        rows.append(metrics)

    df_runs = pd.DataFrame(rows)
    if not df_runs.empty:
        df_runs.to_csv(comp_dir / "model_run_comparison_auto.csv", index=False)
        metric_cols = [c for c in df_runs.columns if c in ["accuracy_from_csv", "accuracy_from_text", "f1_from_csv", "f1_from_text"]]
        for col in metric_cols:
            values = pd.to_numeric(df_runs[col], errors="coerce")
            if values.notna().sum() > 0:
                plt.figure(figsize=(10, 5))
                plt.bar(df_runs["run"], values)
                plt.title(f"Model Comparison: {col}")
                plt.xlabel("Model Run")
                plt.ylabel(col)
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.savefig(comp_dir / f"model_comparison_{col}.png", dpi=200)
                plt.close()
    else:
        (comp_dir / "model_run_comparison_auto.csv").write_text("run,path\n", encoding="utf-8")
    return df_runs


def write_readme(out: Path, report: dict, copied: list[dict], run_df: pd.DataFrame, logger: Logger) -> None:
    lines = []
    lines.append("# ASL EDA Export Package\n\n")
    lines.append("This folder contains materials for the ASL hand gesture recognition EDA report.\n\n")
    lines.append("## Dataset summary\n")
    for k, v in report.items():
        lines.append(f"- {k}: {v}\n")
    lines.append("\n## Copied model evaluation files\n")
    if copied:
        for item in copied:
            lines.append(f"- {item['copied_to']}  (from {item['source']})\n")
    else:
        lines.append("- No evaluation files were copied. Check names such as classification_report, confusion_matrix, evaluation_metrics, training_curve, wrong_predictions.\n")
    lines.append("\n## Model comparison\n")
    if not run_df.empty:
        lines.append("- See model_comparison/model_run_comparison_auto.csv\n")
    else:
        lines.append("- No model_* run folders found.\n")
    lines.append("\n## Suggested report focus\n")
    lines.append("- Compare sampling versions such as 005 vs 010 using class balance, validation accuracy, confusion matrix, and wrong-prediction patterns.\n")
    lines.append("- Use class distribution and split distribution charts as basic EDA visuals.\n")
    lines.append("- Use evaluation_metrics, classification_report, and confusion_matrix as model-analysis evidence.\n")
    lines.append("- Optimization discussion: collect more weak-class samples, balance classes, standardize camera distance/lighting, check landmark normalization, and tune training/early stopping.\n")
    (out / "README_FOR_REPORT.md").write_text("".join(lines), encoding="utf-8")
    logger.write("README_FOR_REPORT.md written")


def run_step(name: str, fn, logger: Logger, fatal: bool = False):
    logger.write(f"BEGIN: {name}")
    start = time.time()
    try:
        result = fn()
        logger.write(f"OK: {name} ({time.time() - start:.1f}s)")
        return result
    except Exception as e:
        logger.error(f"FAILED step '{name}': {repr(e)}")
        logger.error(traceback.format_exc())
        if fatal:
            raise
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="", help="ASL Hand project folder")
    parser.add_argument("--dataset", type=str, default="", help="Optional exact dataset CSV path")
    args = parser.parse_args()

    print("ASL EDA one-click DEBUG extractor")
    print("Default project path:")
    print(DEFAULT_PROJECT)
    project_text = args.project.strip()
    if not project_text:
        typed = input("Press Enter to use default, or paste another ASL Hand folder path: ").strip().strip('"')
        project_text = typed if typed else DEFAULT_PROJECT

    project = Path(project_text).expanduser().resolve()
    if not project.exists():
        print(f"ERROR: Project folder not found: {project}")
        input("Press Enter to exit...")
        return

    out = project / OUT_FOLDER_NAME
    out.mkdir(parents=True, exist_ok=True)
    logger = Logger(out / "ERROR_LOG.txt")
    progress = StepProgress(5, logger)

    logger.write(f"Project: {project}")
    logger.write(f"Output: {out}")
    logger.write(f"Python: {sys.executable}")

    progress.next("find dataset")
    if args.dataset.strip():
        dataset_path = Path(args.dataset.strip().strip('"')).expanduser().resolve()
        logger.write(f"Using manually specified dataset: {dataset_path}")
    else:
        dataset_path = run_step("find dataset CSV", lambda: find_dataset_csv(project, out, logger), logger)

    progress.next("dataset EDA and charts")
    report = run_step("dataset EDA", lambda: dataset_eda(dataset_path, out, logger), logger) or {"dataset_found": False}

    progress.next("copy evaluation files")
    copied = run_step("copy evaluation files", lambda: copy_eval_files(project, out, logger), logger) or []

    progress.next("model comparison")
    run_df = run_step("model comparison", lambda: model_comparison(project, out, logger), logger)
    if run_df is None:
        run_df = pd.DataFrame()

    progress.next("write report README")
    run_step("write README", lambda: write_readme(out, report, copied, run_df, logger), logger)

    logger.write("DONE")
    print("\nFinished. Output folder:")
    print(out)
    print("\nIf anything failed, send me this file:")
    print(out / "ERROR_LOG.txt")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nFATAL ERROR:\n")
        print(traceback.format_exc())
    input("\nPress Enter to exit...")
