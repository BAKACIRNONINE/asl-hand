r"""
part_e_prepare_asl36_from_local.py

One-click Part E helper:
1. Inspect local A-Z and digit datasets
2. Convert digit X.npy / Y.npy into image folders 0-9
3. Create z_codes/ASL_36 from z_codes/ASL_26 or z_codes/test
4. Patch common 26-class constants into 36-class constants
5. Write guidance/part_e_local_dataset_report.md

Run from ASL Hand root:
  python ai_guidance/part_e_prepare_asl36_from_local.py
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path


CLASS_36_LINE = 'CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + list("0123456789")\nNUM_CLASSES = len(CLASS_NAMES)\n'


def project_root() -> Path:
    here = Path(__file__).resolve()
    if here.parent.name.lower() == "ai_guidance":
        return here.parent.parent
    return Path.cwd().resolve()


def find_file(root: Path, name: str):
    hits = list(root.rglob(name))
    return hits[0] if hits else None


def find_digits_npy(root: Path):
    x = find_file(root / "datasets", "X.npy")
    y = find_file(root / "datasets", "Y.npy")
    return x, y


def convert_digits_npy(root: Path, report: list[str]):
    x_path, y_path = find_digits_npy(root)

    if not x_path or not y_path:
        report.append("- Digits X.npy / Y.npy not found")
        return None

    report.append(f"- Found digits X.npy: `{x_path.relative_to(root)}`")
    report.append(f"- Found digits Y.npy: `{y_path.relative_to(root)}`")

    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "pillow"])
        import numpy as np
        from PIL import Image

    X = np.load(x_path)
    Y = np.load(y_path)

    report.append(f"- X shape: `{X.shape}`")
    report.append(f"- Y shape: `{Y.shape}`")

    out_dir = root / "datasets" / "asl_digits_images"

    if out_dir.exists() and any(out_dir.rglob("*.png")):
        report.append(f"- Digits images already exist: `{out_dir.relative_to(root)}`")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    if Y.ndim == 2:
        labels = Y.argmax(axis=1)
    else:
        labels = Y.astype("int64").reshape(-1)

    count = 0
    for i, img in enumerate(X):
        label = str(int(labels[i]))
        label_dir = out_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        arr = img
        if arr.max() <= 1:
            arr = arr * 255
        arr = arr.astype("uint8")

        if arr.ndim == 2:
            image = Image.fromarray(arr, mode="L").convert("RGB")
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            image = Image.fromarray(arr.squeeze(-1), mode="L").convert("RGB")
        else:
            image = Image.fromarray(arr).convert("RGB")

        image.save(label_dir / f"digit_{i:05d}.png")
        count += 1

    report.append(f"- Converted digit npy to images: `{out_dir.relative_to(root)}`")
    report.append(f"- Digit image count: `{count}`")
    return out_dir


def copy_asl36_code(root: Path, report: list[str]):
    src_candidates = [
        root / "z_codes" / "ASL_26",
        root / "z_codes" / "test",
        root / "z_codes" / "latest",
    ]

    src = next((p for p in src_candidates if p.exists()), None)
    dst = root / "z_codes" / "ASL_36"

    if not src:
        report.append("- Source code folder not found: tried z_codes/ASL_26, z_codes/test, z_codes/latest")
        return None

    dst.mkdir(parents=True, exist_ok=True)

    copied = 0
    for file in src.glob("*.py"):
        target = dst / file.name
        if not target.exists():
            shutil.copy2(file, target)
            copied += 1

    report.append(f"- ASL_36 source folder: `{dst.relative_to(root)}`")
    report.append(f"- Copied {copied} new .py files from `{src.relative_to(root)}`")
    return dst


def patch_python_file(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    original = text

    # Replace common class definitions.
    text = re.sub(
        r'CLASS_NAMES\s*=\s*list\("ABCDEFGHIJKLMNOPQRSTUVWXYZ"\)\s*\n\s*NUM_CLASSES\s*=\s*26',
        CLASS_36_LINE.strip(),
        text,
    )
    text = re.sub(
        r'CLASS_NAMES\s*=\s*\[chr\(ord\("A"\)\s*\+\s*i\)\s*for\s*i\s*in\s*range\(26\)\]\s*\n\s*NUM_CLASSES\s*=\s*26',
        CLASS_36_LINE.strip(),
        text,
    )
    text = re.sub(
        r'CLASS_NAMES\s*=\s*\[chr\(ord\(\'A\'\)\s*\+\s*i\)\s*for\s*i\s*in\s*range\(26\)\]\s*\n\s*NUM_CLASSES\s*=\s*26',
        CLASS_36_LINE.strip(),
        text,
    )

    # Replace hard-coded output sizes where obvious.
    text = text.replace("nn.Linear(128, 26)", "nn.Linear(128, NUM_CLASSES)")
    text = text.replace("Linear(128, 26)", "Linear(128, NUM_CLASSES)")
    text = text.replace("num_classes=26", "num_classes=NUM_CLASSES")
    text = text.replace("NUM_CLASSES = 26", "NUM_CLASSES = len(CLASS_NAMES)")

    # If no CLASS_NAMES exists in training/inference scripts, insert after imports.
    if "CLASS_NAMES" not in text and path.name.startswith(("2-", "3-", "4-", "5-", "6-")):
        lines = text.splitlines(True)
        insert_at = 0
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                insert_at = i + 1
        lines.insert(insert_at, "\n" + CLASS_36_LINE + "\n")
        text = "".join(lines)

    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def patch_asl36(dst: Path, report: list[str]):
    changed = []
    for py in dst.glob("*.py"):
        if patch_python_file(py):
            changed.append(py.name)

    if changed:
        report.append("- Patched 36-class constants in:")
        for name in changed:
            report.append(f"  - `{name}`")
    else:
        report.append("- No Python files patched automatically. Manual check required.")


def write_report(root: Path, report: list[str]):
    guidance = root / "guidance"
    guidance.mkdir(exist_ok=True)
    path = guidance / "part_e_local_dataset_report.md"
    path.write_text(
        "# Part E Local Dataset Report\n\n"
        + "\n".join(report)
        + "\n\n## Required ASL_36 code changes\n\n"
        "```text\n"
        "1. CLASS_NAMES: A-Z + 0-9\n"
        "2. NUM_CLASSES: 36\n"
        "3. Model output: nn.Linear(128, NUM_CLASSES)\n"
        "4. Prepare script: read alphabet image folders and digit image folders\n"
        "5. Train/evaluate/inference/web demo: load ASL_36 model and 36-class label list\n"
        "6. Output folders: use runs/model_36_* to avoid overwriting ASL_26 results\n"
        "```\n",
        encoding="utf-8",
    )
    return path


def main():
    root = project_root()
    report = []
    report.append(f"- Project root: `{root}`")

    alphabet_candidates = list((root / "datasets").rglob("ASL_Alphabet_Dataset")) + list((root / "datasets").rglob("asl_alphabet"))
    if alphabet_candidates:
        report.append(f"- Found alphabet dataset: `{alphabet_candidates[0].relative_to(root)}`")
    else:
        report.append("- Alphabet dataset folder not found")

    convert_digits_npy(root, report)

    dst = copy_asl36_code(root, report)
    if dst:
        patch_asl36(dst, report)

    report_path = write_report(root, report)

    print("[DONE]")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
