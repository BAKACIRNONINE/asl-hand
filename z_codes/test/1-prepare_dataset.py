"""
1-prepare_dataset.py

ASL MediaPipe CSV dataset generator.

对应教程：
4.1-4.4 质量检查：
- 归一化逻辑固定：每只手都减 landmark[0]
- CSV 特征列固定：126
- Windows 多进程保护：if __name__ == "__main__"
- tqdm 显示进度

5.1-5.4 架构实现：
- MediaPipe Hands 提取关键点
- wrist normalization
- mirror augmentation
- CSV 中间层

输出结构：
runs/model_001_sample0.05/
    csv_landmark_dataset/
        dataset_info.csv
        train.csv
        val.csv
        sampled_images_manifest.csv
        dataset_generation_report.md
    skipped_images/
    model_run_config.json

说明：
- 使用 mp.solutions.hands
- 不需要 hand_landmarker.task
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import random
import shutil
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
from tqdm import tqdm


MAX_HANDS = 2
LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3
FEATURE_LENGTH = MAX_HANDS * LANDMARKS_PER_HAND * COORDS_PER_LANDMARK
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "datasets").exists() and (parent / "runs").exists():
            return parent
        if (parent / "z_codes").exists() and (parent / "datasets").exists():
            return parent
    return current.parents[2]


def sample_tag(sample_ratio: float) -> str:
    return f"sample{sample_ratio:.2f}"


def safe_note(note: str) -> str:
    note = note.strip()
    note = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in note)
    return note.strip("_")


def get_next_model_run_dir(runs_dir: Path, sample_ratio: float, note: str = "") -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)

    max_index = 0
    for folder in runs_dir.iterdir():
        if not folder.is_dir() or not folder.name.startswith("model_"):
            continue

        parts = folder.name.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            max_index = max(max_index, int(parts[1]))

    folder_name = f"model_{max_index + 1:03d}_{sample_tag(sample_ratio)}"
    note = safe_note(note)
    if note:
        folder_name += f"_{note}"

    model_run_dir = runs_dir / folder_name
    model_run_dir.mkdir(parents=True, exist_ok=False)
    return model_run_dir


def build_csv_header() -> list[str]:
    header = ["image_path", "class_index"]
    for hand_index in range(1, MAX_HANDS + 1):
        for landmark_index in range(LANDMARKS_PER_HAND):
            for coord in ("x", "y", "z"):
                header.append(f"hand{hand_index}_landmark{landmark_index}_{coord}")
    return header


def create_mediapipe_hands(static_image_mode: bool = True):
    import mediapipe as mp_module

    if not hasattr(mp_module, "solutions"):
        raise RuntimeError(
            "当前 mediapipe 安装没有 mp.solutions。请在当前环境中重新安装 mediapipe："
            "python -m pip install --upgrade --force-reinstall mediapipe"
        )

    return mp_module.solutions.hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=0.9,
    )


def normalize_one_hand(hand_landmarks) -> list[float]:
    wrist = hand_landmarks.landmark[0]
    features = []

    for landmark in hand_landmarks.landmark:
        features.extend([
            landmark.x - wrist.x,
            landmark.y - wrist.y,
            landmark.z - wrist.z,
        ])

    return features


def extract_features_from_image(image_path: Path, hands) -> list[float] | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None

    features = []
    for hand_landmarks in results.multi_hand_landmarks[:MAX_HANDS]:
        features.extend(normalize_one_hand(hand_landmarks))

    while len(features) < FEATURE_LENGTH:
        features.append(0.0)

    return features[:FEATURE_LENGTH]


def mirror_features(features: list[float]) -> list[float]:
    mirrored = []
    for index, value in enumerate(features):
        mirrored.append(-value if index % 3 == 0 else value)
    return mirrored


def collect_class_tasks(dataset_dir: Path, sample_ratio: float) -> list[dict]:
    class_tasks = []
    class_dirs = sorted([path for path in dataset_dir.iterdir() if path.is_dir()])

    for class_dir in class_dirs:
        class_name = class_dir.name.strip().upper()

        if len(class_name) != 1 or not class_name.isalpha():
            continue

        class_index = ord(class_name) - ord("A")
        image_paths = sorted([
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ])

        if not image_paths:
            print(f"[跳过] {class_name}: 没有图片")
            continue

        sample_count = int(round(len(image_paths) * sample_ratio))
        if sample_ratio > 0 and sample_count < 1:
            sample_count = 1

        selected_paths = random.sample(image_paths, sample_count) if sample_count < len(image_paths) else image_paths

        print(f"[类别] {class_name}: total={len(image_paths)}, selected={len(selected_paths)}")

        class_tasks.append({
            "class_name": class_name,
            "class_index": class_index,
            "image_paths": selected_paths,
        })

    return class_tasks


def process_class_task(task: dict) -> dict:
    class_name = task["class_name"]
    class_index = task["class_index"]
    image_paths = task["image_paths"]

    rows = []
    manifest_rows = []
    skipped_paths = []

    hands = create_mediapipe_hands(static_image_mode=True)

    try:
        for image_path in image_paths:
            features = extract_features_from_image(image_path, hands)

            if features is None:
                skipped_paths.append(str(image_path.resolve()))
                manifest_rows.append([
                    str(image_path.resolve()),
                    class_index,
                    class_name,
                    "selected",
                    "skipped_no_landmarks",
                ])
                continue

            rows.append([str(image_path.resolve()), class_index] + features)
            rows.append([str(image_path.resolve()), class_index] + mirror_features(features))

            manifest_rows.append([
                str(image_path.resolve()),
                class_index,
                class_name,
                "selected",
                "valid",
            ])
    finally:
        hands.close()

    return {
        "class_name": class_name,
        "class_index": class_index,
        "rows": rows,
        "manifest_rows": manifest_rows,
        "skipped_paths": skipped_paths,
        "selected_count": len(image_paths),
        "valid_count": len(rows) // 2,
        "skipped_count": len(skipped_paths),
    }


def split_rows_by_class(rows: list[list], val_ratio: float, seed: int) -> tuple[list[list], list[list]]:
    rng = random.Random(seed)
    rows_by_class = defaultdict(list)

    for row in rows:
        rows_by_class[int(row[1])].append(row)

    train_rows = []
    val_rows = []

    for class_rows in rows_by_class.values():
        rng.shuffle(class_rows)

        val_count = int(round(len(class_rows) * val_ratio))
        if len(class_rows) > 1 and val_count < 1:
            val_count = 1

        val_rows.extend(class_rows[:val_count])
        train_rows.extend(class_rows[val_count:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    return train_rows, val_rows


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def copy_skipped_images(skipped_paths: list[str], skipped_dir: Path) -> None:
    if not skipped_paths:
        return

    skipped_dir.mkdir(parents=True, exist_ok=True)

    for source_text in skipped_paths:
        source_path = Path(source_text)
        if not source_path.exists():
            continue

        class_name = source_path.parent.name
        target_dir = skipped_dir / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / source_path.name

        if not target_path.exists():
            try:
                shutil.copy2(source_path, target_path)
            except OSError:
                pass


def write_config(
    model_run_dir: Path,
    args,
    dataset_dir: Path,
    row_count: int,
    train_count: int,
    val_count: int,
) -> None:
    config = {
        "model_run_name": model_run_dir.name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stage": "csv_landmark_dataset",
        "sample_ratio": args.sample_ratio,
        "val_ratio": args.val_ratio,
        "source_dataset": str(dataset_dir),
        "uses_mediapipe": "mp.solutions.hands",
        "hand_landmarker_task_required": False,
        "feature_length": FEATURE_LENGTH,
        "max_hands": MAX_HANDS,
        "landmarks_per_hand": LANDMARKS_PER_HAND,
        "mirror_augmentation": True,
        "normalization": "subtract landmark[0] wrist from every landmark",
        "rows_total": row_count,
        "rows_train": train_count,
        "rows_val": val_count,
    }

    (model_run_dir / "model_run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_report(
    report_path: Path,
    args,
    model_run_dir: Path,
    dataset_dir: Path,
    stats: list[dict],
    elapsed: float,
    row_count: int,
    train_count: int,
    val_count: int,
) -> None:
    selected_total = sum(item["selected_count"] for item in stats)
    valid_total = sum(item["valid_count"] for item in stats)
    skipped_total = sum(item["skipped_count"] for item in stats)

    lines = [
        "# ASL CSV Landmark Dataset Generation Report",
        "",
        "## Tutorial Checks",
        "",
        "- MediaPipe feature extractor: `mp.solutions.hands`",
        "- Feature length: `126`",
        "- Normalization: subtract wrist `landmark[0]`",
        "- Mirror augmentation: enabled",
        "- CSV middle layer: enabled",
        "- hand_landmarker.task required: `False`",
        "",
        "## Run Summary",
        "",
        f"- Model run: `{model_run_dir.name}`",
        f"- Dataset dir: `{dataset_dir}`",
        f"- Sample ratio: `{args.sample_ratio}`",
        f"- Validation ratio: `{args.val_ratio}`",
        f"- Workers: `{args.workers}`",
        f"- Selected images: `{selected_total}`",
        f"- Valid images: `{valid_total}`",
        f"- Skipped images: `{skipped_total}`",
        f"- Total CSV rows: `{row_count}`",
        f"- Train rows: `{train_count}`",
        f"- Val rows: `{val_count}`",
        f"- Elapsed seconds: `{elapsed:.2f}`",
        "",
        "## Per-class stats",
        "",
        "| Class | Selected | Valid | Skipped | Rows |",
        "|---|---:|---:|---:|---:|",
    ]

    for item in sorted(stats, key=lambda value: value["class_name"]):
        lines.append(
            f"| {item['class_name']} | {item['selected_count']} | "
            f"{item['valid_count']} | {item['skipped_count']} | {len(item['rows'])} |"
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def ask_sample_ratio(default: float = 0.05) -> float:
    while True:
        user_input = input(f"请输入采样比例 sample ratio，回车默认 {default}，例如 0.05 或 5：").strip()

        if not user_input:
            return default

        try:
            value = float(user_input)
        except ValueError:
            print("[错误] 请输入数字，例如 0.05、5、10")
            continue

        if value > 1:
            value = value / 100.0

        if 0 < value <= 1:
            return value

        print("[错误] 采样比例必须在 0-1 之间，或输入 1-100 的百分比数字")


def parse_args():
    root = find_project_root()

    parser = argparse.ArgumentParser(description="Generate ASL MediaPipe landmark CSV dataset.")
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=root / "datasets" / "ASL_Alphabet_Dataset" / "asl_alphabet_train",
        help="ASL train image dataset folder containing A-Z subfolders.",
    )
    parser.add_argument("--runs_dir", type=Path, default=root / "runs")
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=None,
        help="Image sampling ratio per class, e.g. 0.05. If omitted, the script asks at runtime.",
    )
    parser.add_argument(
        "--no_prompt",
        action="store_true",
        help="Do not ask for sample ratio. Use default 0.05 if --sample_ratio is omitted.",
    )
    parser.add_argument("--val_ratio", type=float, default=0.20)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes. Use 1 for safest Windows debugging; try 2-4 later.",
    )
    parser.add_argument("--note", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save_skipped_images",
        action="store_true",
        help="Copy skipped images into the model run skipped_images folder.",
    )

    return parser.parse_args()


def main():
    mp.freeze_support()
    args = parse_args()
    start_time = time.time()

    if args.sample_ratio is None:
        args.sample_ratio = 0.05 if args.no_prompt else ask_sample_ratio(default=0.05)

    if args.sample_ratio > 1:
        args.sample_ratio = args.sample_ratio / 100.0

    if not args.dataset_dir.exists():
        raise SystemExit(f"[错误] 找不到数据集文件夹：{args.dataset_dir}")

    if args.sample_ratio <= 0:
        raise SystemExit("[错误] sample_ratio 必须大于 0")

    random.seed(args.seed)
    model_run_dir = get_next_model_run_dir(args.runs_dir, args.sample_ratio, args.note)

    csv_dir = model_run_dir / "csv_landmark_dataset"
    skipped_dir = model_run_dir / "skipped_images"
    csv_dir.mkdir(parents=True, exist_ok=True)
    skipped_dir.mkdir(parents=True, exist_ok=True)

    dataset_csv_path = csv_dir / "dataset_info.csv"
    train_csv_path = csv_dir / "train.csv"
    val_csv_path = csv_dir / "val.csv"
    manifest_path = csv_dir / "sampled_images_manifest.csv"
    report_path = csv_dir / "dataset_generation_report.md"

    print("=" * 80)
    print("ASL MediaPipe CSV dataset generation")
    print("=" * 80)
    print(f"Dataset dir: {args.dataset_dir}")
    print(f"Model run:   {model_run_dir}")
    print(f"CSV output:  {csv_dir}")
    print(f"Sample:      {args.sample_ratio}")
    print("=" * 80)

    class_tasks = collect_class_tasks(args.dataset_dir, args.sample_ratio)
    if not class_tasks:
        raise SystemExit("[错误] 没有找到 A-Z 类别文件夹或图片")

    stats = []
    all_rows = []
    all_manifest_rows = []
    all_skipped_paths = []

    if args.workers <= 1:
        for task in tqdm(class_tasks, desc="Processing classes"):
            result = process_class_task(task)
            stats.append(result)
            all_rows.extend(result["rows"])
            all_manifest_rows.extend(result["manifest_rows"])
            all_skipped_paths.extend(result["skipped_paths"])
    else:
        with mp.Pool(processes=args.workers) as pool:
            iterator = pool.imap_unordered(process_class_task, class_tasks)
            for result in tqdm(iterator, total=len(class_tasks), desc="Processing classes"):
                stats.append(result)
                all_rows.extend(result["rows"])
                all_manifest_rows.extend(result["manifest_rows"])
                all_skipped_paths.extend(result["skipped_paths"])

    if not all_rows:
        raise SystemExit("[错误] 没有生成任何有效 landmark row，请检查图片或 MediaPipe 环境")

    header = build_csv_header()
    train_rows, val_rows = split_rows_by_class(
        rows=all_rows,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # 4.2: CSV 列名与模型输入对齐
    expected_columns = 2 + FEATURE_LENGTH
    if len(header) != expected_columns:
        raise RuntimeError(f"CSV header should have {expected_columns} columns, got {len(header)}")

    write_csv(dataset_csv_path, header, all_rows)
    write_csv(train_csv_path, header, train_rows)
    write_csv(val_csv_path, header, val_rows)

    write_csv(
        manifest_path,
        ["image_path", "class_index", "class_name", "selected_status", "landmark_status"],
        all_manifest_rows,
    )

    if args.save_skipped_images:
        copy_skipped_images(all_skipped_paths, skipped_dir)

    elapsed = time.time() - start_time

    write_config(
        model_run_dir=model_run_dir,
        args=args,
        dataset_dir=args.dataset_dir,
        row_count=len(all_rows),
        train_count=len(train_rows),
        val_count=len(val_rows),
    )

    write_report(
        report_path=report_path,
        args=args,
        model_run_dir=model_run_dir,
        dataset_dir=args.dataset_dir,
        stats=stats,
        elapsed=elapsed,
        row_count=len(all_rows),
        train_count=len(train_rows),
        val_count=len(val_rows),
    )

    print("=" * 80)
    print("[完成] CSV landmark dataset generated")
    print(f"Model run:       {model_run_dir}")
    print(f"dataset_info:    {dataset_csv_path}")
    print(f"train.csv:       {train_csv_path}")
    print(f"val.csv:         {val_csv_path}")
    print(f"manifest:        {manifest_path}")
    print(f"report:          {report_path}")
    print(f"rows total:      {len(all_rows)}")
    print(f"rows train/val:  {len(train_rows)} / {len(val_rows)}")
    print(f"time:            {elapsed:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
