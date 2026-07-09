"""
4-realtime_inference.py

Realtime ASL inference from webcam.

默认：
- 使用 GPU 推断：--device cuda
- 默认先打开摄像头 0
- 如果摄像头 0 打不开，会扫描可用摄像头并让用户选择
- 如果没有 CUDA，直接报错；临时调试可加 --device cpu

输入：
runs/model_xxx_sample0.xx/model_training/best_model.pth

输出：
runs/model_xxx_sample0.xx/realtime_inference_test/realtime_inference_log_001.txt
"""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


MAX_HANDS = 2
LANDMARKS_PER_HAND = 21
FEATURE_LENGTH = MAX_HANDS * LANDMARKS_PER_HAND * 3
NUM_CLASSES = len(CLASS_NAMES)
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
    print("请选择要用于实时识别的 model run：")
    print("=" * 80)

    for index, folder in enumerate(candidates, start=1):
        model_path = folder / "model_training" / "best_model.pth"
        metrics_path = folder / "model_training" / "training_metrics.json"

        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                acc = metrics.get("best_val_accuracy_percent", None)
                acc_text = f"best acc: {acc:.2f}%" if acc is not None else "best acc: unknown"
            except Exception:
                acc_text = "best acc: unknown"
        else:
            acc_text = "best acc: unknown"

        model_status = "model: yes" if model_path.exists() else "model: no"
        print(f"[{index}] {folder.name:30s} | {model_status:10s} | {acc_text}")

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


def next_log_file(folder: Path, prefix: str, suffix: str = ".txt") -> Path:
    folder.mkdir(parents=True, exist_ok=True)

    max_index = 0
    for file in folder.iterdir():
        if not file.is_file() or not file.name.startswith(prefix + "_"):
            continue
        number_text = file.stem.replace(prefix + "_", "", 1)
        if number_text.isdigit():
            max_index = max(max_index, int(number_text))

    return folder / f"{prefix}_{max_index + 1:03d}{suffix}"


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


def torch_load_any(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(model_path: Path, device):
    if not model_path.exists():
        raise SystemExit(f"[错误] 找不到模型文件：{model_path}")

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


def get_mediapipe_modules():
    import mediapipe as mp_module

    if not hasattr(mp_module, "solutions"):
        raise RuntimeError(
            "当前 mediapipe 安装没有 mp.solutions。请重新安装 mediapipe："
            "python -m pip install --upgrade --force-reinstall mediapipe"
        )

    return mp_module.solutions.hands, mp_module.solutions.drawing_utils


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


def extract_features(results) -> np.ndarray | None:
    if not results.multi_hand_landmarks:
        return None

    features = []

    for hand_landmarks in results.multi_hand_landmarks[:MAX_HANDS]:
        features.extend(normalize_one_hand(hand_landmarks))

    while len(features) < FEATURE_LENGTH:
        features.append(0.0)

    return np.array(features[:FEATURE_LENGTH], dtype=np.float32).reshape(1, -1)


def resolve_device(device_text: str) -> torch.device:
    if device_text == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "[错误] 默认要求使用 GPU/CUDA，但当前 PyTorch 检测不到 CUDA。\n"
                "临时 CPU 调试可运行：python \"z_codes/test/4-realtime_inference.py\" --device cpu"
            )
        return torch.device("cuda")

    if device_text == "cpu":
        return torch.device("cpu")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inference_worker(model, class_names, device, feature_queue, state, stop_event, confidence_threshold):
    while not stop_event.is_set():
        try:
            features = feature_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        with torch.no_grad():
            tensor = torch.from_numpy(features).to(device, non_blocking=True)
            outputs = model(tensor)

            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)

            confidence_value = float(confidence.item())
            predicted_index = int(predicted.item())

        if confidence_value < confidence_threshold:
            label = f"Unknown {confidence_value:.2f}"
        else:
            label = f"{class_names[predicted_index]} {confidence_value:.2f}"

        with state["lock"]:
            state["label"] = label
            state["prediction_count"] += 1

        feature_queue.task_done()


def get_capture_backend():
    """
    Windows 上优先使用 DirectShow，摄像头打开更稳定。
    其他系统使用 OpenCV 默认 backend。
    """
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def get_windows_camera_names() -> list[str]:
    """
    尽量从 Windows 设备管理器读取摄像头显示名称。

    注意：
    OpenCV 的 camera index 和 Windows 设备名称不一定 100% 一一对应。
    所以这里按扫描到的 index 顺序做近似匹配。
    """
    if not sys.platform.startswith("win"):
        return []

    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            "Get-CimInstance Win32_PnPEntity | "
            "Where-Object { "
            "  $_.PNPClass -in @('Camera','Image') -or "
            "  $_.Name -match 'Camera|Webcam|USB Video|Integrated' "
            "} | "
            "Select-Object -ExpandProperty Name"
        ),
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return []

    names = []
    for line in result.stdout.splitlines():
        name = line.strip()
        if name and name not in names:
            names.append(name)

    return names


def try_open_camera_index(camera_index: int):
    backend = get_capture_backend()
    cap = cv2.VideoCapture(camera_index, backend)

    if cap.isOpened():
        success, _ = cap.read()
        if success:
            return cap

    cap.release()
    return None


def scan_available_cameras(max_index: int = 10) -> list[dict]:
    """
    扫描可用摄像头 index，并尽量附带显示名称。
    """
    windows_names = get_windows_camera_names()
    cameras = []

    print("=" * 80)
    print("正在扫描可用摄像头...")
    print("=" * 80)

    for index in range(max_index + 1):
        cap = try_open_camera_index(index)
        if cap is None:
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        cap.release()

        if index < len(windows_names):
            display_name = windows_names[index]
        else:
            display_name = f"Camera index {index}"

        cameras.append({
            "index": index,
            "name": display_name,
            "width": width,
            "height": height,
            "fps": fps,
        })

    return cameras


def choose_camera_interactively(cameras: list[dict]) -> int:
    if not cameras:
        raise SystemExit("[错误] 没有扫描到可用摄像头")

    print("=" * 80)
    print("请选择摄像头：")
    print("=" * 80)

    for number, camera in enumerate(cameras, start=1):
        fps_text = f"{camera['fps']:.1f}" if camera["fps"] else "unknown"
        print(
            f"[{number}] index={camera['index']} | "
            f"{camera['name']} | "
            f"{camera['width']}x{camera['height']} | fps={fps_text}"
        )

    print("[0] 退出")
    print("=" * 80)

    while True:
        choice = input("请输入编号：").strip()

        if choice == "0":
            raise SystemExit("已退出")

        if choice.isdigit():
            number = int(choice)
            if 1 <= number <= len(cameras):
                return int(cameras[number - 1]["index"])

        print("[错误] 请输入有效编号")


def open_camera(camera_index: int, max_scan_index: int = 10):
    """
    默认先尝试 camera_index，当前默认是 0。
    如果失败，列出所有扫描到的摄像头，让用户选择。
    """
    cap = try_open_camera_index(camera_index)

    if cap is not None:
        print(f"[摄像头] 已打开默认摄像头 index={camera_index}")
        return cap, camera_index

    print(f"[警告] 默认摄像头 index={camera_index} 打不开")
    cameras = scan_available_cameras(max_index=max_scan_index)
    selected_index = choose_camera_interactively(cameras)

    cap = try_open_camera_index(selected_index)
    if cap is None:
        raise SystemExit(f"[错误] 选择的摄像头无法打开：{selected_index}")

    print(f"[摄像头] 已打开用户选择的摄像头 index={selected_index}")
    return cap, selected_index


def draw_label(frame, text: str):
    cv2.putText(
        frame,
        f"Predicted: {text}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def parse_args():
    root = find_project_root()

    parser = argparse.ArgumentParser(description="Run realtime ASL inference. GPU-first.")
    parser.add_argument("--runs_dir", type=Path, default=root / "runs")
    parser.add_argument("--model_run", type=Path, default=None)
    parser.add_argument("--camera_index", type=int, default=0, help="Default camera index. If it fails, the script scans available cameras.")
    parser.add_argument("--max_camera_scan", type=int, default=10, help="Maximum camera index to scan when default camera fails.")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--confidence_threshold", type=float, default=0.30)
    return parser.parse_args()


def main():
    args = parse_args()

    model_run_dir = choose_model_run(args.runs_dir) if args.model_run is None else args.model_run.resolve()
    model_path = model_run_dir / "model_training" / "best_model.pth"
    realtime_dir = model_run_dir / "realtime_inference_test"
    log_path = next_log_file(realtime_dir, "realtime_inference_log")

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model, class_names = load_model(model_path, device)

    mp_hands, mp_drawing = get_mediapipe_modules()
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.7,
    )

    cap, actual_camera_index = open_camera(args.camera_index, args.max_camera_scan)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    feature_queue = queue.Queue(maxsize=1)
    state = {"label": "...", "prediction_count": 0, "lock": threading.Lock()}
    stop_event = threading.Event()

    worker = threading.Thread(
        target=inference_worker,
        args=(model, class_names, device, feature_queue, state, stop_event, args.confidence_threshold),
        daemon=True,
    )
    worker.start()

    frame_count = 0
    start_time = time.time()

    log_path.write_text(
        "\n".join([
            "ASL realtime inference log",
            f"Created at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Model run: {model_run_dir}",
            f"Model path: {model_path}",
            f"Device: {device}",
            f"Camera requested first: {args.camera_index}",
            f"Camera used: {actual_camera_index}",
            f"Resolution: {args.width}x{args.height}",
            f"Target FPS: {args.fps}",
            f"Confidence threshold: {args.confidence_threshold}",
            "",
        ]),
        encoding="utf-8",
    )

    print("=" * 80)
    print("ASL realtime inference")
    print(f"Model run: {model_run_dir.name}")
    print(f"Model:     {model_path}")
    print(f"Camera:    {actual_camera_index}")
    print("Press q or Q to quit.")
    print("=" * 80)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("[警告] 摄像头读取失败")
                continue

            frame_count += 1
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                features = extract_features(results)
                if features is not None and not feature_queue.full():
                    feature_queue.put_nowait(features)

                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            with state["lock"]:
                label = state["label"]

            draw_label(frame, label)
            cv2.imshow("ASL Hand Landmark Classification", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break

    finally:
        stop_event.set()
        worker.join(timeout=1.0)
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

        elapsed = time.time() - start_time

        with log_path.open("a", encoding="utf-8") as file:
            file.write(f"Frames processed: {frame_count}\n")
            file.write(f"Predictions made: {state['prediction_count']}\n")
            file.write(f"Elapsed seconds: {elapsed:.2f}\n")

        print("=" * 80)
        print("[完成] Realtime inference stopped")
        print(f"Frames processed: {frame_count}")
        print(f"Predictions made: {state['prediction_count']}")
        print(f"Log: {log_path}")
        print("=" * 80)


if __name__ == "__main__":
    main()
