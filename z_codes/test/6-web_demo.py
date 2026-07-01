"""
6-web_demo.py

ASL Hand Web Demo for tutorial extension D.

Goal:
- Wrap the trained PyTorch ASL model as a small REST API
- Use the browser webcam as the frontend camera
- Send webcam frames to the backend
- Backend uses MediaPipe Hands + PyTorch model to predict ASL letters

Input:
runs/model_xxx_sample0.xx/
    model_training/best_model.pth

Run:
    python "z_codes/test/6-web_demo.py"

Open:
    http://127.0.0.1:8000

Dependencies:
    python -m pip install fastapi uvicorn

Notes:
- MediaPipe landmark extraction still runs on CPU through mp.solutions.hands
- PyTorch MLP inference defaults to GPU: --device cuda
- This is a local teaching/demo web app, not a production deployment
"""

from __future__ import annotations

import argparse
import base64
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel


MAX_HANDS = 2
LANDMARKS_PER_HAND = 21
FEATURE_LENGTH = MAX_HANDS * LANDMARKS_PER_HAND * 3
NUM_CLASSES = 26
CLASS_NAMES = [chr(ord("A") + index) for index in range(NUM_CLASSES)]


HTML_PAGE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>ASL Hand Web Demo</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body {
      margin: 0;
      font-family: Arial, sans-serif;
      background: #111827;
      color: #f9fafb;
    }
    .container {
      max-width: 980px;
      margin: 0 auto;
      padding: 24px;
    }
    h1 {
      margin-bottom: 8px;
    }
    .subtitle {
      color: #cbd5e1;
      margin-bottom: 20px;
    }
    .layout {
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 20px;
    }
    video, canvas {
      width: 100%;
      border-radius: 14px;
      background: #000;
      box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    }
    .panel {
      background: #1f2937;
      border-radius: 14px;
      padding: 18px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }
    .prediction {
      font-size: 64px;
      font-weight: bold;
      margin: 10px 0;
      color: #22c55e;
    }
    .small {
      color: #cbd5e1;
      font-size: 14px;
      line-height: 1.5;
    }
    button {
      border: 0;
      border-radius: 10px;
      padding: 10px 14px;
      margin-right: 8px;
      cursor: pointer;
      background: #2563eb;
      color: white;
      font-weight: bold;
    }
    button.stop {
      background: #dc2626;
    }
    .row {
      margin: 12px 0;
    }
    code {
      background: #0f172a;
      padding: 2px 5px;
      border-radius: 4px;
    }
    @media (max-width: 780px) {
      .layout {
        grid-template-columns: 1fr;
      }
      .prediction {
        font-size: 48px;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>ASL Hand Web Demo</h1>
    <div class="subtitle">
      Browser camera → FastAPI backend → MediaPipe landmarks → PyTorch ASL prediction
    </div>

    <div class="layout">
      <div>
        <video id="video" autoplay playsinline muted></video>
        <canvas id="canvas" width="320" height="240" style="display:none;"></canvas>
      </div>

      <div class="panel">
        <div class="row">
          <button id="startBtn">Start Camera</button>
          <button id="stopBtn" class="stop">Stop</button>
        </div>

        <div class="small">Prediction</div>
        <div id="prediction" class="prediction">...</div>

        <div class="row small">
          Confidence: <span id="confidence">-</span>
        </div>
        <div class="row small">
          Hands detected: <span id="hands">-</span>
        </div>
        <div class="row small">
          Backend latency: <span id="latency">-</span> ms
        </div>
        <div class="row small">
          Status: <span id="status">idle</span>
        </div>

        <hr style="border-color:#374151;" />

        <div class="small">
          This demo sends one JPEG frame every <code>150ms</code> to the local backend.
          It is for local project demonstration only.
        </div>
      </div>
    </div>
  </div>

<script>
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");

const predictionEl = document.getElementById("prediction");
const confidenceEl = document.getElementById("confidence");
const handsEl = document.getElementById("hands");
const latencyEl = document.getElementById("latency");
const statusEl = document.getElementById("status");

let stream = null;
let timer = null;
let busy = false;

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: "user"
      },
      audio: false
    });

    video.srcObject = stream;
    statusEl.textContent = "camera running";

    if (timer) clearInterval(timer);
    timer = setInterval(sendFrame, 150);
  } catch (err) {
    statusEl.textContent = "camera error: " + err;
  }
}

function stopCamera() {
  if (timer) {
    clearInterval(timer);
    timer = null;
  }

  if (stream) {
    for (const track of stream.getTracks()) {
      track.stop();
    }
    stream = null;
  }

  video.srcObject = null;
  statusEl.textContent = "stopped";
}

async function sendFrame() {
  if (!stream || busy) return;
  if (video.videoWidth === 0 || video.videoHeight === 0) return;

  busy = true;

  canvas.width = 320;
  canvas.height = 240;

  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
  ctx.restore();

  const imageData = canvas.toDataURL("image/jpeg", 0.75);
  const t0 = performance.now();

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ image: imageData })
    });

    const result = await response.json();
    const t1 = performance.now();

    if (result.ok) {
      predictionEl.textContent = result.label;
      confidenceEl.textContent = result.confidence.toFixed(3);
      handsEl.textContent = result.hands_detected;
      latencyEl.textContent = Math.round(t1 - t0);
      statusEl.textContent = "ok";
    } else {
      predictionEl.textContent = "...";
      confidenceEl.textContent = "-";
      handsEl.textContent = result.hands_detected ?? "-";
      latencyEl.textContent = Math.round(t1 - t0);
      statusEl.textContent = result.error || "no prediction";
    }
  } catch (err) {
    statusEl.textContent = "request error";
  } finally {
    busy = false;
  }
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);
</script>
</body>
</html>
"""


class FrameRequest(BaseModel):
    image: str


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


@dataclass
class ServerState:
    model_run_dir: Path
    model_path: Path
    device: torch.device
    model: nn.Module
    class_names: list[str]
    hands: object
    mediapipe_lock: threading.Lock


APP_STATE: ServerState | None = None
app = FastAPI(title="ASL Hand Web Demo")


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
    print("请选择用于 Web Demo 的 model run：")
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


def resolve_device(device_text: str) -> torch.device:
    if device_text == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "[错误] 默认要求使用 GPU/CUDA，但当前 PyTorch 检测不到 CUDA。\n"
                "临时 CPU 调试可运行：python \"z_codes/test/6-web_demo.py\" --device cpu"
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


def load_model(model_path: Path, device: torch.device):
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


def create_mediapipe_hands():
    import mediapipe as mp

    if not hasattr(mp, "solutions"):
        raise RuntimeError(
            "当前 mediapipe 没有 mp.solutions。"
            "本项目 Web Demo 需要 mediapipe==0.10.21。"
        )

    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.7,
    )


def decode_base64_image(image_text: str) -> np.ndarray:
    if "," in image_text:
        image_text = image_text.split(",", 1)[1]

    image_bytes = base64.b64decode(image_text)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame_bgr is None:
        raise ValueError("Could not decode image")

    return frame_bgr


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


def extract_features_from_frame(frame_bgr: np.ndarray, hands, lock: threading.Lock):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    with lock:
        results = hands.process(frame_rgb)

    if not results.multi_hand_landmarks:
        return None, 0

    features = []

    for hand_landmarks in results.multi_hand_landmarks[:MAX_HANDS]:
        features.extend(normalize_one_hand(hand_landmarks))

    while len(features) < FEATURE_LENGTH:
        features.append(0.0)

    features = np.array(features[:FEATURE_LENGTH], dtype=np.float32).reshape(1, -1)
    hands_detected = len(results.multi_hand_landmarks)

    return features, hands_detected


def predict_from_features(state: ServerState, features: np.ndarray):
    with torch.no_grad():
        tensor = torch.from_numpy(features).to(state.device, non_blocking=True)
        logits = state.model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)

    confidence_value = float(confidence.item())
    predicted_index = int(predicted.item())
    label = state.class_names[predicted_index]

    return label, confidence_value, predicted_index


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(HTML_PAGE)


@app.get("/health")
def health():
    if APP_STATE is None:
        return JSONResponse({"ok": False, "error": "server state not initialized"}, status_code=500)

    return {
        "ok": True,
        "model_run": APP_STATE.model_run_dir.name,
        "model_path": str(APP_STATE.model_path),
        "device": str(APP_STATE.device),
        "classes": APP_STATE.class_names,
    }


@app.post("/predict")
def predict(request: FrameRequest):
    if APP_STATE is None:
        return JSONResponse({"ok": False, "error": "server state not initialized"}, status_code=500)

    start = time.perf_counter()

    try:
        frame_bgr = decode_base64_image(request.image)
        features, hands_detected = extract_features_from_frame(
            frame_bgr,
            APP_STATE.hands,
            APP_STATE.mediapipe_lock,
        )

        if features is None:
            return {
                "ok": False,
                "error": "no hand detected",
                "hands_detected": 0,
                "elapsed_ms": (time.perf_counter() - start) * 1000,
            }

        label, confidence, predicted_index = predict_from_features(APP_STATE, features)

        return {
            "ok": True,
            "label": label,
            "predicted_index": predicted_index,
            "confidence": confidence,
            "hands_detected": hands_detected,
            "elapsed_ms": (time.perf_counter() - start) * 1000,
        }

    except Exception as exc:
        return JSONResponse(
            {
                "ok": False,
                "error": str(exc),
                "elapsed_ms": (time.perf_counter() - start) * 1000,
            },
            status_code=400,
        )


def parse_args():
    root = find_project_root()

    parser = argparse.ArgumentParser(description="Run ASL Hand browser webcam demo.")
    parser.add_argument("--runs_dir", type=Path, default=root / "runs")
    parser.add_argument("--model_run", type=Path, default=None)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)

    return parser.parse_args()


def main():
    global APP_STATE

    args = parse_args()
    model_run_dir = choose_model_run(args.runs_dir) if args.model_run is None else args.model_run.resolve()

    if not model_run_dir.exists():
        raise SystemExit(f"[错误] model run 不存在：{model_run_dir}")

    model_path = model_run_dir / "model_training" / "best_model.pth"
    device = resolve_device(args.device)

    print("=" * 80)
    print("ASL Hand Web Demo")
    print(f"Model run: {model_run_dir}")
    print(f"Model path: {model_path}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    model, class_names = load_model(model_path, device)
    hands = create_mediapipe_hands()

    APP_STATE = ServerState(
        model_run_dir=model_run_dir,
        model_path=model_path,
        device=device,
        model=model,
        class_names=class_names,
        hands=hands,
        mediapipe_lock=threading.Lock(),
    )

    import uvicorn

    print("=" * 80)
    print("Open this URL in your browser:")
    print(f"http://{args.host}:{args.port}")
    print("Press Ctrl+C in this terminal to stop the server.")
    print("=" * 80)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
