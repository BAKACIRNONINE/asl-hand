# AI-Assisted Programming: Building an ASL Recognition System from Scratch

---

## High-Level Overview

This project takes you through the complete lifecycle of a real AI vision recognition system — from data preparation and model training to real-time camera inference. The goal: **recognize the 26 letters of the American Sign Language (ASL) alphabet** and display the predicted letter live through a webcam feed.

The system consists of three modules:

| Module | File | Purpose |
|--------|------|---------|
| Data Preparation | `prepare_dataset.py` | Extract hand landmarks with MediaPipe and produce a CSV feature file |
| Model Training | `train_model.py` | Read the CSV and train a fully connected neural network with PyTorch |
| Real-Time Inference | `realtime_inference.py` | Load the trained model and recognize gestures via webcam in real time |

**Why is this architecture worth studying?**

Traditional image classification AI requires massive training on raw pixels, which is computationally expensive. This project introduces a critical "middle layer" technology: **MediaPipe**. It translates each hand gesture image into the 3D coordinates of 21 hand landmarks (126 floating-point values total), dramatically compressing the input dimensionality. The neural network then only needs to learn from these lightweight landmark features rather than millions of raw pixels — delivering enormous gains in both training speed and accuracy.

This embodies a principle you will encounter throughout real-world AI engineering: **use feature engineering to reduce problem complexity, rather than brute-forcing raw data with larger models.**

---

## Part 1: Preparing the Dataset

### 1.1 Dataset Source

This project uses the publicly available ASL Alphabet image dataset on Kaggle:

> **Dataset URL:**  
> https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset?resource=download-directory

The dataset contains images of hand gestures for all 26 English letters, organized by class folder (A, B, C…Z), with hundreds of images per class.

### 1.2 Download and Directory Structure

After downloading and extracting, place the training dataset at the following path (must match the path used in the code):

```
project_root/
├── ASL_Alphabet_Dataset/
│   └── asl_alphabet_train/
│       ├── A/
│       │   ├── A1.jpg
│       │   ├── A2.jpg
│       │   └── ...
│       ├── B/
│       └── ... (26 folders total)
├── 1-Prepare_dataset_V4.1.py
├── 2-Train_NN_v4.1.py
└── 3-NN_realtime_test_V4.1.py
```

### 1.3 Sampling Strategy

The code sets a sampling ratio (`sampling_ratio = 0.05`), meaning only 5% of images are randomly selected from each class.  
This design choice is deliberate: the full dataset is very large and processing all of it would take too long, yet 5% sampling is already sufficient to train a high-accuracy model.  
Try adjusting this parameter (e.g., to `0.1` or `0.2`) and observe how it affects training results.

---

## Part 2: Setting Up the Project Environment

### 2.1 Prerequisites

- Python 3.8 or later (3.10 recommended)
- pip installed
- A virtual environment is strongly recommended (venv or conda)

### 2.2 Dependency List

All Python packages required for this project:

```
opencv-python
mediapipe
torch
torchvision
pandas
scikit-learn
tqdm
```

### 2.3 One-Click Install Script

Create a `requirements.txt` file with the following contents:

```txt
opencv-python>=4.8.0
mediapipe>=0.10.0
torch>=2.0.0
torchvision>=0.15.0
pandas>=2.0.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

Then run the following commands in your terminal:

```bash
# (Recommended) Create a virtual environment first
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows

# Install all dependencies
pip install -r requirements.txt
```

### 2.4 Verify Installation

Run the following command to confirm all packages are correctly installed:

```bash
python -c "import cv2; import mediapipe; import torch; print('All packages OK')"
```

If you see `All packages OK`, your environment is ready.

---

## Part 3: AI Prompts

This project is designed to be built using VS Code + GitHub Copilot (or any other AI coding assistant). The following prompts are split into three stages, each providing detailed context to help the AI understand the design intent.

---

### Prompt 1: Data Preparation Module

**Purpose:** Guide the AI to write a script that batch-processes raw images into a structured CSV of hand landmark features.

---

```
# [CONTEXT]
I am building an ASL (American Sign Language) alphabet recognition system.
The dataset is from Kaggle: folders named A–Z, each containing hundreds of hand gesture images.

# [GOAL]
Write a Python script that:
1. Iterates through each class folder (A–Z) in the dataset directory.
2. For each image, uses MediaPipe Hands to detect hand landmarks.
3. Normalizes all 21 landmarks by subtracting the position of landmark[0] (the wrist),
   so that the data is translation-invariant.
4. Supports up to 2 hands; if fewer are detected, pads the remaining data with zeros.
5. Adds a horizontally flipped version of each valid sample (mirror augmentation).
6. Randomly samples only `sampling_ratio` (e.g. 5%) of images per class to keep the
   dataset manageable.
7. Uses Python multiprocessing to parallelize processing across class folders.
8. Writes all results to a CSV file with columns:
   [Image Path, Class Index, Hand1_Landmark0_x, Hand1_Landmark0_y, Hand1_Landmark0_z, ..., Hand2_Landmark20_z]

# [WHY MEDIAPIPE]
MediaPipe reduces the raw image (200×200 pixels = 40,000 values) to just 126 floating-point
numbers (2 hands × 21 landmarks × 3 coordinates). This collapses the input space
dramatically, making the downstream neural network far simpler and faster to train.

# [WHY A CSV FILE]
Decoupling feature extraction from model training is good engineering practice.
The CSV can be reused for different models without re-running the expensive MediaPipe pass.

# [CONSTRAINTS]
- Use static_image_mode=True and min_detection_confidence=0.9 for MediaPipe Hands.
- Skip images where no landmarks are detected (save them to a temp folder for inspection if desired).
- Use the folder name as the class label; map A→0, B→1, …, Z→25.
- Use tqdm to show progress.
```

---

### Prompt 2: Model Training Module

**Purpose:** Guide the AI to read the CSV and train a fully connected neural network classifier.

---

```
# [CONTEXT]
I have a CSV file (`dataset_info.csv`) generated by MediaPipe hand landmark extraction.
Each row: [image_path, class_index (0–25), 126 landmark features (floats)]

# [GOAL]
Write a PyTorch training script that:
1. Reads the CSV, extracts feature columns and class labels.
2. Splits data into 80% train / 20% test (random_state=42).
3. Defines a 3-layer fully connected network:
   Input(126) → Linear(256) + BatchNorm + Dropout(0.3) → ReLU
   → Linear(128) + BatchNorm + Dropout(0.3) → ReLU
   → Linear(26) [output logits]
4. Trains with:
   - Loss: CrossEntropyLoss
   - Optimizer: Adam (lr=0.001)
   - LR scheduler: ReduceLROnPlateau (monitor val accuracy, patience=10, factor=0.1)
   - Gradient accumulation: every 4 steps
   - Early stopping: stop if val accuracy does not improve for 50 epochs
5. Saves the best model (highest val accuracy) to `best_model.pth`.
6. Saves the final model to `last_model.pth`.
7. Prints epoch loss, val accuracy, and training time per epoch.

# [WHY THIS ARCHITECTURE]
The input is only 126 numbers, so a simple 3-layer MLP is sufficient.
BatchNorm stabilizes training; Dropout prevents overfitting on the small dataset.
ReduceLROnPlateau + early stopping prevents wasted compute.

# [CONSTRAINTS]
- Device: CPU (no GPU required for this model size).
- Gradient accumulation allows effective batch size scaling without extra memory.
- Max 5000 epochs, but early stopping should trigger well before that.
```

---

### Prompt 3: Real-Time Inference Module

**Purpose:** Guide the AI to write a real-time webcam gesture recognition application.

---

```
# [CONTEXT]
I have a trained PyTorch model (`best_model.pth`) that classifies ASL hand gestures
from MediaPipe landmark features (126 floats).

# [GOAL]
Write a Python script that:
1. Opens the webcam (camera index 1), sets resolution to 320×240, target FPS 60.
2. On each frame:
   a. Run MediaPipe Hands (static_image_mode=False, max_num_hands=2,
      min_detection_confidence=0.6, min_tracking_confidence=0.7).
   b. Extract landmarks using the same normalization as training
      (subtract landmark[0] from all others; pad with zeros if < 2 hands).
   c. Run the model in a background thread (to avoid blocking the main camera loop).
   d. Draw hand landmarks on the frame with MediaPipe drawing utilities.
   e. Overlay the predicted letter on the frame (green text, top-left).
3. Quit on pressing 'q'.

# [IMPORTANT — CONSISTENCY]
The landmark extraction logic (normalization + padding) must be IDENTICAL to the
data preparation script. Any mismatch will cause silent accuracy degradation.

# [WHY THREADING]
Running model inference in a separate thread allows the camera loop to run smoothly
at high FPS without stuttering, improving the user experience.

# [CONSTRAINTS]
- Load model with weights_only=False (PyTorch full model pickle).
- Class names: ['A', 'B', ..., 'Z'] (26 letters, index 0–25).
- Keep the main loop non-blocking; show the last predicted label until updated.
```

---

## Part 4: Using the Prompts — How to Observe AI Coding and Ensure Quality

When using GitHub Copilot (or any other AI coding tool), pay close attention to the following points:

### 4.1 How to Enter Prompts

Using GitHub Copilot Chat in VS Code (sidebar panel):
1. Create a new empty `.py` file.
2. Open Copilot Chat (shortcut `Ctrl+Alt+I` / `Cmd+Alt+I`).
3. Paste one of the prompts above into the chat box and send it.
4. Copilot will generate a code draft; click "Insert into Editor" to insert it.
5. For parts you are not satisfied with, ask follow-up questions, e.g.: `"Modify the landmark extraction to also handle the case where only 1 hand is detected."`

### 4.2 Observing AI Output — Key Items to Check

| Check Item | Explanation |
|------------|-------------|
| **Normalization logic consistency** | The landmark processing in data preparation and real-time inference **must be identical**; otherwise the input distribution will shift, causing recognition failure. |
| **CSV columns match model input** | Verify that the number of feature columns read during training (`iloc[:, 2:]`) equals the model input layer size (126). |
| **Multiprocessing guard** | The data preparation script must launch multiprocessing inside an `if __name__ == '__main__':` block, otherwise it will crash on Windows. |
| **Device consistency** | Both training and inference should use `device = torch.device("cpu")` — no GPU needed — ensuring cross-platform compatibility. |
| **Camera index** | The script uses `cv2.VideoCapture(1)` (index 1, external camera); if you only have a built-in camera, change it to `0`. |

### 4.3 Common AI-Generated Errors and How to Correct Them

**Issue 1: AI forgets padding (zero-fill to 126)**
- Symptom: When only 1 hand is detected, the feature vector has only 63 values, causing a model dimension error.
- Corrective prompt: `"Add a while loop after landmark extraction to pad the list with zeros until its length reaches 126."`

**Issue 2: AI uses absolute coordinates without normalization**
- Symptom: Model accuracy is very low because coordinate values vary with hand position in the frame.
- Corrective prompt: `"Subtract landmark[0]'s x, y, z from all landmarks before appending to the list."`

**Issue 3: AI forgets `model.eval()` mode**
- Symptom: BatchNorm/Dropout remain in training mode during inference, causing unstable predictions.
- Corrective prompt: `"Call model.eval() before inference, and wrap it with torch.no_grad() to disable gradient computation."`

### 4.4 Debugging and Observation Tips

- **Print intermediate values**: During training, print the val accuracy every 10 epochs to observe the convergence curve.
- **Save the best model**: The code automatically saves `best_model.pth`. Use this for inference after training, not `last_model.pth`.
- **Adjust the sampling ratio**: Increase `sampling_ratio` from 0.05 to 0.1 and observe how doubling the data affects accuracy (typically improves by 2–5%).
- **Watch the early stopping trigger point**: If early stopping fires before epoch 100, the model may be underfitting — try increasing network width.

---

## Part 5: Architectural Strengths

### 5.1 MediaPipe as Feature Extractor — The Critical Dimensionality Reduction

| Comparison | Direct Image Training | MediaPipe Landmark Approach |
|------------|----------------------|-----------------------------|
| Input dimension | ~40,000 (200×200 image) | 126 (landmark coordinates) |
| Model parameters | Millions | ~60,000 |
| Training time | Hours (GPU required) | Minutes (CPU only) |
| Data requirement | Tens of thousands of images | Hundreds suffice |
| Generalization | Sensitive to background/lighting | Background-agnostic |

### 5.2 Normalization Design — Translation Invariance

Subtracting the wrist (landmark[0]) coordinates from all landmarks ensures the hand can be recognized consistently regardless of where it appears in the frame. This is a simple yet profoundly effective feature engineering technique.

### 5.3 Mirror Data Augmentation — Low-Cost Data Expansion

For every valid image, an additional version with x-coordinates negated is generated — equivalent to a "left/right hand mirror." This doubles the dataset size at virtually zero cost while also improving recognition for left-handed users.

### 5.4 Decoupled Architecture — CSV as the Middle Layer

Separating feature extraction (Script 1) from model training (Script 2) means:
- You can train multiple model architectures without re-running the expensive MediaPipe pass.
- The CSV file can be used with other ML frameworks (scikit-learn, XGBoost, etc.) for comparative experiments.

### 5.5 Production-Ready Engineering Habits

- **Early stopping + best model checkpointing**: Prevents overfitting and preserves the optimal state.
- **Adaptive learning rate scheduling**: ReduceLROnPlateau automatically lowers the learning rate when validation accuracy plateaus.
- **Gradient accumulation**: Simulates a larger effective batch size without increasing memory consumption.
- **Multi-threaded inference**: In the real-time application, inference runs on a separate thread so it never blocks the camera capture main loop.

---

## Supplementary Materials (Enriching the Learning Experience)

The following modules can be added as extensions to make the tutorial more comprehensive and engaging:

### A. Visualize the Training Process
Have students use `matplotlib` to plot a `val accuracy vs epoch` curve, providing an intuitive view of the model's convergence.

```python
# Add at the end of the training loop:
import matplotlib.pyplot as plt
plt.plot(val_accuracies)
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy (%)')
plt.title('Training Progress')
plt.savefig('training_curve.png')
```

### B. Confusion Matrix Analysis
After training, generate a confusion matrix to identify the most commonly confused letter pairs (e.g., M/N, U/V), helping students understand the model's weaknesses.

### C. Compare Different Models
Using the same CSV, train both:
- The MLP from this project
- scikit-learn's RandomForest (`from sklearn.ensemble import RandomForestClassifier`)
- Compare their accuracy and training time to help students appreciate the importance of choosing the "right tool."

### D. Deploy to the Web (Advanced)
Wrap the model as a REST API using Flask or FastAPI, pair it with a browser-based camera interface, and experience the full pipeline from model to application.

### E. Challenge Task: Support Digits 0–9
ASL digit datasets are also available on Kaggle. Encourage students to modify the code to expand from 26 to 36 classes and adjust the model output layer accordingly.

### F. Ethics and Limitations Discussion
Guide students to think about:
- Why is 5% sampling enough? More data is not always better.
- How well does this model generalize across different skin tones and lighting conditions?
- What practical impact does sign language recognition have for the Deaf and hard-of-hearing community?

---

## Summary

This project demonstrates an elegant AI engineering pipeline:

```
Raw Images → MediaPipe Landmark Extraction → CSV → PyTorch MLP Training → Real-Time Webcam Inference
```

Its core philosophy: **replace end-to-end pixel learning with structured priors (hand skeletal landmarks)**, achieving high-accuracy gesture recognition at extremely low computational cost. This is a textbook example of the "lightweight AI" design philosophy widely adopted in industry.

---

*Tutorial — Final English Version*
