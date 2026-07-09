# ASL Hand Recognition Project Guidance

> Status: In Progress  
> Suggested location: `guidance/ASL_Project_Guidance_Merged.md`  
> Purpose: Merge the project README and the F ethics/limitations guidance into one simple, expandable project document

---

## 1. Project Overview

This project is a lightweight ASL alphabet hand gesture recognition system

It uses MediaPipe to extract hand landmark features, then trains machine learning models to classify isolated ASL alphabet letters

The current project should be described as:

```text
ASL static alphabet handshape classifier
```

not as:

```text
full ASL translator
```

because the current system recognizes isolated A-Z hand gestures only

---

## 2. Current Goal

The goal is to build a complete learning pipeline for AI-based gesture recognition:

```text
1. Dataset preparation
2. Hand landmark feature extraction
3. PyTorch MLP model training
4. Model evaluation
5. Realtime webcam inference
6. RandomForest model comparison
7. Local web demo
8. Ethics and limitations documentation
```

---

## 3. How the System Works

The model does not train directly on full image pixels

Workflow:

```text
Raw ASL images
→ MediaPipe hand landmark extraction
→ 126 numerical landmark features
→ CSV dataset
→ PyTorch MLP / RandomForest training
→ Model evaluation
→ Webcam or browser demo prediction
```

Feature format:

```text
2 hands × 21 landmarks × 3 coordinates = 126 features
```

This makes the project smaller, faster, and easier to debug than a full image-based CNN pipeline

---

## 4. Current Project Structure

```text
ASL Hand/
├── ai_guidance/
│   ├── copilot-instructions.md
│   └── git helper scripts
├── guidance/
│   ├── ASL_Project_Guidance_Merged.md
│   └── other tutorial / archive notes
├── z_codes/
│   ├── example/
│   ├── latest/
│   └── test/
│       ├── 1-prepare_dataset.py
│       ├── 2-train_model.py
│       ├── 3-evaluate_model.py
│       ├── 4-realtime_inference.py
│       ├── 5-compare_models.py
│       └── 6-web_demo.py
├── datasets/
├── runs/
├── README.md
└── .gitignore
```

Notes:

```text
datasets/
runs/
model weights
videos
zip files
```

are usually ignored by GitHub to keep the repository lightweight

---

## 5. Implemented Scripts

### 5.1 Dataset Preparation

File:

```text
z_codes/test/1-prepare_dataset.py
```

Purpose:

```text
- Read ASL alphabet images
- Extract MediaPipe hand landmarks
- Normalize landmark coordinates
- Save train.csv and val.csv
- Support sampling and mirror augmentation
```

Current feature output:

```text
126 landmark features
```

---

### 5.2 Model Training

File:

```text
z_codes/test/2-train_model.py
```

Current model:

```text
PyTorch MLP
126 input features
→ 256 hidden units
→ 128 hidden units
→ 26 output classes
```

Current successful result:

```text
Best validation accuracy: about 98.65%
Best epoch: about 76
Early stopping: about epoch 126
```

This is a strong validation result for the current sampled dataset, but it does not guarantee perfect webcam performance

---

### 5.3 Model Evaluation

File:

```text
z_codes/test/3-evaluate_model.py
```

Purpose:

```text
- Generate training curve
- Generate confusion matrix
- Save classification report
- Save wrong prediction analysis
- Save evaluation report
```

This supports tutorial extensions:

```text
A. Visualize training process
B. Confusion matrix analysis
```

---

### 5.4 Realtime Webcam Inference

File:

```text
z_codes/test/4-realtime_inference.py
```

Purpose:

```text
- Load best_model.pth
- Open webcam
- Extract hand landmarks
- Predict ASL letter in real time
```

Common controls:

```text
Q / q: quit
E / e: switch camera if supported
```

Important note:

```text
MediaPipe hand detection usually runs on CPU
PyTorch model inference can run on GPU
```

---

### 5.5 Model Comparison

File:

```text
z_codes/test/5-compare_models.py
```

Purpose:

```text
Compare PyTorch MLP with scikit-learn RandomForestClassifier
```

Why it matters:

```text
The landmark CSV is structured numerical data, so a traditional machine learning model can be a useful baseline
```

This supports tutorial extension:

```text
C. Compare another model
```

---

### 5.6 Local Web Demo

File:

```text
z_codes/test/6-web_demo.py
```

Purpose:

```text
Browser webcam
→ local FastAPI backend
→ MediaPipe landmark extraction
→ PyTorch prediction
→ result shown on webpage
```

Run:

```bash
python z_codes/test/6-web_demo.py
```

Open:

```text
http://127.0.0.1:8000
```

This supports tutorial extension:

```text
D. Web deployment / browser demo
```

---

## 6. Current Usage

Recommended run order:

```bash
python z_codes/test/1-prepare_dataset.py
python z_codes/test/2-train_model.py
python z_codes/test/3-evaluate_model.py
python z_codes/test/4-realtime_inference.py
python z_codes/test/5-compare_models.py
python z_codes/test/6-web_demo.py
```

If only testing the trained model:

```bash
python z_codes/test/4-realtime_inference.py
```

If testing browser mode:

```bash
python z_codes/test/6-web_demo.py
```

---

## 7. Current Strengths

```text
- Complete AI learning pipeline
- Uses landmark features instead of full image pixels
- Lightweight model
- Good validation accuracy on current dataset
- Easy to evaluate with confusion matrix and reports
- Supports realtime webcam testing
- Includes RandomForest baseline
- Includes local web demo
- GitHub repository can stay lightweight by ignoring large files
```

---

## 8. Current Limitations

The project currently does not recognize:

```text
- full ASL sentences
- continuous signing
- ASL grammar
- facial expressions
- body posture
- conversation meaning
- digits 0-9 unless a new 36-class dataset/model is added
```

Technical limitations:

```text
- MediaPipe failure affects the whole pipeline
- Lighting and camera angle can reduce accuracy
- Similar letters may be confused
- Validation accuracy may be higher than real webcam accuracy
- 5% dataset sampling is useful for speed but not enough for production
```

---

## 9. Ethics and Responsible Use

This project should be presented as an educational AI demo, not a replacement for ASL learning, Deaf culture, or human interpreters

Responsible wording:

```text
This project recognizes isolated ASL alphabet handshapes using MediaPipe landmarks and machine learning
It is not a complete sign language translation system
```

Avoid claiming:

```text
- full ASL translation
- human-level understanding
- medical, legal, or emergency reliability
- complete accessibility solution
```

Privacy notes:

```text
- Do not save webcam images unless necessary
- Tell users when camera is active
- Do not upload camera frames without consent
- Keep the current web demo local-only
```

Fairness testing should include:

```text
- different hand sizes
- different skin tones
- left and right hands
- different lighting
- different camera qualities
- different backgrounds
- different user mobility conditions
```

---

## 10. Future Work

### 10.1 Support ASL Digits

Future extension:

```text
A-Z + 0-9
```

This requires changing the model from:

```text
26 classes
```

to:

```text
36 classes
```

Needed updates:

```text
- dataset structure
- class list
- model output dimension
- training script
- evaluation script
- realtime inference script
- web demo script
```

Recommended structure:

```text
z_codes/test_36_classes/
```

---

### 10.2 Improve Realtime Stability

Possible improvements:

```text
- prediction smoothing
- confidence threshold
- better camera selection
- better lighting guide
- warning when no hand is detected
```

---

### 10.3 Improve Dataset and Evaluation

Possible improvements:

```text
- compare 5%, 10%, 20%, and full dataset
- test multiple users
- test different backgrounds
- test left/right hands separately
- save more detailed error cases
```

---

### 10.4 Improve Deployment

Possible improvements:

```text
- better web UI
- local-only privacy notice
- package requirements.txt
- one-click setup script
- lightweight device testing
```

---

## 11. GitHub / Archive Notes

Large files should usually not be uploaded to GitHub:

```text
datasets/
runs/
*.pth
*.pt
*.onnx
*.mp4
*.zip
```

Recommended GitHub content:

```text
README.md
guidance/
ai_guidance/
z_codes/
requirements.txt
.gitignore
```

Recommended external storage for large files:

```text
Google Drive
OneDrive
GitHub Release
Git LFS
```

---

## 12. Tutorial Extension Status

```text
A. Visualize training process
Status: Implemented in 3-evaluate_model.py

B. Confusion matrix analysis
Status: Implemented in 3-evaluate_model.py

C. Compare RandomForest
Status: Implemented in 5-compare_models.py

D. Web deployment
Status: Implemented as local FastAPI demo in 6-web_demo.py

E. Support digits 0-9
Status: Future work; requires 36-class dataset and model pipeline

F. Ethics and limitations
Status: Integrated into this guidance document
```

---

## 13. Short Project Description

This project is an in-progress ASL alphabet recognition system. It uses MediaPipe to convert hand images into 126 landmark features, then trains machine learning models such as a PyTorch MLP and RandomForest to classify isolated A-Z hand gestures. The project includes dataset preparation, training, evaluation, realtime webcam inference, model comparison, and a local web demo. It is useful as an educational AI pipeline, but it should not be described as a full ASL translation system

---

## 14. Update Log

```text
2026-07: Merged README and F ethics guidance into one project guidance document
```
