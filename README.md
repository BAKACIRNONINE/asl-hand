# ASL Hand Gesture Recognition

A lightweight ASL alphabet recognition project that uses hand landmark features to classify American Sign Language letters from images, webcam input, or a local web demo.

This project is designed as a complete learning pipeline: prepare data, extract hand landmarks, train a model, evaluate predictions, compare models, and test real-time recognition.

## Features

- ASL alphabet gesture classification
- Hand landmark based feature extraction
- PyTorch MLP training pipeline
- Random Forest baseline comparison
- Model evaluation with accuracy, classification report, confusion matrix, and wrong prediction analysis
- Real-time webcam inference
- Local web demo for browser-based testing
- Large files excluded from GitHub to keep the repository lightweight

## How It Works

The project does not train directly on full image pixels. Instead, it extracts hand landmark coordinates from each ASL gesture image and converts them into numerical features. These features are then used to train classification models.

Basic workflow:

```text
ASL image dataset
        ↓
Hand landmark extraction
        ↓
Landmark CSV dataset
        ↓
Model training
        ↓
Model evaluation
        ↓
Real-time / web prediction
```

This makes the model smaller and easier to train than a full image-based deep learning model.

## Project Structure

```text
ASL Hand/
  ai_guidance/
    ASL_Tutorial_CN.md
    ASL_Tutorial_EN.md
    copilot-instructions.md
    F_ethics_limitations.md

  z_codes/
    test/
      1-prepare_dataset.py
      2-train_model.py
      3-evaluate_model.py
      4-realtime_inference.py
      5-compare_models.py
      6-web_demo.py

  datasets/
    csv_landmark_dataset/

  runs/
    model_training/
    model_evaluation/
```

Some folders may not appear on GitHub because large datasets, model weights, training runs, and media files are ignored by default.

## Installation

Recommended Python version:

```bash
python 3.10
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available yet, install the main packages manually:

```bash
pip install pandas numpy scikit-learn torch opencv-python mediapipe matplotlib fastapi uvicorn
```

## Usage

### 1. Prepare the Dataset

Organize the ASL image dataset and prepare it for feature extraction.

```bash
python z_codes/test/1-prepare_dataset.py
```

### 2. Train the Model

Train the PyTorch MLP model using extracted landmark features.

```bash
python z_codes/test/2-train_model.py
```

### 3. Evaluate the Model

Generate evaluation results such as classification report, confusion matrix, training curve, and wrong prediction analysis.

```bash
python z_codes/test/3-evaluate_model.py
```

### 4. Run Real-Time Inference

Use a webcam to recognize ASL letters in real time.

```bash
python z_codes/test/4-realtime_inference.py
```

Common controls:

```text
Q / q: quit camera window
E / e: switch camera if supported
```

### 5. Compare Models

Compare the PyTorch MLP model with a Random Forest baseline.

```bash
python z_codes/test/5-compare_models.py
```

### 6. Run the Web Demo

Start the local web demo.

```bash
python z_codes/test/6-web_demo.py
```

Then open:

```text
http://127.0.0.1:8000
```

## Strengths

- Lightweight compared with full image-based models
- Uses structured landmark features instead of raw pixels
- Easier to debug and analyze than a black-box image model
- Good for learning the full machine learning workflow
- Includes both neural network and traditional machine learning comparison
- Supports real-time webcam testing
- Keeps GitHub clean by ignoring large generated files

## Limitations

- Focuses on ASL alphabet letters, not full ASL translation
- Accuracy depends on hand detection quality
- Lighting, camera angle, background, and hand position can affect predictions
- Similar ASL letters may still be confused
- The local web demo is for testing and learning, not production use
- The model may not generalize well to every signer without more diverse data

## Suggested Improvements

- Add more training samples for commonly confused letters
- Test with different hands, lighting conditions, backgrounds, and camera angles
- Improve landmark normalization
- Add better left-hand and right-hand handling
- Add confidence display and prediction smoothing for real-time use
- Improve the web interface
- Test deployment on lightweight devices

## Data and Model Files

Large files are intentionally not stored in this repository. This includes:

- Raw datasets
- Processed datasets
- Training outputs
- Model weights
- Checkpoints
- Video files
- Compressed archives

Recommended local folders:

```text
datasets/
runs/
weights/
checkpoints/
```

## Notes

This project is mainly for learning, experimentation, and demonstrating a complete ASL recognition workflow. It should not be used as a real accessibility tool without more testing, more diverse data, and careful evaluation.
