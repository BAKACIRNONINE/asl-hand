# F. Ethics and Limitations Guidance  
# ASL Hand Recognition Project Archive Note

> Suggested location: `guidance/F_ethics_limitations_guidance.md`  
> Purpose: This file documents the ethical discussion, project limitations, and future-work reasoning for the ASL Hand Recognition project. It works like an archive README for the “F” extension section.

---

## 1. What this file is for

This file is not a runnable program.  
It is a project guidance and archive document for the ASL Hand Recognition system.

The main code pipeline already covers:

```text
1-prepare_dataset.py
2-train_model.py
3-evaluate_model.py
4-realtime_inference.py
5-compare_models.py
6-web_demo.py
```

The “F” section is different from C/D/E because it is not mainly about adding another model or feature.  
Instead, it explains what the project can do, what it cannot do, and what ethical risks should be considered before using the model in real situations.

This document can be used as:

```text
- a README-style archive note
- a guidance file for future development
- a written discussion section for the tutorial
- supporting documentation for evaluation reports
```

---

## 2. Current project summary

The current ASL Hand Recognition project uses a landmark-based machine learning pipeline.

The overall workflow is:

```text
Raw ASL hand images
→ MediaPipe hand landmark extraction
→ 126 numerical landmark features
→ CSV dataset
→ PyTorch MLP model training
→ model evaluation
→ realtime webcam inference
→ optional model comparison
→ optional local web demo
```

The landmark feature format is:

```text
2 hands × 21 landmarks × 3 coordinates = 126 features
```

The project focuses on recognizing isolated ASL alphabet gestures:

```text
A-Z
26 classes
```

The current model does not recognize:

```text
- full sign language sentences
- continuous hand motion
- facial expressions
- body posture
- grammar
- meaning in conversation
- ASL digits 0-9 unless a separate dataset and 36-class model are added
```

---

## 3. What has already been implemented

### 3.1 Dataset preparation

Implemented in:

```text
z_codes/test/1-prepare_dataset.py
```

This script:

```text
- reads the ASL alphabet image dataset
- samples a selected percentage of images
- uses MediaPipe Hands to extract hand landmarks
- normalizes each hand by subtracting the wrist landmark
- pads missing hands with zeros
- applies mirror augmentation
- saves train.csv and val.csv
```

The script uses legacy MediaPipe:

```python
mp.solutions.hands
```

Therefore, the environment is intentionally fixed around:

```text
mediapipe==0.10.21
numpy==1.26.4
opencv-contrib-python==4.11.0.86
```

---

### 3.2 Model training

Implemented in:

```text
z_codes/test/2-train_model.py
```

The model is a small PyTorch MLP:

```text
126 input features
→ 256 hidden units
→ 128 hidden units
→ 26 output classes
```

The current training workflow supports:

```text
- CUDA GPU by default
- early stopping
- best_model.pth
- last_model.pth
- training_log.csv
- training_report.md
- training_metrics.json
```

The current successful run achieved about:

```text
Best validation accuracy: 98.65%
Best epoch: 76
Early stopping: epoch 126
```

This result shows that MediaPipe landmarks are strong features for isolated ASL alphabet recognition in the sampled dataset.

---

### 3.3 Evaluation

Implemented in:

```text
z_codes/test/3-evaluate_model.py
```

This script generates:

```text
- training_curve.png
- confusion_matrix.png
- classification_report.csv
- wrong_predictions.csv
- top_confusions.csv
- evaluation_metrics.json
- evaluation_report.md
```

This covers the tutorial extension A and B:

```text
A. Visualizing the training process
B. Confusion matrix analysis
```

---

### 3.4 Realtime inference

Implemented in:

```text
z_codes/test/4-realtime_inference.py
```

The realtime script:

```text
- loads best_model.pth
- opens the webcam
- defaults to camera index 0
- scans available cameras if index 0 fails
- allows user camera selection
- supports q or Q to quit
- uses MediaPipe for hand landmarks
- uses PyTorch MLP for prediction
```

Important note:

```text
MediaPipe hand detection runs on CPU.
PyTorch MLP prediction runs on GPU by default.
```

This is expected because legacy `mp.solutions.hands` does not directly use the same CUDA pipeline as PyTorch.

---

### 3.5 Model comparison

Implemented in:

```text
z_codes/test/5-compare_models.py
```

This covers tutorial extension C:

```text
C. Compare a different model such as RandomForest
```

It compares:

```text
- PyTorch MLP
- scikit-learn RandomForestClassifier
```

Both models use the same CSV landmark features.

The purpose is to test whether a traditional machine learning model can perform competitively on the same structured landmark data.

---

### 3.6 Web demo

Implemented in:

```text
z_codes/test/6-web_demo.py
```

This covers tutorial extension D:

```text
D. Web deployment / browser demo
```

It creates a local FastAPI web demo:

```text
Browser webcam
→ frame sent to local backend
→ MediaPipe landmark extraction
→ PyTorch MLP prediction
→ result shown on webpage
```

This is a local demonstration, not production deployment.

---

## 4. Why F is not a normal program

Extensions C and D are software features:

```text
C = add another model comparison script
D = add a web demo
```

Extension E would also be a software/data feature:

```text
E = expand the dataset and model from A-Z to A-Z + 0-9
```

But F is different:

```text
F = ethics, limitations, fairness, accessibility, and real-world responsibility
```

Therefore, F should be archived as a guidance document instead of being treated as another runnable Python file.

A good location is:

```text
ASL Hand/guidance/F_ethics_limitations_guidance.md
```

This keeps the code folder clean while preserving the reasoning behind the project.

---

## 5. Technical limitations

### 5.1 This is not full sign language translation

The current system only recognizes isolated ASL alphabet gestures.

It does not understand:

```text
- continuous signing
- ASL grammar
- facial expressions
- body motion
- sentence-level meaning
- conversational context
```

This is important because sign language is a full natural language, not just a set of hand shapes.

Calling this project a “sign language translator” would be inaccurate.  
A more accurate description is:

```text
ASL alphabet handshape classifier
```

or:

```text
ASL static alphabet recognition demo
```

---

### 5.2 The model depends on MediaPipe

The PyTorch model does not directly see the original image.  
It only sees landmark numbers generated by MediaPipe.

So the system depends on two stages:

```text
Stage 1: MediaPipe detects hand landmarks
Stage 2: PyTorch model classifies landmarks
```

If MediaPipe fails, the classifier receives poor or missing input.

Common failure cases include:

```text
- low light
- motion blur
- hand partially outside the frame
- hand occlusion
- unusual camera angles
- complex backgrounds
- multiple hands overlapping
- very fast motion
```

This means the reported validation accuracy does not fully represent real webcam performance.

---

### 5.3 Dataset sampling limitation

The project uses a sampling ratio such as:

```text
0.05 = 5%
```

This is useful for:

```text
- fast experiments
- classroom demonstration
- debugging
- quick model iteration
```

However, 5% sampling should not be interpreted as enough for production use.

A stronger experiment should compare:

```text
0.05
0.10
0.20
full dataset
```

and measure:

```text
- validation accuracy
- confusion matrix
- realtime webcam stability
- error types
- training time
- model size
```

---

### 5.4 Accuracy does not mean real-world reliability

A validation accuracy around 98% is strong, but it does not guarantee robust real-world use.

Possible reasons:

```text
- validation images may be similar to training images
- dataset background may be controlled
- gestures may be centered and clear
- lighting may be cleaner than real webcam conditions
- user hand shapes may differ from dataset examples
```

Therefore, webcam testing is necessary even when CSV validation accuracy is high.

---

### 5.5 Static letters are easier than real signing

The ASL alphabet dataset uses isolated static letters.

Real signing often includes:

```text
- movement
- timing
- rhythm
- facial expression
- grammar
- spatial reference
- body posture
```

This project should be viewed as an entry-level recognition system, not a complete accessibility solution.

---

## 6. Ethical considerations

### 6.1 Respect Deaf culture and signed languages

ASL is a real language with its own grammar, culture, and community.

A technical project should avoid reducing ASL to only:

```text
hand pose classification
```

A more respectful framing is:

```text
This project is a technical demonstration for recognizing isolated ASL alphabet handshapes.
It is not a replacement for ASL learning, Deaf culture, or human interpreters.
```

---

### 6.2 Avoid overclaiming

The project should not claim:

```text
- full ASL translation
- complete accessibility solution
- human-level understanding
- reliable use in medical, legal, or emergency contexts
```

The project can claim:

```text
- isolated ASL alphabet recognition
- MediaPipe landmark-based classification
- educational AI pipeline demonstration
- local realtime webcam demo
```

---

### 6.3 Fairness and diversity

Even though landmark features reduce dependence on raw image color and background, fairness still matters.

The system should be tested across:

```text
- different skin tones
- different hand sizes
- different hand shapes
- left and right hands
- different camera qualities
- different lighting conditions
- different mobility conditions
```

Without this testing, it is not possible to know whether the model works equally well for everyone.

---

### 6.4 Privacy

Camera-based systems process visual information.

Even if this project only uses local webcam frames, privacy should still be considered.

For local use:

```text
- avoid saving webcam images unless necessary
- clearly tell users when the camera is active
- do not upload video frames to remote servers without consent
- avoid collecting personal biometric data
```

The current local web demo should be described as:

```text
local-only demonstration
```

not as a cloud service.

---

### 6.5 Accessibility responsibility

Assistive technology should support users, not create new barriers.

Potential positive uses:

```text
- education
- ASL alphabet practice
- human-computer interaction
- accessible input method prototype
```

Potential risks:

```text
- inaccurate predictions causing misunderstanding
- users relying on the system in serious communication
- oversimplifying sign language
- ignoring Deaf community feedback
```

A responsible future version should involve feedback from people who actually use ASL.

---

## 7. Extension E as future work

Extension E is:

```text
Support ASL digits 0-9
```

This is not just a small label change.

The current model uses:

```text
26 classes: A-Z
```

To support digits, the project would need:

```text
36 classes: A-Z + 0-9
```

That requires updates to:

```text
- dataset structure
- class name list
- model output dimension
- training script
- evaluation script
- realtime inference script
- web demo script
```

Suggested future structure:

```text
z_codes/test_36_classes/
```

This avoids breaking the stable A-Z pipeline.

Possible future class list:

```python
CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
]
```

---

## 8. Recommended wording for final report

A concise final-project statement could be:

> This project implements a MediaPipe landmark-based ASL alphabet recognition system. It recognizes isolated A-Z hand gestures using 126 normalized hand landmark features and a lightweight PyTorch MLP classifier. The system includes dataset generation, training, evaluation, realtime webcam inference, model comparison, and a local web demo. However, it should be understood as an educational static alphabet classifier rather than a complete sign language translation system. Real-world deployment would require more diverse data, fairness testing, privacy review, and feedback from ASL users and Deaf communities.

---

## 9. Recommended project archive structure

Suggested final archive:

```text
ASL Hand/
├── datasets/
├── runs/
│   └── model_002_sample0.05/
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
├── guidance/
│   ├── ASL_Tutorial_CN.md
│   ├── ASL_Tutorial_EN.md
│   ├── PROJECT_STRUCTURE.md
│   └── F_ethics_limitations_guidance.md
└── README.md
```

---

## 10. Final status of tutorial extensions

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
Status: Documented in this guidance file
```

---

## 11. Short Chinese summary

本项目目前已经完成了 ASL 字母识别的主流程：数据准备、训练、评估、实时摄像头识别、模型对比和本地 Web Demo。

F 部分不适合继续写成一个新的 Python 功能，因为它本质上是项目反思和风险分析。更合适的做法是把它放在 `guidance/` 文件夹中，作为一个类似 README 的归档说明文件。

这个文件的作用是说明：

```text
- 当前模型实际能做什么
- 当前模型不能做什么
- 为什么它不是完整手语翻译器
- 数据采样和验证准确率有什么局限
- MediaPipe 检测失败会怎样影响结果
- 公平性、隐私和无障碍技术责任
- E 部分为什么应该作为未来工作
```

这样项目结构更清楚，也更符合教程中 F 部分“Ethics and Limitations Discussion”的性质。
