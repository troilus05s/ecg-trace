# ECG Anomaly Detection
**Time-Series ML Project | CNN + BiLSTM + Attention | MIT-BIH Arrhythmia Database**

A medical-grade time-series analysis system designed to detect cardiac arrhythmias in real-time. This project uses a hybrid deep learning architecture to capture both local waveform shapes (morphology) and long-term rhythm patterns. It includes an interactive Streamlit dashboard featuring Explainable AI (Grad-CAM) to visually justify the model's clinical predictions.

---

## Key Results (Test Set)

Our model was evaluated using a strict **patient-level split** to prevent data leakage and ensure true clinical validity. Using a highly penalized loss function to prioritize catching anomalies, the model achieved:

- **ROC-AUC:** `0.9267`
- **Precision:** `85.69%`
- **Recall (Sensitivity):** `73.29%`  
  *(Note: Effective recall can be further optimized via the threshold slider in the web app)*

---

## Project Structure

```text
ecg_anomaly/
├── data_pipeline.py      # Data download, windowing, SMOTE balancing
├── train_evaluate.py     # Model definition, training loop, evaluation, Grad-CAM
├── app.py                # Streamlit web UI
├── requirements.txt
├── data/                 # Created by data_pipeline.py
│   ├── X_train.npy
│   ├── y_train.npy
│   ├── X_val.npy
│   ├── y_val.npy
│   ├── X_test.npy
│   └── y_test.npy
└── models/               # Created by train_evaluate.py
    ├── best_model.pth
    ├── training_curves.png
    ├── confusion_matrix.png
    └── roc_curve.png
```

---

## Setup

```bash
# 1. Create virtual environment (recommended)
python -m venv ecg_env
source ecg_env/bin/activate        # Windows: ecg_env\Scripts\activate

# 2. Install dependencies
pip install -r Requirements.txt
```

---

## How to Run (Two Options)

You can run this project in two ways: jump straight to the interactive web app using a pre-trained model, or build the entire pipeline from scratch.

### Option A: Quick Start (Use Pre-Trained Model)

Best if you want to test the UI, live streaming, and Explainable AI features immediately.

1. Go to the Releases tab on this GitHub repository.
2. Download the `best_model.pth` file from the latest release.
3. Create a folder named `models` inside the main project directory.
4. Place the downloaded `best_model.pth` into the `models/` folder.
5. Launch the Streamlit app:

```bash
streamlit run app.py
```

6. Open your browser to `http://localhost:8501`.

---

### Option B: End-to-End (Train from Scratch)

Best if you want to reproduce the data processing, train the model, and generate evaluation metrics.

#### Step 1 — Build the dataset

Downloads MIT-BIH directly from PhysioNet. No account needed.

```bash
# Full dataset (all 48 records, ~10–15 min download)
python data_pipeline.py

# Quick test run (12 records, ~2 min)
python data_pipeline.py --quick
```

Output: `data/` folder with train/val/test `.npy` files + class distribution chart.

---

#### Step 2 — Train the model

```bash
# Optimal settings for high Recall (Pos-Weight 4.0, LR 5e-4)
python train_evaluate.py --epochs 40 --batch 128 --lr 5e-4 --pos-weight 4.0
```

Note on `--pos-weight`: This determines how much extra you penalize the model for missing a real arrhythmia. Higher = more sensitive model, more false alarms.

Output: `models/` folder with checkpoint + training curve plots.

---

#### Step 3 — Launch the web app

```bash
streamlit run app.py
```

---

## App Features

- **Instant Classification:** Upload a `.npy` beat window for immediate analysis.
- **Live Hospital Monitor:** Stream a demo record directly from PhysioNet (auto-advances beat by beat).
- **Grad-CAM Overlay:** A heatmap shows exactly which part of the ECG triggered the model's alert.
- **Attention Map:** Visualizes the LSTM's temporal focus across the sequence.
- **Adjustable Threshold:** A slider to manually tune the model's sensitivity in real-time.
- **Beat History:** A scrolling log tracking the patient's recent heartbeats and confidence scores.

---

## Architecture Overview

```text
Input (1, 360)
    │
    ▼
┌─────────────────────────────────┐
│  CNN Block 1  Conv1D(1→32, k=5) │
│              + SE Attention      │
│              + MaxPool → (32,180)│
├─────────────────────────────────┤
│  CNN Block 2  Conv1D(32→64, k=3)│
│              + SE Attention      │
│              + MaxPool → (64,90) │
├─────────────────────────────────┤
│  CNN Block 3  Conv1D(64→128,k=3)│
│              + SE Attention      │
│              + MaxPool → (128,45)│
└──────────────┬──────────────────┘
               │ permute → (45, 128)
               ▼
┌──────────────────────────────────┐
│  BiLSTM  hidden=128 layers=2     │
│  Output: (45, 256)               │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  Temporal Attention              │
│  Output: (256,) context vector   │
└──────────────┬───────────────────┘
               │
               ▼
┌─────────────────────┐
│  Linear(256→64→1)   │  → Sigmoid → probability of ABNORMAL
└─────────────────────┘
```

---

## Why CNN + BiLSTM beats either alone

| Architecture | Catches | Misses |
|-------------|--------|--------|
| CNN only | QRS shape anomalies, peak morphology | Long rhythm patterns, RR interval changes |
| LSTM only | Sequential rhythm (AFib pattern) | Fine waveform shape details |
| CNN + BiLSTM | Both morphology AND rhythm | Minimal |

---

## Grad-CAM Explained

Grad-CAM (Gradient-weighted Class Activation Mapping) provides clinical transparency by:

1. Running a forward pass through the model  
2. Back-propagating gradients to the last CNN layer  
3. Weighting each feature map by its average gradient  
4. Summing the results into a 1D saliency map over the ECG samples  

Result: A heatmap over the raw ECG showing which samples triggered the classification. In practice, abnormal beats heavily highlight the QRS complex region where the anomaly occurs.

---

## Key Design Decisions

**Patient-Level Splitting:**  
We split the records into Train/Val/Test *before* extracting the beats. This prevents the model from memorizing patient-specific ECG patterns.

**SMOTE Usage:**  
Always apply SMOTE only on the training set. Applying it before splitting leads to data leakage and inflated metrics.

**Loss Function Choice:**  
BCEWithLogitsLoss with `pos_weight=4.0` penalizes missed anomalies more heavily, prioritizing safety.

**Why Recall Matters:**  
A model predicting all beats as normal can still achieve ~75% accuracy. Recall for the abnormal class ensures real anomalies are detected.
