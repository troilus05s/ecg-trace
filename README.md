# ECG Anomaly Detection
**Time-Series ML Project | CNN + BiLSTM + Attention | MIT-BIH Arrhythmia Database**

---

## Project structure

```
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
pip install -r requirements.txt
```

---

## Execution (3 commands)

### Step 1 — Build the dataset
Downloads MIT-BIH directly from PhysioNet. No account needed.

```bash
# Full dataset (all 48 records, ~10–15 min download)
python data_pipeline.py

# Quick test run (10 records, ~2 min)
python data_pipeline.py --quick
```

**Output:** `data/` folder with train/val/test `.npy` files + class distribution chart.

---

### Step 2 — Train the model

```bash
# Default settings (40 epochs, batch 128, lr 1e-3)
python train_evaluate.py

# Custom settings
python train_evaluate.py --epochs 60 --batch 64 --lr 5e-4 --pos-weight 3.0
```

**`--pos-weight`** is how much extra you penalise missing a real arrhythmia.
Higher = more sensitive model, more false alarms. Default 2.0 is a good balance.

**Output:** `models/` folder with checkpoint + training curve plots.

**Expected results on full dataset:**
| Metric | Target |
|--------|--------|
| F1-Score | ≥ 0.93 |
| Recall | ≥ 0.91 |
| ROC-AUC | ≥ 0.97 |

---

### Step 3 — Launch the web app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

**App features:**
- Upload a `.npy` beat window for instant classification
- Stream a live demo from PhysioNet (auto-advances beat by beat)
- Grad-CAM overlay shows which part of the ECG triggered the alert
- Attention map shows LSTM's temporal focus
- Adjustable decision threshold slider
- Beat history log

---

## Architecture overview

```
Input (1, 360)
    │
    ▼
┌─────────────────────────────────┐
│  CNN Block 1  Conv1D(1→32, k=5) │  ← Detects fine local features
│              + SE Attention      │
│              + MaxPool → (32,180)│
├─────────────────────────────────┤
│  CNN Block 2  Conv1D(32→64, k=3)│  ← Mid-level pattern recognition
│              + SE Attention      │
│              + MaxPool → (64,90) │
├─────────────────────────────────┤
│  CNN Block 3  Conv1D(64→128,k=3)│  ← High-level waveform encoding
│              + SE Attention      │
│              + MaxPool → (128,45)│
└──────────────┬──────────────────┘
               │ permute → (45, 128)
               ▼
┌──────────────────────────────────┐
│  BiLSTM  hidden=128 layers=2     │  ← Temporal rhythm patterns
│  Output: (45, 256)               │     bidirectional = past + future context
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  Temporal Attention              │  ← Focus on key timesteps
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
|---|---|---|
| CNN only | QRS shape anomalies, peak morphology | Long rhythm patterns, RR interval changes |
| LSTM only | Sequential rhythm (AFib pattern) | Fine waveform shape details |
| **CNN + BiLSTM** | **Both morphology AND rhythm** | Little — best of both worlds |

---

## Grad-CAM explained

Grad-CAM (Gradient-weighted Class Activation Mapping) works by:
1. Running a forward pass through the model
2. Back-propagating gradients to the last CNN layer
3. Weighting each feature map by its average gradient
4. Summing → a 1D saliency map over the ECG samples

The result: a heatmap over the raw ECG showing *which samples* triggered the classification. In practice, abnormal beats highlight the QRS complex region where the anomaly occurs.

---

## Key design decisions

**SMOTE before or after splitting?**
Always fit SMOTE only on the training set. Applying it before splitting leaks synthetic data into validation/test sets, making evaluation metrics artificially optimistic.

**Why BCEWithLogitsLoss + pos_weight?**
Even after SMOTE, false negatives (missed anomalies) are clinically worse than false positives. `pos_weight=2.0` applies an additional 2× penalty for each missed abnormal beat during training.

**Why Recall as primary metric?**
A model that flags 100% of beats as Normal gets ~75% accuracy on MIT-BIH. Recall for the abnormal class measures what actually matters: how many real arrhythmias the model catches.