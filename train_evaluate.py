"""
ECG Anomaly Detection — Model Training & Evaluation
=====================================================
Architecture: CNN feature extractor → Bidirectional LSTM → Attention → Classifier

Why this hybrid?
  • CNN layers learn LOCAL morphological features (QRS shape, T-wave height, P-wave presence)
  • BiLSTM reads the CNN feature sequence capturing TEMPORAL rhythm patterns across time steps
  • Attention layer weights the most diagnostically important time positions
  • Together they match or exceed pure CNN and pure LSTM on MIT-BIH benchmarks

Outputs
-------
models/best_model.pth          (best checkpoint by validation F1)
models/training_curves.png
models/confusion_matrix.png
models/roc_curve.png
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

os.makedirs("models", exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
#  MODEL ARCHITECTURE
# ─────────────────────────────────────────────
class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation block.
    Learns which CNN channels (feature maps) matter most for the current beat.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, T)
        scale = self.fc(x).unsqueeze(-1)   # (B, C, 1)
        return x * scale


class TemporalAttention(nn.Module):
    """
    Soft attention over LSTM timesteps.
    Learns WHICH parts of the heartbeat window to focus on.
    The attention weights can later be visualised (alternative to Grad-CAM).
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        # lstm_out: (B, T, hidden_dim)
        scores = self.attention(lstm_out).squeeze(-1)   # (B, T)
        weights = F.softmax(scores, dim=1)              # (B, T)
        context = (lstm_out * weights.unsqueeze(-1)).sum(dim=1)  # (B, hidden_dim)
        return context, weights


class ECGHybridModel(nn.Module):
    """
    CNN + BiLSTM + Attention classifier.

    Input  : (B, 1, 360)  — batch × 1 channel × 360 time samples
    Output : (B, 1)       — raw logit (apply sigmoid for probability)
    """
    def __init__(self, lstm_hidden=128, lstm_layers=2, dropout=0.4):
        super().__init__()

        # ── Block 1: local feature detection (fine-grained)
        self.block1 = nn.Sequential(
            nn.Conv1d(1,  32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),              # 360 → 180
            nn.Dropout(0.2)
        )
        self.attn1 = ChannelAttention(32)

        # ── Block 2: mid-level pattern recognition
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),              # 180 → 90
            nn.Dropout(0.2)
        )
        self.attn2 = ChannelAttention(64)

        # ── Block 3: high-level waveform encoding
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),              # 90 → 45
            nn.Dropout(0.2)
        )
        self.attn3 = ChannelAttention(128)

        # ── BiLSTM: reads the CNN feature sequence in both directions
        # Hidden × 2 because bidirectional
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        # ── Temporal attention over LSTM output
        self.temporal_attn = TemporalAttention(lstm_hidden * 2)

        # ── Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)             # single logit output
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x, return_attention=False):
        # CNN feature extraction
        x = self.block1(x)
        x = self.attn1(x)
        x = self.block2(x)
        x = self.attn2(x)
        x = self.block3(x)
        x = self.attn3(x)

        # (B, 128, 45) → (B, 45, 128) for LSTM
        x = x.permute(0, 2, 1)

        # BiLSTM
        lstm_out, _ = self.lstm(x)        # (B, 45, 256)

        # Temporal attention pooling
        context, attn_weights = self.temporal_attn(lstm_out)  # (B, 256)

        logits = self.classifier(context) # (B, 1)

        if return_attention:
            return logits, attn_weights
        return logits


# ─────────────────────────────────────────────
#  GRAD-CAM
# ─────────────────────────────────────────────
class GradCAM:
    """
    Computes a saliency map over the raw ECG signal showing which samples
    contributed most to the model's prediction.
    Hooks into the last CNN block's output.
    """
    def __init__(self, model: ECGHybridModel):
        self.model = model
        self.activations = None
        self.gradients   = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd(m, inp, out):
            self.activations = out.detach()

        def bwd(m, inp, out):
            self.gradients = out[0].detach()

        # Hook into the final conv layer of block3
        target = self.model.block3[0]   # first Conv1d of block3
        target.register_forward_hook(fwd)
        target.register_backward_hook(bwd)

    def generate(self, x: torch.Tensor) -> np.ndarray:
        """x: (1, 1, 360) single sample tensor."""
        self.model.eval()
        x = x.requires_grad_(True)
        logit = self.model(x)
        self.model.zero_grad()
        logit.backward()

        # Weighted activation map
        weights = self.gradients.mean(dim=-1, keepdim=True)   # (1, C, 1)
        cam = (weights * self.activations).sum(dim=1).relu()  # (1, T')
        cam = cam.squeeze().cpu().numpy()

        # Upsample to original 360 length
        cam_up = np.interp(
            np.linspace(0, 1, 360),
            np.linspace(0, 1, len(cam)),
            cam
        )
        # Normalise to [0, 1]
        if cam_up.max() > 0:
            cam_up = cam_up / cam_up.max()
        return cam_up


# ─────────────────────────────────────────────
#  TRAINING UTILITIES
# ─────────────────────────────────────────────
def make_dataloaders(data_dir="data", batch_size=128):
    def load(name):
        return torch.tensor(np.load(os.path.join(data_dir, f"{name}.npy")))

    X_train = load("X_train").float()
    y_train = load("y_train").float()
    X_val   = load("X_val").float()
    y_val   = load("y_val").float()
    X_test  = load("X_test").float()
    y_test  = load("y_test").float()

    train_dl = DataLoader(TensorDataset(X_train, y_train),
                          batch_size=batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(TensorDataset(X_val,   y_val),
                          batch_size=batch_size, shuffle=False, num_workers=0)
    test_dl  = DataLoader(TensorDataset(X_test,  y_test),
                          batch_size=batch_size, shuffle=False, num_workers=0)

    return train_dl, val_dl, test_dl, y_test.numpy()


def evaluate(model, loader, criterion, threshold=0.5):
    """Run one pass over a dataloader, return metrics dict."""
    model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []

    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            logits = model(X_b).squeeze()
            loss = criterion(logits, y_b)
            total_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_b.cpu().numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds      = (all_probs >= threshold).astype(int)

    return {
        "loss":      total_loss / len(loader),
        "f1":        f1_score(all_labels, preds, zero_division=0),
        "recall":    recall_score(all_labels, preds, zero_division=0),
        "precision": precision_score(all_labels, preds, zero_division=0),
        "roc_auc":   roc_auc_score(all_labels, all_probs),
        "probs":     all_probs,
        "labels":    all_labels,
    }


# ─────────────────────────────────────────────
#  PLOT HELPERS
# ─────────────────────────────────────────────
def plot_training_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    metrics = [
        ('train_loss', 'val_loss',      'Loss',      'Loss'),
        ('train_f1',   'val_f1',        'F1-Score',  'F1'),
        ('train_rec',  'val_rec',       'Recall',    'Recall'),
        ('train_prec', 'val_prec',      'Precision', 'Precision'),
    ]

    colours = {'train': '#4A9EBF', 'val': '#E05C4B'}

    for idx, (tr_key, va_key, title, ylabel) in enumerate(metrics):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.plot(epochs, history[tr_key], color=colours['train'], lw=1.8, label='Train')
        ax.plot(epochs, history[va_key], color=colours['val'],   lw=1.8, label='Val',
                linestyle='--')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Training Curves — CNN+BiLSTM+Attention', fontsize=14, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Training curves saved → {save_path}")


def plot_confusion(cm, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    fig.colorbar(im)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Abnormal'])
    ax.set_yticklabels(['Normal', 'Abnormal'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                    fontsize=14, color='white' if cm[i,j] > cm.max()/2 else 'black')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved → {save_path}")


def plot_roc(labels, probs, auc, save_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='#4A9EBF', lw=2, label=f'ROC (AUC = {auc:.4f})')
    ax.plot([0,1],[0,1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ROC curve saved → {save_path}")


# ─────────────────────────────────────────────
#  MAIN TRAINING LOOP
# ─────────────────────────────────────────────
def train(
    epochs=40,
    batch_size=128,
    lr=1e-3,
    pos_weight=2.0,
    data_dir="data",
    patience=8
):
    print(f"\n{'='*55}")
    print(f"  ECG Anomaly — Training")
    print(f"  Device: {DEVICE}  |  Epochs: {epochs}  |  Batch: {batch_size}")
    print(f"{'='*55}\n")

    # ── Data
    train_dl, val_dl, test_dl, y_test = make_dataloaders(data_dir, batch_size)
    print(f"  Train batches: {len(train_dl)}  |  Val batches: {len(val_dl)}  |  Test batches: {len(test_dl)}\n")

    # ── Model
    model = ECGHybridModel().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {total_params:,}\n")

    # ── Loss: pos_weight penalises false negatives (missing real anomalies)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight]).to(DEVICE)
    )

    # ── Optimiser + scheduler
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='max', factor=0.5, patience=4
    )

    # ── History
    history = {k: [] for k in [
        'train_loss','val_loss',
        'train_f1','val_f1',
        'train_rec','val_rec',
        'train_prec','val_prec'
    ]}

    best_val_f1   = 0.0
    early_stop    = 0

    for epoch in range(1, epochs + 1):
        # ── Training pass
        model.train()
        train_loss = 0.0

        for X_b, y_b in train_dl:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimiser.zero_grad()
            logits = model(X_b).squeeze()
            loss   = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            train_loss += loss.item()

        # ── Validation metrics
        train_metrics = evaluate(model, train_dl, criterion)
        val_metrics   = evaluate(model, val_dl,   criterion)

        scheduler.step(val_metrics['f1'])

        # ── Record history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
        history['train_rec'].append(train_metrics['recall'])
        history['val_rec'].append(val_metrics['recall'])
        history['train_prec'].append(train_metrics['precision'])
        history['val_prec'].append(val_metrics['precision'])

        # ── Checkpoint
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save(model.state_dict(), "models/best_model.pth")
            early_stop = 0
            flag = " ✓ (saved)"
        else:
            early_stop += 1
            flag = f"  (no improvement {early_stop}/{patience})"

        print(
            f"  Epoch {epoch:02d}/{epochs} | "
            f"Loss {val_metrics['loss']:.4f} | "
            f"F1 {val_metrics['f1']:.4f} | "
            f"Recall {val_metrics['recall']:.4f} | "
            f"AUC {val_metrics['roc_auc']:.4f}"
            f"{flag}"
        )

        if early_stop >= patience:
            print(f"\n  Early stopping triggered at epoch {epoch}.")
            break

    # ── Final evaluation on held-out test set
    print(f"\n{'─'*55}")
    print("  Loading best checkpoint for test evaluation...")
    model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
    test_metrics = evaluate(model, test_dl, criterion)

    cm = confusion_matrix(test_metrics['labels'], (test_metrics['probs'] >= 0.5).astype(int))

    print(f"\n  ── TEST RESULTS ──────────────────────────────")
    print(f"  F1-Score  : {test_metrics['f1']:.4f}")
    print(f"  Recall    : {test_metrics['recall']:.4f}   ← most important in medical use")
    print(f"  Precision : {test_metrics['precision']:.4f}")
    print(f"  ROC-AUC   : {test_metrics['roc_auc']:.4f}")
    print(f"  Confusion matrix:\n{cm}")

    # ── Save plots
    plot_training_curves(history,  "models/training_curves.png")
    plot_confusion(cm,             "models/confusion_matrix.png")
    plot_roc(test_metrics['labels'],
             test_metrics['probs'],
             test_metrics['roc_auc'],
             "models/roc_curve.png")

    print(f"\n  Best model  → models/best_model.pth")
    return model


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train CNN+BiLSTM ECG anomaly model")
    parser.add_argument("--epochs",     type=int,   default=40)
    parser.add_argument("--batch",      type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--data",       type=str,   default="data")
    parser.add_argument("--pos-weight", type=float, default=2.0,
                        help="Extra penalty for missing real anomalies")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        pos_weight=args.pos_weight,
        data_dir=args.data
    )
