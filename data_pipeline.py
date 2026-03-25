"""
ECG Anomaly Detection — Data Pipeline (Corrected)
===================================================
Key fixes over v1:
  1. PATIENT-LEVEL SPLIT  — records are divided into train/val/test BEFORE
     any beat extraction. Beats from Patient 100 will ONLY appear in one
     split, never across multiple. Prevents the model memorising individual
     ECG fingerprints instead of learning arrhythmia features.

  2. BUTTERWORTH BANDPASS — applied inside load_record() on the raw signal
     before windowing. Removes baseline wander (< 0.5 Hz) AND powerline
     interference (50/60 Hz) and muscle noise (> 45 Hz).

  3. SMOTE only on training beats — unchanged from v1, but now guaranteed
     clean because val/test are entirely held-out patients.

Outputs
-------
data/X_train.npy  (N_train, 1, 360)  float32
data/y_train.npy  (N_train,)          int64
data/X_val.npy    (N_val,   1, 360)
data/y_val.npy    (N_val,)
data/X_test.npy   (N_test,  1, 360)
data/y_test.npy   (N_test,)
data/split_info.txt               which patients went where
data/label_distribution.png
data/sample_windows.png
"""

import os
import numpy as np
import wfdb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

ALL_RECORDS = [
    "100","101","102","103","104","105","106","107",
    "108","109","111","112","113","114","115","116",
    "117","118","119","121","122","123","124","200",
    "201","202","203","205","207","208","209","210",
    "212","213","214","215","217","219","220","221",
    "222","223","228","230","231","232","233","234"
]

# AAMI EC57 standard — industry-accepted binary grouping
NORMAL_LABELS   = {'N', 'L', 'R', 'e', 'j'}
ABNORMAL_LABELS = {'A', 'a', 'J', 'S', 'V', 'E', 'F', 'f', '!', 'x', 'Q', '/'}

SAMPLING_RATE = 360   # Hz — MIT-BIH standard
WINDOW_SIZE   = 180   # samples each side of R-peak → 360 total


# ─────────────────────────────────────────────
#  FIX 2 — Butterworth bandpass filter
# ─────────────────────────────────────────────
def bandpass_filter(signal: np.ndarray,
                    lowcut: float = 0.5,
                    highcut: float = 45.0,
                    fs: int = SAMPLING_RATE,
                    order: int = 4) -> np.ndarray:
    """
    4th-order Butterworth bandpass filter.

    lowcut  = 0.5 Hz  → removes baseline wander (slow breathing drift)
    highcut = 45.0 Hz → removes powerline (50/60 Hz) and muscle noise

    Nyquist = fs / 2 = 180 Hz for MIT-BIH.
    We normalise cutoffs to Nyquist before passing to butter().

    filtfilt() applies the filter twice (forward + backward) giving
    zero phase distortion — important for preserving R-peak timing.
    """
    nyquist = fs / 2.0
    low  = lowcut  / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


# ─────────────────────────────────────────────
#  STEP 1 — Load one record + filter
# ─────────────────────────────────────────────
def load_record(record_id: str):
    """
    Download record from PhysioNet, apply bandpass filter,
    return (clean_signal, r_peak_indices, beat_symbols).
    """
    try:
        record     = wfdb.rdrecord(record_id, pn_dir='mitdb')
        annotation = wfdb.rdann(record_id, 'atr', pn_dir='mitdb')
        raw_signal = record.p_signal[:, 0]   # MLII lead

        # Apply bandpass BEFORE extracting windows
        clean_signal = bandpass_filter(raw_signal)

        return clean_signal, annotation.sample, annotation.symbol
    except Exception as e:
        print(f"    [SKIP] Record {record_id}: {e}")
        return None, None, None


# ─────────────────────────────────────────────
#  STEP 2 — Extract fixed-length windows
# ─────────────────────────────────────────────
def extract_windows(signal: np.ndarray,
                    r_peaks: np.ndarray,
                    symbols: list):
    """Slice the continuous ECG into WINDOW_SIZE*2 segments centred on R-peaks."""
    X, y = [], []
    for peak, sym in zip(r_peaks, symbols):
        start, end = peak - WINDOW_SIZE, peak + WINDOW_SIZE
        if start < 0 or end > len(signal):
            continue
        window = signal[start:end]
        if sym in NORMAL_LABELS:
            X.append(window); y.append(0)
        elif sym in ABNORMAL_LABELS:
            X.append(window); y.append(1)
    return X, y


# ─────────────────────────────────────────────
#  STEP 3 — Per-window Z-score normalisation
# ─────────────────────────────────────────────
def normalise(X: np.ndarray) -> np.ndarray:
    """Z-score each window independently — removes remaining amplitude differences."""
    mu  = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mu) / std


# ─────────────────────────────────────────────
#  STEP 4 — SMOTE oversampling
# ─────────────────────────────────────────────
def apply_smote(X: np.ndarray, y: np.ndarray):
    """
    SMOTE generates synthetic abnormal beats by interpolating between
    real minority-class examples in feature space.
    Applied ONLY to training data — never val or test.
    """
    print(f"    Before SMOTE — Normal: {(y==0).sum():,}  Abnormal: {(y==1).sum():,}")
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_flat = X.reshape(len(X), -1)
    X_res, y_res = sm.fit_resample(X_flat, y)
    X_res = X_res.reshape(-1, 1, X.shape[1]).astype(np.float32)
    print(f"    After  SMOTE — Normal: {(y_res==0).sum():,}  Abnormal: {(y_res==1).sum():,}")
    return X_res, y_res


# ─────────────────────────────────────────────
#  HELPER — Process a list of records into arrays
# ─────────────────────────────────────────────
def process_group(record_ids: list, label: str):
    """
    Load and window-extract all beats from a set of patient records.
    Returns raw (N, 360) float32 array and (N,) int64 label array.
    """
    all_X, all_y = [], []
    for i, rec_id in enumerate(record_ids, 1):
        print(f"    [{i:02d}/{len(record_ids)}] {rec_id}", end=' ')
        signal, r_peaks, symbols = load_record(rec_id)
        if signal is None:
            continue
        X_rec, y_rec = extract_windows(signal, r_peaks, symbols)
        all_X.extend(X_rec)
        all_y.extend(y_rec)
        n_abn = sum(1 for lbl in y_rec if lbl == 1)
        print(f"-> {len(X_rec)} beats  ({n_abn} abnormal)")

    if not all_X:
        return np.zeros((0, WINDOW_SIZE * 2), dtype=np.float32), np.zeros(0, dtype=np.int64)

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)
    print(f"  [{label}] total: {len(X):,} beats — "
          f"Normal: {(y==0).sum():,}  Abnormal: {(y==1).sum():,}\n")
    return X, y


# ─────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────
def plot_distribution(splits: dict, save_path: str):
    """Bar chart showing class balance across all three splits."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colours = ['#4A9EBF', '#E05C4B']

    for ax, (name, y) in zip(axes, splits.items()):
        if len(y) == 0:
            ax.set_title(f"{name} (empty)")
            continue
        counts = [(y == 0).sum(), (y == 1).sum()]
        bars = ax.bar(['Normal', 'Abnormal'], counts, color=colours,
                       width=0.55, edgecolor='white')
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.set_ylabel('Beats')
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(counts) * 0.01,
                    f'{count:,}', ha='center', va='bottom', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Class distribution per split (after SMOTE on train)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Distribution chart -> {save_path}")


def plot_sample_windows(X_train: np.ndarray, y_train_raw: np.ndarray, save_path: str):
    """Plot one normal and one abnormal window for visual sanity-check."""
    if (y_train_raw == 0).sum() == 0 or (y_train_raw == 1).sum() == 0:
        return  # not enough data to plot both classes
    normal_idx   = np.where(y_train_raw == 0)[0][0]
    abnormal_idx = np.where(y_train_raw == 1)[0][0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    t = np.arange(WINDOW_SIZE * 2) / SAMPLING_RATE * 1000  # ms

    for ax, idx, title, col in zip(
        axes,
        [normal_idx, abnormal_idx],
        ['Normal beat (class 0)', 'Abnormal beat (class 1)'],
        ['#4A9EBF', '#E05C4B']
    ):
        ax.plot(t, X_train[idx], color=col, lw=1.2)
        ax.axvline(x=500, color='gray', lw=0.8, linestyle=':', alpha=0.6,
                   label='R-peak (500 ms)')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Normalised amplitude')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Sample windows    -> {save_path}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def build_dataset(records=None, quick_run=False):
    """
    Full patient-safe pipeline.

    quick_run=True  -> 12 records (fast iteration / CI testing)
    quick_run=False -> all 48 records (production quality)
    """
    if records is None:
        records = ALL_RECORDS[:12] if quick_run else ALL_RECORDS

    print(f"\n{'='*60}")
    print(f"  MIT-BIH Data Pipeline  (patient-level split)")
    print(f"  Records: {len(records)}  |  Quick run: {quick_run}")
    print(f"{'='*60}\n")

    # ── FIX 1: Split the PATIENT RECORDS first, not the beats ──────────
    #
    # train_test_split on the list of record IDs ensures every beat from
    # Patient 100 goes to EXACTLY ONE split. No ECG fingerprint information
    # leaks between splits.
    #
    # 80% train / 10% val / 10% test
    train_recs, temp_recs = train_test_split(records, test_size=0.20, random_state=42)
    val_recs,  test_recs  = train_test_split(temp_recs,  test_size=0.50, random_state=42)

    print(f"  Patient split:")
    print(f"    Train : {len(train_recs)} records -> {sorted(train_recs)}")
    print(f"    Val   : {len(val_recs)}  records -> {sorted(val_recs)}")
    print(f"    Test  : {len(test_recs)}  records -> {sorted(test_recs)}")
    print()

    # Save split info for reproducibility and audit trail
    with open(os.path.join(DATA_DIR, "split_info.txt"), "w") as f:
        f.write(f"TRAIN ({len(train_recs)}): {sorted(train_recs)}\n")
        f.write(f"VAL   ({len(val_recs)}):  {sorted(val_recs)}\n")
        f.write(f"TEST  ({len(test_recs)}):  {sorted(test_recs)}\n")

    # ── Extract windows per patient group ───────────────────────────────
    print("  Loading TRAIN patients...")
    X_train_raw, y_train_raw = process_group(train_recs, "Train")

    print("  Loading VAL patients...")
    X_val_raw, y_val = process_group(val_recs, "Val")

    print("  Loading TEST patients...")
    X_test_raw, y_test = process_group(test_recs, "Test")

    # ── Normalise each split's windows independently ────────────────────
    # Per-window z-score has no fitting step, so no leakage risk here.
    print("  Normalising all splits...")
    X_train_norm = normalise(X_train_raw)
    X_val        = normalise(X_val_raw)
    X_test       = normalise(X_test_raw)

    # ── SMOTE only on training set ──────────────────────────────────────
    print("\n  Applying SMOTE to training set only...")
    X_train, y_train = apply_smote(X_train_norm, y_train_raw)

    # Reshape val and test to (N, 1, 360) for PyTorch Conv1d
    X_val  = X_val.reshape(-1,  1, WINDOW_SIZE * 2).astype(np.float32)
    X_test = X_test.reshape(-1, 1, WINDOW_SIZE * 2).astype(np.float32)
    # X_train already shaped (N, 1, 360) by apply_smote

    # ── Save arrays ──────────────────────────────────────────────────────
    print("\n  Saving arrays...")
    splits_to_save = {
        'X_train': X_train, 'y_train': y_train,
        'X_val':   X_val,   'y_val':   y_val,
        'X_test':  X_test,  'y_test':  y_test,
    }
    for name, arr in splits_to_save.items():
        path = os.path.join(DATA_DIR, f"{name}.npy")
        np.save(path, arr)
        print(f"    {name}.npy  shape={arr.shape}  dtype={arr.dtype}")

    # ── Plots ──────────────────────────────────────────────────────────
    plot_distribution(
        {'Train (post-SMOTE)': y_train,
         'Val (real patients)': y_val,
         'Test (real patients)': y_test},
        os.path.join(DATA_DIR, "label_distribution.png")
    )
    plot_sample_windows(
        X_train_norm, y_train_raw,
        os.path.join(DATA_DIR, "sample_windows.png")
    )

    print(f"\n  Done.")
    print(f"  Train : {X_train.shape}")
    print(f"  Val   : {X_val.shape}")
    print(f"  Test  : {X_test.shape}")
    print(f"  See data/split_info.txt for the patient assignment.\n")

    return splits_to_save


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MIT-BIH ECG Data Pipeline")
    parser.add_argument("--quick", action="store_true",
                        help="Use 12 records for fast testing")
    args = parser.parse_args()
    build_dataset(quick_run=args.quick)