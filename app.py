"""
ECG Anomaly Detection — Streamlit Web App
==========================================
Run with:  streamlit run app.py

Features
--------
• Upload a saved ECG .npy window OR stream a demo record live
• Real-time beat classification with confidence score
• Grad-CAM saliency overlay on the ECG waveform
• Attention weight heatmap showing model focus
• Beat-by-beat history log with timestamps
• Downloadable prediction report
"""

import os
import time
import datetime
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
import wfdb

# ── Import model and Grad-CAM from training script
from train_evaluate import ECGHybridModel, GradCAM, DEVICE


# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ECG Anomaly Monitor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for a clean clinical aesthetic
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stMetric { background: #1a1d27; border-radius: 10px; padding: 16px; }
    .stMetric label { color: #8b9bbd !important; font-size: 13px !important; }
    .alert-normal   { background:#0d3320; color:#4ade80; border:1px solid #166534;
                       border-radius:8px; padding:10px 16px; font-weight:600; font-size:15px; }
    .alert-abnormal { background:#3b0d0d; color:#f87171; border:1px solid #7f1d1d;
                       border-radius:8px; padding:10px 16px; font-weight:600; font-size:15px; }
    .beat-row-normal   { background:#0d3320; border-left:3px solid #4ade80;
                          padding:6px 12px; border-radius:4px; margin:3px 0; font-size:13px; }
    .beat-row-abnormal { background:#3b0d0d; border-left:3px solid #f87171;
                          padding:6px 12px; border-radius:4px; margin:3px 0; font-size:13px; }
    h1 { color: #e2e8f0 !important; }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MODEL LOADING (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model(model_path="models/best_model.pth"):
    model = ECGHybridModel().to(DEVICE)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        return model
    return None


# ─────────────────────────────────────────────
#  INFERENCE HELPERS
# ─────────────────────────────────────────────
def predict(model, window: np.ndarray, threshold=0.5):
    """
    window : 1-D numpy array of shape (360,) — already normalised
    Returns (label_str, probability, attention_weights)
    """
    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits, attn = model(x, return_attention=True)
    prob = torch.sigmoid(logits).item()
    label = "ABNORMAL" if prob >= threshold else "Normal"
    return label, prob, attn.squeeze().cpu().numpy()


def get_gradcam(model, window: np.ndarray) -> np.ndarray:
    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    gc = GradCAM(model)
    return gc.generate(x)


def normalise_window(window: np.ndarray) -> np.ndarray:
    mu  = window.mean()
    std = window.std() + 1e-8
    return (window - mu) / std


# ─────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────
def plot_ecg_with_cam(window: np.ndarray, cam: np.ndarray, label: str, prob: float):
    """ECG signal with Grad-CAM heat overlay."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5),
                                    gridspec_kw={'height_ratios': [3, 1]},
                                    facecolor='#0f1117')

    t = np.arange(len(window))
    color = '#f87171' if label == 'ABNORMAL' else '#4ade80'

    # ── ECG waveform
    ax1.set_facecolor('#1a1d27')
    ax1.plot(t, window, color=color, lw=1.2, zorder=3)

    # ── Grad-CAM overlay as shaded fill
    ax1.fill_between(t,
                     window.min() - 0.1,
                     window.max() + 0.1,
                     alpha=cam * 0.55,
                     color='#ef4444',
                     zorder=2)

    ax1.axvline(x=180, color='#94a3b8', lw=0.8, linestyle=':', alpha=0.6)
    ax1.set_ylabel("Amplitude (mV)", color='#8b9bbd', fontsize=10)
    ax1.tick_params(colors='#8b9bbd')
    ax1.spines[:].set_color('#2d3748')
    confidence_str = f"{prob*100:.1f}%"
    ax1.set_title(
        f"ECG Beat  ·  Prediction: {label}  ·  Confidence: {confidence_str}",
        color='white', fontsize=12, fontweight='bold', pad=8
    )

    # ── Grad-CAM intensity bar
    ax2.set_facecolor('#1a1d27')
    ax2.imshow(cam[np.newaxis, :], aspect='auto', cmap='Reds',
               extent=[0, len(window), 0, 1], vmin=0, vmax=1)
    ax2.set_yticks([])
    ax2.set_xlabel("Sample index  (R-peak at 180)", color='#8b9bbd', fontsize=10)
    ax2.tick_params(colors='#8b9bbd')
    ax2.spines[:].set_color('#2d3748')
    ax2.set_title("Grad-CAM saliency  (red = model focus)",
                  color='#8b9bbd', fontsize=9, pad=4)

    plt.tight_layout(pad=1.5)
    return fig


def plot_attention(attn_weights: np.ndarray):
    """Plot LSTM attention weights as a heatmap."""
    fig, ax = plt.subplots(figsize=(10, 1.5), facecolor='#0f1117')
    ax.set_facecolor('#1a1d27')
    ax.imshow(attn_weights[np.newaxis, :], aspect='auto', cmap='YlOrRd',
              extent=[0, len(attn_weights), 0, 1])
    ax.set_yticks([])
    ax.set_xlabel("LSTM timestep position", color='#8b9bbd', fontsize=10)
    ax.tick_params(colors='#8b9bbd')
    ax.spines[:].set_color('#2d3748')
    ax.set_title("Temporal attention weights", color='#8b9bbd', fontsize=9, pad=4)
    plt.tight_layout(pad=1.0)
    return fig


def plot_history(history: list):
    """Scrolling bar chart of recent beat predictions."""
    if not history:
        return None

    recent = history[-30:]
    probs  = [h['prob'] for h in recent]
    colors = ['#f87171' if p >= 0.5 else '#4ade80' for p in probs]

    fig, ax = plt.subplots(figsize=(10, 2.5), facecolor='#0f1117')
    ax.set_facecolor('#1a1d27')
    ax.bar(range(len(probs)), probs, color=colors, width=0.7)
    ax.axhline(y=0.5, color='#94a3b8', lw=1, linestyle='--', alpha=0.6)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Prob.", color='#8b9bbd', fontsize=9)
    ax.set_xlabel("Recent beats (oldest → newest)", color='#8b9bbd', fontsize=9)
    ax.set_title("Beat history", color='white', fontsize=10, fontweight='bold')
    ax.tick_params(colors='#8b9bbd')
    ax.spines[:].set_color('#2d3748')

    normal_patch   = mpatches.Patch(color='#4ade80', label='Normal')
    abnormal_patch = mpatches.Patch(color='#f87171', label='Abnormal')
    ax.legend(handles=[normal_patch, abnormal_patch],
              facecolor='#1a1d27', labelcolor='#e2e8f0', fontsize=8, loc='upper left')

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
#  DEMO RECORD LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_demo_record(record_id="100", max_beats=200):
    """Pull a record from PhysioNet and extract beat windows."""
    try:
        record     = wfdb.rdrecord(record_id, pn_dir='mitdb')
        annotation = wfdb.rdann(record_id, 'atr', pn_dir='mitdb')
        signal     = record.p_signal[:, 0]

        windows, ground_truth = [], []
        normal_labels   = {'N','L','R','e','j'}
        abnormal_labels = {'A','a','J','S','V','E','F','f','!','x','Q','/'}

        for peak, sym in zip(annotation.sample, annotation.symbol):
            if peak - 180 < 0 or peak + 180 > len(signal):
                continue
            win = signal[peak - 180 : peak + 180]
            win = normalise_window(win)
            if sym in normal_labels:
                windows.append(win); ground_truth.append(0)
            elif sym in abnormal_labels:
                windows.append(win); ground_truth.append(1)
            if len(windows) >= max_beats:
                break

        return np.array(windows), np.array(ground_truth)
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────
def main():
    # ── Session state
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'demo_idx' not in st.session_state:
        st.session_state.demo_idx = 0

    # ── Load model
    model = load_model()

    # ── Header
    st.title("🫀 ECG Anomaly Monitor")
    st.markdown(
        "<p style='color:#8b9bbd;margin-top:-12px;'>CNN + BiLSTM + Attention  ·  MIT-BIH Arrhythmia Database</p>",
        unsafe_allow_html=True
    )
    st.divider()

    # ── Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        threshold = st.slider("Decision threshold", 0.3, 0.8, 0.5, 0.01,
                               help="Lower = more sensitive (catches more anomalies, more false alarms)")
        st.caption("In clinical use, lower thresholds are preferred to minimise missed anomalies.")

        st.divider()
        st.subheader("Demo record")
        demo_record = st.selectbox("MIT-BIH record", ["100","101","119","200","208","231"],
                                    help="Record 208 & 231 have high arrhythmia density")
        max_beats = st.slider("Max beats to load", 50, 300, 150, 10)

        st.divider()
        show_gradcam = st.toggle("Show Grad-CAM", value=True)
        show_attn    = st.toggle("Show attention map", value=True)

        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

        if model is None:
            st.error("Model not found at models/best_model.pth\nRun train_evaluate.py first.")

    # ── Two tabs: Upload vs Demo
    tab_upload, tab_demo, tab_about = st.tabs(["📂 Upload window", "▶ Demo stream", "ℹ️ About"])

    # ════════════════════════════════════════
    #  TAB 1 — Upload a .npy window
    # ════════════════════════════════════════
    with tab_upload:
        st.markdown("Upload a **single ECG beat window** as a `.npy` file — shape `(360,)` or `(1,1,360)`.")
        uploaded = st.file_uploader("Choose .npy file", type=["npy"])

        if uploaded and model:
            raw = np.load(uploaded)
            window = raw.flatten()[:360].astype(np.float32)
            window = normalise_window(window)

            label, prob, attn = predict(model, window, threshold)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Prediction", label)
            col2.metric("Confidence", f"{prob*100:.1f}%")
            col3.metric("Threshold",  f"{threshold*100:.0f}%")
            col4.metric("Model", "CNN+BiLSTM")

            if label == "ABNORMAL":
                st.markdown('<div class="alert-abnormal">⚠️ ABNORMAL beat detected — review recommended</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-normal">✓ Normal sinus rhythm</div>',
                            unsafe_allow_html=True)

            st.divider()

            if show_gradcam:
                cam = get_gradcam(model, window)
                fig = plot_ecg_with_cam(window, cam, label, prob)
                st.pyplot(fig)
                plt.close()

            if show_attn:
                fig_attn = plot_attention(attn)
                st.pyplot(fig_attn)
                plt.close()

    # ════════════════════════════════════════
    #  TAB 2 — Demo stream
    # ════════════════════════════════════════
    with tab_demo:
        if model is None:
            st.warning("Train the model first, then reload this page.")
            st.stop()

        col_load, col_run = st.columns([2, 1])
        with col_load:
            if st.button("📥 Load record from PhysioNet", type="primary"):
                with st.spinner(f"Downloading record {demo_record} from PhysioNet..."):
                    windows, gt = load_demo_record(demo_record, max_beats)
                if windows is None:
                    st.error(f"Could not load record {demo_record}: {gt}")
                else:
                    st.session_state.demo_windows = windows
                    st.session_state.demo_gt      = gt
                    st.session_state.demo_idx     = 0
                    st.success(f"Loaded {len(windows)} beats from record {demo_record}")

        if 'demo_windows' not in st.session_state:
            st.info("Click 'Load record' to stream an ECG from PhysioNet.")
        else:
            windows = st.session_state.demo_windows
            gt      = st.session_state.demo_gt
            idx     = st.session_state.demo_idx

            # ── Navigation
            c1, c2, c3 = st.columns([1, 2, 1])
            with c1:
                if st.button("⏮ Previous") and idx > 0:
                    st.session_state.demo_idx -= 1
                    st.rerun()
            with c2:
                st.markdown(
                    f"<p style='text-align:center;color:#8b9bbd'>Beat {idx+1} / {len(windows)}</p>",
                    unsafe_allow_html=True
                )
            with c3:
                if st.button("Next ⏭") and idx < len(windows) - 1:
                    st.session_state.demo_idx += 1
                    st.rerun()

            auto_play = st.toggle("Auto-play (1 beat/sec)", value=False)

            window = windows[idx]
            label, prob, attn = predict(model, window, threshold)
            ground_truth_label = "Abnormal" if gt[idx] == 1 else "Normal"

            # ── Add to history
            st.session_state.history.append({
                'beat': idx,
                'label': label,
                'prob': prob,
                'gt': ground_truth_label,
                'time': datetime.datetime.now().strftime("%H:%M:%S")
            })

            # ── Metrics row
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Prediction",  label)
            col2.metric("Confidence",  f"{prob*100:.1f}%")
            col3.metric("Ground truth (MIT-BIH)", ground_truth_label)
            match = label.lower().replace("abnormal","abnormal") == ground_truth_label.lower()
            col4.metric("Match", "✓ Correct" if match else "✗ Incorrect")

            if label == "ABNORMAL":
                st.markdown('<div class="alert-abnormal">⚠️ ABNORMAL beat — arrhythmia detected</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-normal">✓ Normal sinus rhythm</div>',
                            unsafe_allow_html=True)

            # ── ECG plot
            if show_gradcam:
                cam = get_gradcam(model, window)
                fig = plot_ecg_with_cam(window, cam, label, prob)
            else:
                cam = np.zeros(360)
                fig = plot_ecg_with_cam(window, cam, label, prob)
            st.pyplot(fig)
            plt.close()

            if show_attn:
                fig_attn = plot_attention(attn)
                st.pyplot(fig_attn)
                plt.close()

            # ── Beat history chart
            if len(st.session_state.history) > 1:
                st.divider()
                st.subheader("Beat history")
                fig_hist = plot_history(st.session_state.history)
                if fig_hist:
                    st.pyplot(fig_hist)
                    plt.close()

            # ── History log table
            if st.session_state.history:
                with st.expander("Full prediction log", expanded=False):
                    for h in reversed(st.session_state.history[-50:]):
                        cls = "beat-row-abnormal" if h['label'] == 'ABNORMAL' else "beat-row-normal"
                        st.markdown(
                            f'<div class="{cls}">'
                            f'Beat {h["beat"]+1}  ·  {h["time"]}  ·  '
                            f'<b>{h["label"]}</b>  ({h["prob"]*100:.1f}%)  '
                            f'GT: {h["gt"]}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

            # ── Auto-play
            if auto_play and idx < len(windows) - 1:
                time.sleep(1.0)
                st.session_state.demo_idx += 1
                st.rerun()

    # ════════════════════════════════════════
    #  TAB 3 — About
    # ════════════════════════════════════════
    with tab_about:
        st.markdown("""
## Architecture: CNN + BiLSTM + Attention

**Why this hybrid?**

| Layer | Role |
|-------|------|
| Conv1D blocks (3×) | Extract local morphological features — P wave shape, QRS width, T wave height |
| Channel attention (SE) | Weight which feature maps matter for this beat |
| Bidirectional LSTM | Learn temporal rhythm patterns across the beat sequence in both directions |
| Temporal attention | Spotlight the most diagnostically important timestep positions |
| Linear classifier | Binary: Normal vs Abnormal |

**Dataset:** MIT-BIH Arrhythmia Database (PhysioNet) — 48 two-channel ECG recordings, 30 minutes each, 360 Hz.

**Labelling:** AAMI EC57 standard — simplifies 15+ beat types to Normal vs Abnormal.

**Balancing:** SMOTE on training set only (no leakage into val/test).

**Key metrics:**
- **Recall** is the primary metric — missing a real arrhythmia (false negative) is clinically worse than a false alarm
- **F1-score** balances precision and recall
- **ROC-AUC** shows overall discriminative ability

**Explainability:**
- **Grad-CAM** highlights which ECG samples activated the final convolutional layer
- **Temporal attention** shows which LSTM timesteps the model weighted most
        """)

        st.divider()

        if model:
            total = sum(p.numel() for p in model.parameters() if p.requires_grad)
            st.metric("Model parameters", f"{total:,}")
            st.metric("Device", str(DEVICE))


if __name__ == "__main__":
    main()