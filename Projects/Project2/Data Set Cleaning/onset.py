import librosa 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.signal import find_peaks

def improved_onset_detector(x, sr, hop_length=512, frame_length=1024, threshold=0.1):
    
    rmse = librosa.feature.rms(
        y=x,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True)[0]

    rmse_diff = np.diff(rmse)
    novelty = np.concatenate([rmse_diff, [0]])  # pad to keep same length as rmse

    # ── Step 3: Half-wave rectification ───────────────────────────────────
    # Set negative values to 0 — we only care about energy *increases*,
    # which mark the beginning of a transient (onset).
    novelty[novelty < 0] = 0

    # ── Step 4: Normalize to 0–1 ──────────────────────────────────────────
    # Makes the threshold consistent regardless of overall signal loudness.
    if novelty.max() > 0:
        novelty = novelty / novelty.max()

    # ── Step 5: Local-maxima peak picking with threshold ──────────────────
    # From the lesson: a peak is valid only if it exceeds the threshold AND
    # is greater than both its immediate neighbors (true local maximum).
    # This avoids picking flat plateaus or the shoulders of broad peaks.
    t = librosa.frames_to_time(np.arange(len(novelty)), sr=sr, hop_length=hop_length)

    peak_indices = []
    for i in range(1, len(novelty) - 1):
        if novelty[i] > threshold:
            if novelty[i] > novelty[i - 1] and novelty[i] > novelty[i + 1]:
                peak_indices.append(i)

    peak_indices = np.array(peak_indices)
    onset_times = t[peak_indices] if len(peak_indices) > 0 else np.array([])

    # ── Plot ───────────────────────────────────────────────────────────────
    plt.figure(figsize=(12, 4))
    plt.plot(t, novelty, label="Novelty Function (RMS derivative)")
    plt.axhline(threshold, color='grey', linestyle=':', label=f"threshold={threshold}")
    for o in onset_times:
        plt.axvline(o, color='red', alpha=0.4)
    plt.xlabel("Time (s)")
    plt.title("Improved Onset Detection (RMS Novelty + Local Maxima)")
    plt.legend()
    plt.show()

    return onset_times, novelty, t
