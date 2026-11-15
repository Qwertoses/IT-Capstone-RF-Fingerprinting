import numpy as np
import sigmf
from sigmf import SigMFFile

# --- Configuration ---
INPUT_PREFIX = "capture"        # Matches capture.sigmf-meta and capture.sigmf-data
OUTPUT_FILE = "clean_bursts.npy"
PRE_TRIGGER = 200               # 200 samples @ 20 MSPS = 10 us before burst (captures turn-on transient)
WINDOW_LEN = 2048               # 2048 samples @ 20 MSPS = 102.4 us total slice per burst
MIN_GAP_SAMPLES = 100000        # Minimum spacing between bursts (5 ms @ 20 MSPS; actual gap is ~400k)
THRESHOLD_MULTIPLIER = 6.0      # Trigger when signal power exceeds 6x the median noise floor

# 1. Load SigMF metadata and memory-map the 1.5 GB binary file (avoids RAM spikes)
handle = sigmf.sigmffile.fromfile(INPUT_PREFIX)
samp_rate = handle.get_global_field(SigMFFile.SAMPLE_RATE_KEY)
iq_data = handle.read_samples(autoscale=False)  # Returns complex64 array

# 2. Estimate noise floor robustly using median instantaneous power
power = np.abs(iq_data) ** 2
noise_floor = np.median(power)
threshold = noise_floor * THRESHOLD_MULTIPLIER

# Smooth power slightly over 32 samples (1.6 us) to prevent false triggers on single noise spikes
kernel = np.ones(32) / 32.0
smooth_power = np.convolve(power, kernel, mode="same")

# 3. Find rising edges where smoothed power crosses above threshold
above_thresh = smooth_power > threshold
rising_edges = np.where(np.diff(above_thresh.astype(int)) == 1)[0]

# 4. Slice clean bursts with pre-trigger and refractory lockout
clean_bursts = []
last_trigger = -MIN_GAP_SAMPLES

for edge in rising_edges:
    # Enforce minimum gap so we only trigger once per OFDM packet
    if (edge - last_trigger) < MIN_GAP_SAMPLES:
        continue

    start_idx = edge - PRE_TRIGGER
    end_idx = start_idx + WINDOW_LEN

    # Discard bursts clipped by the start or end of the 10-second file
    if start_idx < 0 or end_idx > len(iq_data):
        continue

    burst = iq_data[start_idx:end_idx].copy()

    # 5. Normalize burst to unit RMS power (preserves transient shape & phase, removes gain bias)
    rms = np.sqrt(np.mean(np.abs(burst) ** 2))
    if rms > 0:
        burst = burst / rms

    clean_bursts.append(burst)
    last_trigger = edge

clean_bursts = np.array(clean_bursts, dtype=np.complex64)
np.save(OUTPUT_FILE, clean_bursts)

print(f"Original file samples: {len(iq_data):,} (~{iq_data.nbytes / 1e9:.2f} GB)")
print(f"Extracted clean bursts: {len(clean_bursts)} bursts of {WINDOW_LEN} samples")
print(f"Saved clean dataset: {OUTPUT_FILE} ({clean_bursts.nbytes / 1e6:.2f} MB)")
