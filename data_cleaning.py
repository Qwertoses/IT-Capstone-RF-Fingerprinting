import numpy as np
import sigmf
from scipy.signal import butter, sosfilt
from sigmf import SigMFFile

# --- Configuration ---
INPUT_PREFIX = "Device_A_run_1_center_freq_1000"  # Match your SigMF filename (without extension)
OUTPUT_FILE = "clean_bursts.npy"
PRE_TRIGGER = 200          # 200 samples @ 20 MSPS = 10 us before burst (captures turn-on transient)
WINDOW_LEN = 2048          # 2048 samples @ 20 MSPS = 102.4 us total slice per burst
MIN_GAP_SAMPLES = 200000   # 10 ms lockout @ 20 MSPS (prevents double-triggering on one burst)
THRESHOLD_MULTIPLIER = 8.0 # Trigger when in-band power is 8x above the in-band noise floor
MIN_BURST_SAMPLES = 3000   # Burst must stay above threshold for >= 150 us (rejects short noise spikes)

# 1. Load SigMF metadata and raw samples
handle = sigmf.sigmffile.fromfile(INPUT_PREFIX)
samp_rate = handle.get_global_field(SigMFFile.SAMPLE_RATE_KEY)
iq_data = handle.read_samples(autoscale=False)  # Unfiltered raw complex64 data

# 2. Apply a 1.2 MHz low-pass filter ONLY for the trigger detector
#    (Isolates your 1 MHz OFDM signal from out-of-band Wi-Fi/BLE interference in the room)
sos = butter(4, 1.2e6 / (samp_rate / 2.0), btype="low", output="sos")
filtered_iq = sosfilt(sos, iq_data)

# 3. Compute smoothed in-band power and robust noise floor
inband_power = np.abs(filtered_iq) ** 2
noise_floor = np.median(inband_power)
threshold = noise_floor * THRESHOLD_MULTIPLIER

kernel = np.ones(64) / 64.0
smooth_power = np.convolve(inband_power, kernel, mode="same")

# 4. Find rising edges where in-band power crosses above threshold
above_thresh = smooth_power > threshold
rising_edges = np.where(np.diff(above_thresh.astype(int)) == 1)[0]

# 5. Slice clean bursts from the ORIGINAL unfiltered iq_data
clean_bursts = []
last_trigger = -MIN_GAP_SAMPLES

for edge in rising_edges:
    if (edge - last_trigger) < MIN_GAP_SAMPLES:
        continue

    start_idx = edge - PRE_TRIGGER
    end_idx = start_idx + WINDOW_LEN

    if start_idx < 0 or end_idx > len(iq_data):
        continue

    # Verify the signal stays elevated (filters out brief ambient RF spikes)
    if np.sum(above_thresh[edge : edge + MIN_BURST_SAMPLES]) < (MIN_BURST_SAMPLES * 0.7):
        continue

    # Extract from raw unfiltered iq_data so transient & hardware features remain untouched
    burst = iq_data[start_idx:end_idx].copy()

    # Normalize to unit RMS power
    rms = np.sqrt(np.mean(np.abs(burst) ** 2))
    if rms > 0:
        burst = burst / rms

    clean_bursts.append(burst)
    last_trigger = edge

clean_bursts = np.array(clean_bursts, dtype=np.complex64)
np.save(OUTPUT_FILE, clean_bursts)

print(f"In-band noise floor: {noise_floor:.2e} | Trigger threshold: {threshold:.2e}")
print(f"Extracted clean bursts: {len(clean_bursts)} bursts of {WINDOW_LEN} samples")
print(f"Saved clean dataset: {OUTPUT_FILE} ({clean_bursts.nbytes / 1e6:.2f} MB)")
