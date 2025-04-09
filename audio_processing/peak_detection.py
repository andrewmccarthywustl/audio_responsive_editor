# audio_responsive_editor/audio_processing/peak_detection.py

import librosa
import numpy as np
import csv
import os
from datetime import datetime

# --- Constants (can be adjusted or made configurable) ---
DEFAULT_THRESHOLD_PERCENTILE = 99.7
DEFAULT_MIN_TIME_GAP_SECONDS = 0.5 # Minimum time between detected peaks (for amplitude mode)
DEFAULT_HPSS_MARGIN = 1.0 # Margin for HPSS separation (see librosa docs)
DEFAULT_ONSET_WAIT = 0.04 # Wait time in seconds for onset detector (see librosa docs)
DEFAULT_ONSET_PRE_AVG = 0.08 # Pre-average time in seconds (see librosa docs)
DEFAULT_ONSET_POST_AVG = 0.08 # Post-average time in seconds (see librosa docs)
DEFAULT_ONSET_PRE_MAX = 0.08 # Pre-max time in seconds (see librosa docs)
DEFAULT_ONSET_POST_MAX = 0.08 # Post-max time in seconds (see librosa docs)


def _find_peaks_amplitude(y, sr, percentile_threshold, min_gap_seconds):
    """Finds peaks based on amplitude envelope (original method)."""
    if y is None or len(y) == 0: return []
    amplitude_envelope = np.abs(y)
    if len(amplitude_envelope) == 0: return []

    if np.all(amplitude_envelope == amplitude_envelope[0]):
         threshold = np.max(amplitude_envelope) + 1
    else:
        # Ensure percentile is within valid range [0, 100]
        safe_percentile = np.clip(percentile_threshold, 0, 100)
        threshold = np.percentile(amplitude_envelope, safe_percentile)

    peak_indices = np.where(amplitude_envelope > threshold)[0]
    if len(peak_indices) == 0: return []

    min_samples_gap = int(min_gap_seconds * sr)
    merged_peak_indices = []
    last_peak_index = -min_samples_gap

    # Simplified merging - take first peak index in a cluster above threshold after gap
    current_cluster_start = -1
    for i in range(len(peak_indices)):
        idx = peak_indices[i]
        # Start a new cluster if gap is sufficient
        if idx - last_peak_index >= min_samples_gap:
            # If we were in a cluster, add its start index
            # Find the index of maximum amplitude within the cluster to refine peak position
            cluster_indices = peak_indices[(peak_indices >= last_peak_index + min_samples_gap) & (peak_indices < idx)]
            if len(cluster_indices) > 0:
                 refined_peak_idx = cluster_indices[np.argmax(amplitude_envelope[cluster_indices])]
                 # Check gap again with refined index
                 if refined_peak_idx - last_peak_index >= min_samples_gap:
                       merged_peak_indices.append(refined_peak_idx)
                       last_peak_index = refined_peak_idx

            # Add the current peak as the potential start of a new cluster
            merged_peak_indices.append(idx)
            last_peak_index = idx

    # Remove duplicates just in case merging logic added adjacent indices
    # Convert to numpy array first for unique
    if not merged_peak_indices: return []
    unique_peak_indices = np.unique(np.array(merged_peak_indices))

    # Convert indices to times
    return librosa.samples_to_time(unique_peak_indices, sr=sr)


def _find_onsets(y, sr, backtrack=True):
    """Finds onsets using librosa's default spectral flux method."""
    if y is None or len(y) == 0: return []
    # You might need to adjust parameters like wait, pre_avg, post_avg, pre_max, post_max, delta
    # based on the type of audio to fine-tune onset detection.
    # Using defaults which often work reasonably well.
    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        units='frames',
        backtrack=backtrack, # Backtrack to refine onset timing closer to energy increase
        wait=int(DEFAULT_ONSET_WAIT * sr / 512), # Parameters need to be in STFT frames
        pre_avg=int(DEFAULT_ONSET_PRE_AVG * sr / 512),
        post_avg=int(DEFAULT_ONSET_POST_AVG * sr / 512),
        pre_max=int(DEFAULT_ONSET_PRE_MAX * sr / 512),
        post_max=int(DEFAULT_ONSET_POST_MAX * sr / 512)
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    return onset_times


def find_peaks(audio_path,
               mode='amplitude', # New parameter: 'amplitude', 'percussive', 'harmonic'
               target_peaks=None,
               initial_threshold_percentile=DEFAULT_THRESHOLD_PERCENTILE,
               min_gap_seconds=DEFAULT_MIN_TIME_GAP_SECONDS,
               hpss_margin=DEFAULT_HPSS_MARGIN):
    """
    Analyzes an audio file to find significant peaks or onsets.

    Args:
        audio_path (str): Path to the audio file.
        mode (str): Detection mode: 'amplitude', 'percussive', or 'harmonic'.
        target_peaks (int, optional): Approximate number of events desired.
                                      Currently only implemented for 'amplitude' mode.
        initial_threshold_percentile (float): Starting percentile for 'amplitude' mode threshold.
        min_gap_seconds (float): Minimum time allowed between detected peaks in 'amplitude' mode.
        hpss_margin (float): Margin parameter for HPSS separation.

    Returns:
        list[float]: A list of timestamps (in seconds) corresponding to detected events.
                     Returns an empty list if loading fails or no events are found.
    """
    try:
        print(f"Loading audio for peak detection: {audio_path}")
        y, sr = librosa.load(audio_path, sr=None) # Load with original sample rate
    except Exception as e:
        # Catch specific Numba/NumPy errors if needed
        if "Numba needs NumPy" in str(e):
             print(f"\n------\nError: {e}")
             print("Please ensure NumPy version is compatible with Numba (usually NumPy < 2.2 for recent Numba).")
             print("Try: uv pip install \"numpy<2.2\"\n------")
        else:
            print(f"Error loading audio file {audio_path}: {e}")
        return []

    peak_times = []

    if mode == 'amplitude':
        print(f"Detecting peaks using '{mode}' mode.")
        if target_peaks is None or target_peaks <= 0:
            print(f" Using fixed threshold percentile={initial_threshold_percentile:.1f}%, min_gap={min_gap_seconds}s")
            peak_times = _find_peaks_amplitude(y, sr, initial_threshold_percentile, min_gap_seconds)
        else:
            # Adaptive Thresholding for amplitude mode
            print(f" Attempting to find approximately {target_peaks} peaks (amplitude mode)...")
            # Simplified adaptive logic - might need refinement
            current_threshold = initial_threshold_percentile
            best_peaks = []
            min_diff = float('inf')
            low_thresh, high_thresh, step = 90.0, 99.9, 0.5 # Adjust range/step as needed

            for attempt_thresh in np.arange(high_thresh, low_thresh - step, -step):
                current_peaks = _find_peaks_amplitude(y, sr, attempt_thresh, min_gap_seconds)
                diff = abs(len(current_peaks) - target_peaks)
                if diff <= min_diff: # Prioritize getting closer or equal
                    min_diff = diff
                    best_peaks = current_peaks
                    # print(f"  Trying threshold {attempt_thresh:.1f}% -> {len(current_peaks)} peaks (best diff: {min_diff})")
                if diff == 0: break # Stop if exact match found

            peak_times = best_peaks
            print(f" Selected {len(peak_times)} peaks (target was {target_peaks}) using adaptive threshold.")
            if not peak_times: print(" Warning: No peaks found even with adaptive thresholding.")

    elif mode in ['percussive', 'harmonic']:
        print(f"Detecting onsets using '{mode}' mode (HPSS separation).")
        if target_peaks is not None:
            print(" Warning: target_peaks is not currently used for 'percussive' or 'harmonic' modes.")

        # Perform HPSS
        print(" Performing HPSS...")
        try:
            y_harmonic, y_percussive = librosa.effects.hpss(y, margin=hpss_margin)
        except Exception as hpss_e:
             print(f" Error during HPSS: {hpss_e}. Falling back to full audio onset detection.")
             # Fallback to onset detection on the original signal if HPSS fails
             peak_times = _find_onsets(y, sr)
             mode = 'onset_fallback' # Indicate fallback
        else:
            print(" HPSS complete.")
            # Select component and find onsets
            component = y_percussive if mode == 'percussive' else y_harmonic
            print(f" Finding onsets in '{mode}' component...")
            peak_times = _find_onsets(component, sr)

    else:
        print(f"Error: Unknown detection mode '{mode}'. Use 'amplitude', 'percussive', or 'harmonic'.")
        return []

    print(f"Found {len(peak_times)} events using mode '{mode}'.")
    return sorted(list(np.unique(peak_times))) # Ensure sorted and unique


def format_timestamp(t_seconds):
    """Formats seconds into MM:SS.mmm"""
    if t_seconds < 0: t_seconds = 0
    minutes = int(t_seconds // 60)
    seconds = t_seconds % 60
    return f"{minutes:02d}:{seconds:06.3f}"

def save_timestamps_to_csv(peak_times_seconds, output_csv_path):
    """Saves a list of peak times (in seconds) to a CSV file."""
    try:
        output_dir = os.path.dirname(output_csv_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Event Timestamps (seconds)', 'Event Timestamps (MM:SS.mmm)'])
            for peak_sec in peak_times_seconds:
                writer.writerow([f"{peak_sec:.3f}", format_timestamp(peak_sec)])
        print(f"Event timestamps saved to: {output_csv_path}")
        return True
    except Exception as e:
        print(f"Error saving timestamps to {output_csv_path}: {e}")
        return False

# --- Example Usage (for testing this module directly) ---
if __name__ == "__main__":
    test_audio = 'path/to/your/test_audio.mp3' # <--- IMPORTANT: Set a real path here for testing

    if os.path.exists(test_audio):
        print("\n--- Testing Amplitude Mode ---")
        peaks_amp = find_peaks(test_audio, mode='amplitude', target_peaks=30)
        if peaks_amp: save_timestamps_to_csv(peaks_amp, "test_peaks_amplitude.csv")

        print("\n--- Testing Percussive Mode ---")
        peaks_perc = find_peaks(test_audio, mode='percussive')
        if peaks_perc: save_timestamps_to_csv(peaks_perc, "test_peaks_percussive.csv")

        print("\n--- Testing Harmonic Mode ---")
        peaks_harm = find_peaks(test_audio, mode='harmonic')
        if peaks_harm: save_timestamps_to_csv(peaks_harm, "test_peaks_harmonic.csv")

    else:
        print(f"Test audio file not found: {test_audio}. Cannot run example usage.")