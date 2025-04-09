# audio_responsive_editor/audio_processing/peak_detection.py

import librosa
import numpy as np
import csv
import os
from datetime import datetime

# --- Constants ---
DEFAULT_THRESHOLD_PERCENTILE = 99.7
DEFAULT_MIN_TIME_GAP_SECONDS = 0.5 # Minimum time between detected peaks (for 'peak' method)

# --- Helper: Peak Detection (Original Logic) ---
def _find_peaks_from_amplitude(y, sr, percentile_threshold, min_gap_seconds, target_peaks=None):
    """Helper function to find peaks with specific parameters."""
    if y is None or len(y) == 0: return []
    amplitude_envelope = np.abs(y)
    if len(amplitude_envelope) == 0: return []

    peak_times = [] # Initialize

    if target_peaks is None or target_peaks <= 0:
        # Simple detection using default/initial parameters
        print(f"Detecting peaks with threshold percentile={percentile_threshold:.1f}%, min_gap={min_gap_seconds}s")
        actual_threshold = np.percentile(amplitude_envelope, percentile_threshold) if not np.all(amplitude_envelope == amplitude_envelope[0]) else np.max(amplitude_envelope) + 1
        peak_indices = np.where(amplitude_envelope > actual_threshold)[0]
    else:
        # --- Adaptive Thresholding (Heuristic for Peaks) ---
        print(f"Attempting to find approximately {target_peaks} peaks...")
        best_peak_indices = []
        min_diff = float('inf')
        low_thresh = 95.0
        high_thresh = 99.9
        step = 0.5

        for attempt_thresh_percentile in np.arange(high_thresh, low_thresh - step, -step):
             actual_threshold = np.percentile(amplitude_envelope, attempt_thresh_percentile) if not np.all(amplitude_envelope == amplitude_envelope[0]) else np.max(amplitude_envelope) + 1
             current_peak_indices = np.where(amplitude_envelope > actual_threshold)[0]
             diff = abs(len(current_peak_indices) - target_peaks)

             # Refine check based on *merged* peaks for better accuracy? (More complex)
             # For now, use raw peak count for threshold adjustment heuristic.

             if diff == 0:
                 best_peak_indices = current_peak_indices
                 print(f"  Found close match ({len(best_peak_indices)} raw peaks) with threshold {attempt_thresh_percentile:.1f}%")
                 break
             if diff < min_diff:
                 min_diff = diff
                 best_peak_indices = current_peak_indices

        peak_indices = best_peak_indices
        print(f"Selected threshold yielding {len(peak_indices)} raw peaks (target was {target_peaks}) using adaptive threshold.")
        if not peak_indices.any():
             print("Warning: No peaks found even with adaptive thresholding.")
             return []


    # --- Merge close peaks (Common logic for peak method) ---
    if len(peak_indices) > 0:
        min_samples_gap = int(min_gap_seconds * sr)
        merged_peak_indices = []
        last_peak_index = -min_samples_gap

        # Sort raw indices first
        sorted_peak_indices = np.sort(peak_indices)

        for current_peak_index in sorted_peak_indices:
            if current_peak_index - last_peak_index >= min_samples_gap:
                # Simple merging: just take the first peak in a cluster after the gap
                merged_peak_indices.append(current_peak_index)
                last_peak_index = current_peak_index

        peak_times = librosa.samples_to_time(np.array(merged_peak_indices), sr=sr)
    else:
         peak_times = []


    return peak_times

# --- Helper: Onset Detection ---
def _find_onsets(y, sr, target_peaks=None):
    """Helper function to find onsets using librosa."""
    if y is None or len(y) == 0: return []
    # Note: target_peaks isn't directly used here, but could potentially tune onset parameters
    if target_peaks is not None:
        print(f"Warning: target_peaks argument ignored for 'onset' detection method in this implementation.")

    print("Detecting onsets...")
    # units='time' gives timestamps directly
    # backtrack=True attempts to place the marker closer to the start of the event
    onset_times = librosa.onset.onset_detect(y=y, sr=sr, units='time', backtrack=True)
    return onset_times.tolist() # Convert numpy array to list


# --- Main Function (Renamed and Updated) ---
def find_cut_times(audio_path, method='peak', target_peaks=None, initial_threshold_percentile=DEFAULT_THRESHOLD_PERCENTILE, min_gap_seconds=DEFAULT_MIN_TIME_GAP_SECONDS):
    """
    Analyzes an audio file to find potential cut points using the specified method.

    Args:
        audio_path (str): Path to the audio file.
        method (str): Detection method ('peak' or 'onset').
        target_peaks (int, optional): Approximate number of cuts desired (primarily for 'peak' method).
        initial_threshold_percentile (float): Starting percentile for 'peak' detection.
        min_gap_seconds (float): Minimum time allowed between detected 'peak' events.

    Returns:
        list[float]: A sorted list of timestamps (in seconds) corresponding to detected events.
                     Returns an empty list if loading fails or no events are found.
    """
    try:
        print(f"Loading audio for '{method}' detection: {audio_path}")
        y, sr = librosa.load(audio_path, sr=None) # Load with original sample rate
    except Exception as e:
        # Catch specific librosa/soundfile errors if possible, or numba errors
        # Check for the NumPy version error specifically
        if "Numba needs NumPy" in str(e):
             print(f"\n--- NumPy Version Error ---")
             print(f"Error loading audio: {e}")
             print(f"Numba (used by librosa) requires NumPy 2.1 or less.")
             print(f"Please ensure NumPy is downgraded, e.g., run: uv pip install \"numpy<2.2\"")
             print(f"--- End Error ---")
        else:
             print(f"Error loading audio file {audio_path}: {e}")
        return [] # Return empty list on error

    cut_times = []
    if method == 'peak':
        cut_times = _find_peaks_from_amplitude(y, sr, initial_threshold_percentile, min_gap_seconds, target_peaks)
    elif method == 'onset':
        cut_times = _find_onsets(y, sr, target_peaks)
    # Add elif method == 'beat': ... etc. for other methods later
    else:
        print(f"Warning: Unknown detection method '{method}'. Using default 'peak'.")
        cut_times = _find_peaks_from_amplitude(y, sr, initial_threshold_percentile, min_gap_seconds, target_peaks)

    print(f"Found {len(cut_times)} potential cut points using '{method}' method.")
    return sorted(list(cut_times)) # Ensure sorted output


# --- Utility Functions (Unchanged) ---
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
            writer.writerow(['Timestamp (seconds)', 'Timestamp (MM:SS.mmm)']) # Updated header
            for peak_sec in peak_times_seconds:
                writer.writerow([f"{peak_sec:.3f}", format_timestamp(peak_sec)])
        print(f"Timestamps saved to: {output_csv_path}")
        return True
    except Exception as e:
        print(f"Error saving timestamps to {output_csv_path}: {e}")
        return False

# --- Example Usage (Updated for testing this module directly) ---
if __name__ == "__main__":
    # Replace with an actual audio file path if needed
    test_audio = 'path/to/your/test_audio.mp3' # <--- IMPORTANT: Set a real path here for testing
    temp_dir = "temp_peak_detection_test"
    os.makedirs(temp_dir, exist_ok=True)


    if os.path.exists(test_audio):
        print("\n--- Testing 'peak' method (default) ---")
        times_peak = find_cut_times(test_audio, method='peak')
        if times_peak:
            save_timestamps_to_csv(times_peak, os.path.join(temp_dir,"test_times_peak.csv"))

        print("\n--- Testing 'peak' method with target ---")
        target = 30
        times_peak_targeted = find_cut_times(test_audio, method='peak', target_peaks=target)
        if times_peak_targeted:
             save_timestamps_to_csv(times_peak_targeted, os.path.join(temp_dir, f"test_times_peak_target_{target}.csv"))

        print("\n--- Testing 'onset' method ---")
        times_onset = find_cut_times(test_audio, method='onset')
        if times_onset:
            save_timestamps_to_csv(times_onset, os.path.join(temp_dir, "test_times_onset.csv"))

        print(f"\nTest outputs saved in: {temp_dir}")

    else:
        print(f"Test audio file not found: {test_audio}. Cannot run example usage.")