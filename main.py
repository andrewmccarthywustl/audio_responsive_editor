# audio_responsive_editor/main.py

# --- Start of temporary diagnostic code ---
# Note: If running directly (python main.py...) works reliably now,
# you might consider keeping this sys.path modification.
# If running with `python -m ...` starts working later (e.g., after fixing
# environment issues), you might remove this.
import sys
import os

script_path = os.path.abspath(__file__)
package_dir = os.path.dirname(script_path)
parent_dir = os.path.dirname(package_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    # print(f"DEBUG: Temporarily added {parent_dir} to sys.path")
# print(f"DEBUG: sys.path is now: {sys.path}")
# --- End of temporary diagnostic code ---

import argparse
import tempfile
import shutil
import traceback # Import traceback for better error reporting

# Import functions from our modules
from audio_processing import peak_detection, audio_utils
from video_processing import cutter, effects, combiner
# from utils import file_utils

def main():
    parser = argparse.ArgumentParser(description="Audio-Responsive Video Editor")

    parser.add_argument("-i", "--input", required=True, help="Path to the input video file.")
    parser.add_argument("-o", "--output", required=True, help="Path for the output video file.")
    parser.add_argument("-a", "--audio", default=None, help="Optional path to a separate audio file to use instead of the video's audio.")

    # --- Event Detection Arguments ---
    parser.add_argument("--detect-mode", default='amplitude', choices=['amplitude', 'percussive', 'harmonic'],
                        help="Method for detecting events: 'amplitude' (loudness peaks), "
                             "'percussive' (onsets in percussive component after HPSS), "
                             "'harmonic' (onsets in harmonic component after HPSS).")
    parser.add_argument("-t", "--target-cuts", type=int, default=None,
                        help="Approximate number of cuts desired (currently only affects 'amplitude' mode).")
    parser.add_argument("--min-gap", type=float, default=peak_detection.DEFAULT_MIN_TIME_GAP_SECONDS,
                         help=f"Minimum time gap between peaks in 'amplitude' mode (default: {peak_detection.DEFAULT_MIN_TIME_GAP_SECONDS}s).")
    parser.add_argument("--hpss-margin", type=float, default=peak_detection.DEFAULT_HPSS_MARGIN,
                         help=f"HPSS margin (larger values favor harmonic/percussive separation) (default: {peak_detection.DEFAULT_HPSS_MARGIN}).")
    # --- End Event Detection Arguments ---

    parser.add_argument("-e", "--effects", nargs='+', default=['duotone'], help="List of effects to apply (e.g., duotone). Default: duotone")
    parser.add_argument("-k", "--keep-temp", action='store_true', help="Keep temporary files after processing (for debugging).")

    args = parser.parse_args()

    # --- Validate Inputs ---
    if not os.path.exists(args.input):
        print(f"Error: Input video file not found: {args.input}")
        return
    if args.audio and not os.path.exists(args.audio):
        print(f"Error: Specified audio file not found: {args.audio}")
        return
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Setup Temp Dirs ---
    # Rest of the setup remains the same...
    temp_dir_context = tempfile.TemporaryDirectory(prefix="are_")
    if args.keep_temp:
        base_temp_dir = tempfile.mkdtemp(prefix="are_persistent_")
        print(f"Temporary files will be kept in: {base_temp_dir}")
    else:
        base_temp_dir = temp_dir_context.__enter__()

    audio_file_path = None
    cuts_dir = os.path.join(base_temp_dir, "1_cuts")
    effects_dir = os.path.join(base_temp_dir, "2_effects")
    os.makedirs(cuts_dir, exist_ok=True)
    os.makedirs(effects_dir, exist_ok=True)
    pipeline_successful = False

    try:
        # --- Pipeline Steps ---
        print("Starting pipeline...")

        # 1. Prepare Audio & Detect Peaks/Onsets
        print(f"\nStep 1: Preparing Audio & Detecting Events (mode: {args.detect_mode})...")
        audio_file_path = audio_utils.prepare_audio(args.input, args.audio, base_temp_dir)
        if not audio_file_path:
             raise RuntimeError("Failed to prepare audio file.")

        # Call find_peaks with the selected mode and relevant parameters
        event_times = peak_detection.find_peaks(
            audio_path=audio_file_path,
            mode=args.detect_mode,
            target_peaks=args.target_cuts,
            min_gap_seconds=args.min_gap,
            hpss_margin=args.hpss_margin
            # Pass initial_threshold_percentile if needed, but it's less relevant for onset modes
        )
        print(f"Detected {len(event_times)} potential cut points using mode '{args.detect_mode}'.")
        peak_detection.save_timestamps_to_csv(event_times, os.path.join(base_temp_dir, "detected_events.csv"))

        # 2. Cut Video
        print("\nStep 2: Cutting Video...")
        if not cutter.cut_video_at_times(args.input, event_times, cuts_dir):
             raise RuntimeError("Failed during video cutting stage.")

        # 3. Apply Effects
        print("\nStep 3: Applying Effects...")
        if not effects.apply_effects_to_segments(cuts_dir, effects_dir, args.effects):
             print("Warning: Effect application stage encountered issues, but attempting to continue.")

        # 4. Combine Video & Audio
        print("\nStep 4: Combining Final Video...")
        if not combiner.combine_segments_with_audio(effects_dir, audio_file_path, args.output):
             raise RuntimeError("Failed during final video combination stage.")

        print(f"\nPipeline finished successfully! Output saved to: {args.output}")
        pipeline_successful = True

    except Exception as e:
        print(f"\n--- Pipeline Error ---")
        print(f"Error: {e}")
        print("Traceback:")
        traceback.print_exc()
        print("--- End Traceback ---")

    finally:
        # --- Cleanup ---
        # Cleanup logic remains the same...
        if not args.keep_temp:
            if 'temp_dir_context' in locals() and base_temp_dir.startswith(tempfile.gettempdir()):
                print(f"Cleaning up temporary directory: {base_temp_dir}")
                try:
                    temp_dir_context.__exit__(None, None, None)
                except Exception as cleanup_e:
                    print(f"Warning: Error during automatic temp directory cleanup: {cleanup_e}")
            elif 'base_temp_dir' in locals() and not base_temp_dir.startswith(tempfile.gettempdir()):
                 pass
        elif pipeline_successful:
             print(f"Temporary files kept at: {base_temp_dir}")
        else:
             if 'base_temp_dir' in locals():
                 print(f"Pipeline failed. Temporary files kept for debugging at: {base_temp_dir}")


if __name__ == "__main__":
    main()