# audio_responsive_editor/main.py

# --- Start of temporary diagnostic code ---
# Note: If running directly worked, you might be able to remove this
# But leave it for now just in case. If you go back to trying `python -m ...`
# and it fails without this, then the path issue is still lurking.
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
import traceback

# Import functions from our modules
from audio_processing import peak_detection, audio_utils
from video_processing import cutter, effects, combiner
# from utils import file_utils


def main():
    parser = argparse.ArgumentParser(description="Audio-Responsive Video Editor")

    parser.add_argument("-i", "--input", required=True, help="Path to the input video file.")
    parser.add_argument("-o", "--output", required=True, help="Path for the output video file.")
    parser.add_argument("-a", "--audio", default=None, help="Optional path to a separate audio file to use instead of the video's audio.")
    # --- Updated Argument ---
    parser.add_argument(
        "-m", "--method", # Short flag changed to avoid conflict if we add more
        choices=['peak', 'onset'], # Add 'beat', 'rms' later
        default='peak',
        dest='detection_method', # Store in 'detection_method' attribute
        help="Method for detecting cut points ('peak', 'onset'). Default: peak"
    )
    # --- Keep target cuts argument ---
    parser.add_argument("-t", "--target-cuts", type=int, default=None, help="Approximate number of cuts desired (primarily used by 'peak' method).")
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
    # (Using context manager approach again for robustness)
    with tempfile.TemporaryDirectory(prefix="are_") as base_temp_dir_managed:
        if args.keep_temp:
             base_temp_dir = tempfile.mkdtemp(prefix="are_persistent_")
             print(f"Temporary files will be kept in: {base_temp_dir}")
        else:
             base_temp_dir = base_temp_dir_managed # Use the managed temp dir

        audio_file_path = None
        cuts_dir = os.path.join(base_temp_dir, "1_cuts")
        effects_dir = os.path.join(base_temp_dir, "2_effects")
        os.makedirs(cuts_dir, exist_ok=True)
        os.makedirs(effects_dir, exist_ok=True)
        pipeline_successful = False

        try:
            # --- Pipeline Steps ---
            print(f"Starting pipeline with detection method: '{args.detection_method}'...")

            # 1. Prepare Audio & Detect Cuts
            print("\nStep 1: Preparing Audio & Detecting Cuts...")
            audio_file_path = audio_utils.prepare_audio(args.input, args.audio, base_temp_dir)
            if not audio_file_path:
                 raise RuntimeError("Failed to prepare audio file.")

            # --- Updated function call ---
            cut_times = peak_detection.find_cut_times(
                audio_file_path,
                method=args.detection_method, # Pass the chosen method
                target_peaks=args.target_cuts # Pass target cuts (might be ignored by method)
            )
            print(f"Detected {len(cut_times)} potential cut points.")
            # Save timestamps regardless of method
            peak_detection.save_timestamps_to_csv(cut_times, os.path.join(base_temp_dir, f"detected_{args.detection_method}_times.csv"))

            # 2. Cut Video
            print("\nStep 2: Cutting Video...")
            if not cutter.cut_video_at_times(args.input, cut_times, cuts_dir):
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
            # Cleanup logic is slightly simplified here - relies on context manager for non-kept temp dirs
            if args.keep_temp:
                 if pipeline_successful:
                     print(f"Temporary files kept at: {base_temp_dir}")
                 else:
                     # Check if base_temp_dir was actually created in the keep_temp case
                     if 'base_temp_dir' in locals() and os.path.isdir(base_temp_dir):
                         print(f"Pipeline failed. Temporary files kept for debugging at: {base_temp_dir}")
            # If not keeping temp, the 'with' statement handles cleanup automatically


if __name__ == "__main__":
    main()