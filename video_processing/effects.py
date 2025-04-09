# audio_responsive_editor/video_processing/effects.py

import cv2
import numpy as np
import os
from utils import file_utils # Import the utility for sorting

# --- Duotone Specific ---
# Keep the original DUOTONE_PAIRS list (copied from src/duotone.py)
DUOTONE_PAIRS = [
    ((0, 255, 255), (255, 0, 255)), ((255, 0, 255), (0, 255, 0)),
    ((0, 0, 255), (0, 255, 255)), ((255, 0, 0), (0, 255, 255)),
    ((0, 255, 0), (255, 0, 255)), ((255, 0, 0), (0, 255, 0)),
    ((0, 255, 255), (255, 0, 0)), ((255, 0, 255), (255, 255, 0)),
    ((0, 128, 255), (0, 255, 0)), ((255, 0, 128), (0, 255, 255)),
    # ... (include all pairs from your original file) ...
    ((0, 0, 255), (0, 255, 0)), ((255, 0, 255), (255, 255, 0)),
    ((0, 165, 255), (0, 255, 255)), ((255, 0, 0), (0, 255, 128)),
    ((128, 0, 255), (0, 255, 0)), ((0, 0, 255), (0, 255, 255)),
    ((255, 50, 0), (0, 255, 255)), ((255, 0, 255), (0, 255, 128)),
    ((0, 255, 255), (255, 0, 128)), ((255, 0, 0), (0, 255, 191)),
    ((0, 127, 255), (0, 255, 127)), ((255, 0, 255), (0, 255, 255)),
    ((0, 255, 0), (0, 0, 255)), ((255, 0, 127), (0, 255, 255)),
    ((0, 140, 255), (255, 0, 255)), ((255, 0, 140), (0, 255, 0)),
    ((0, 255, 255), (255, 0, 140)), ((255, 0, 255), (0, 255, 140)),
    ((140, 255, 0), (0, 0, 255)), ((255, 140, 0), (0, 255, 255))
]

def apply_duotone_frame(frame, color1_bgr, color2_bgr):
    """Apply duotone effect to a single frame."""
    if frame is None:
        return None
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Normalize grayscale image to range [0, 1]
        normalized = gray.astype(np.float32) / 255.0

        # Create placeholder for the color result (float32 for precision)
        result_float = np.zeros_like(frame, dtype=np.float32)

        # Apply the duotone mapping for each BGR channel
        for i in range(3): # 0=Blue, 1=Green, 2=Red
            result_float[:,:,i] = normalized * color1_bgr[i] + (1.0 - normalized) * color2_bgr[i]

        # Clip values to [0, 255] and convert back to uint8
        result_uint8 = np.clip(result_float, 0, 255).astype(np.uint8)
        return result_uint8
    except cv2.error as e:
        print(f" OpenCV error applying duotone: {e}")
        return frame # Return original frame on error
    except Exception as e:
        print(f" Error applying duotone: {e}")
        return frame # Return original frame on error


# --- Placeholder for other effects ---
# def apply_pixelate_frame(frame, scale=0.1):
#     if frame is None: return None
#     h, w = frame.shape[:2]
#     small_h, small_w = int(h * scale), int(w * scale)
#     if small_h <= 0 or small_w <=0 : return frame # Avoid zero size
#     temp = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
#     return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

# --- Main Effect Application Function ---

def apply_effects_to_segments(input_segments_folder, output_effects_folder, effects_list=['duotone']):
    """
    Reads video segments, applies selected effects, and saves new segments.

    Args:
        input_segments_folder (str): Folder containing the raw video segments.
        output_effects_folder (str): Folder to save the processed segments.
        effects_list (list[str]): List of effect names to apply (e.g., ['duotone', 'pixelate']).

    Returns:
        bool: True if processing completes (even with skipped files), False on major error.
    """
    if not os.path.isdir(input_segments_folder):
        print(f"Error: Input segments folder not found: {input_segments_folder}")
        return False

    os.makedirs(output_effects_folder, exist_ok=True)

    video_files = [f for f in os.listdir(input_segments_folder) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    if not video_files:
        print(f"Warning: No video files found in {input_segments_folder}")
        return True # Not an error, just nothing to do

    # IMPORTANT: Sort segments chronologically using the utility function
    segment_paths = [os.path.join(input_segments_folder, f) for f in video_files]
    sorted_segment_paths = file_utils.sort_video_segments(segment_paths)

    print(f"Applying effects {effects_list} to {len(sorted_segment_paths)} segments...")

    # --- Effect Processing Loop ---
    duotone_color_index = 0 # Keep track for cycling colors
    processed_count = 0

    for idx, segment_path in enumerate(sorted_segment_paths):
        base_filename = os.path.basename(segment_path)
        output_path = os.path.join(output_effects_folder, base_filename) # Keep original name

        print(f"\nProcessing segment {idx+1}/{len(sorted_segment_paths)}: {base_filename}")

        cap = cv2.VideoCapture(segment_path)
        if not cap.isOpened():
            print(f"  Warning: Could not open segment {base_filename}, skipping...")
            continue

        # Get properties for VideoWriter
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0 or width <= 0 or height <= 0:
             print(f"  Warning: Invalid video properties for {base_filename} (FPS:{fps}, W:{width}, H:{height}). Skipping.")
             cap.release()
             continue


        # Setup writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
             print(f"  Error: Could not open VideoWriter for {output_path}. Skipping.")
             cap.release()
             continue

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break # End of segment

            processed_frame = frame # Start with the original frame

            # --- Apply effects sequentially based on effects_list ---
            if 'duotone' in effects_list:
                # Cycle through duotone pairs for each segment (excluding the very first segment perhaps?)
                # Let's apply duotone to all segments for now, cycling colors
                color1, color2 = DUOTONE_PAIRS[duotone_color_index % len(DUOTONE_PAIRS)]
                processed_frame = apply_duotone_frame(processed_frame, color1, color2)
                # Note: color index increments per segment below, not per frame

            # --- Add other effects here ---
            # if 'pixelate' in effects_list:
            #     processed_frame = apply_pixelate_frame(processed_frame, scale=0.1)

            if processed_frame is None:
                 print(f"  Warning: Frame processing returned None for frame {frame_count}. Using original.")
                 processed_frame = frame # Fallback to original if effect failed badly

            out.write(processed_frame)
            frame_count += 1

        # Increment duotone color index *after* processing all frames of a segment
        if 'duotone' in effects_list:
             duotone_color_index += 1

        cap.release()
        out.release()
        print(f"  Finished processing {frame_count} frames -> {os.path.basename(output_path)}")
        processed_count += 1

    print(f"\nEffect application complete. Processed {processed_count} segments saved in: {output_effects_folder}")
    return True

# --- Example Usage ---
if __name__ == "__main__":
    # Assumes you have run cutter.py first and have segments
    test_input_dir = "temp_cut_test" # Use the output dir from cutter.py test
    test_output_dir = "temp_effects_test"
    test_effects = ['duotone'] # Test only duotone

    if os.path.isdir(test_input_dir):
        print(f"Testing effects {test_effects}...")
        success = apply_effects_to_segments(test_input_dir, test_output_dir, test_effects)
        if success:
            print(f"\nTest effects finished. Check folder: {test_output_dir}")
            # import shutil
            # shutil.rmtree(test_output_dir)
        else:
            print("\nTest effects failed.")
    else:
        print(f"Input directory for effects test not found: {test_input_dir}. Cannot run example.")
        print("(Run the cutter.py example first)")