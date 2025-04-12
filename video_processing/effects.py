# audio_responsive_editor/video_processing/effects.py

import cv2
import numpy as np
import os
import functools
from abc import ABC, abstractmethod # For abstract base class
# Ensure utils package is accessible from video_processing
# If running effects.py directly, you might need path adjustments,
# but when run via main.py, this should work.
try:
    from utils import file_utils
except ImportError:
    # Simple fallback for direct execution testing if utils is one level up
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils import file_utils


# --- Core Infrastructure (Top of File) ---

# Registry to hold effect *classes*
EFFECT_REGISTRY = {}

def register_effect(name, requires_original_frame=False):
    """
    Decorator to register an effect class.

    Args:
        name (str): The name used to call the effect (e.g., 'duotone').
        requires_original_frame (bool): If True, the process method must accept
                                        'original_frame' as a keyword argument.
    """
    def decorator(cls):
        if not issubclass(cls, Effect):
            raise TypeError(f"Class {cls.__name__} must inherit from Effect")
        cls.requires_original_frame = requires_original_frame
        EFFECT_REGISTRY[name.lower()] = cls
        # print(f"Registered effect: '{name.lower()}' -> {cls.__name__}") # Optional debug print
        return cls
    return decorator

class Effect(ABC):
    """Abstract base class for all effects."""
    requires_original_frame = False # Class attribute default

    @abstractmethod
    def process(self, frame, segment_index, original_frame=None):
        """
        Process a single frame.

        Args:
            frame (np.ndarray): The input frame (potentially processed by previous effects).
            segment_index (int): The index of the current video segment (0-based).
                                 Used by effects to vary parameters across segments.
            original_frame (np.ndarray, optional): The original, unaltered frame for this step.
                                                   Provided only if requires_original_frame is True.

        Returns:
            np.ndarray: The processed frame. Should return the input frame on error if possible.
        """
        pass

def apply_effects_to_segments(input_segments_folder, output_effects_folder, effects_list=['duotone']):
    """
    Reads video segments, applies selected effects using registered classes,
    and saves new segments. Effects manage their own parameter variations based on segment index.
    """
    if not os.path.isdir(input_segments_folder):
        print(f"Error: Input segments folder not found: {input_segments_folder}")
        return False

    os.makedirs(output_effects_folder, exist_ok=True)

    video_files = [f for f in os.listdir(input_segments_folder) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    if not video_files:
        print(f"Warning: No video files found in {input_segments_folder}")
        return True # Nothing to process is not necessarily an error

    segment_paths = [os.path.join(input_segments_folder, f) for f in video_files]
    try:
        # Use the utility function for robust sorting
        sorted_segment_paths = file_utils.sort_video_segments(segment_paths)
    except Exception as e:
         print(f"Error sorting video segments: {e}. Using basic sort.")
         # Fallback to basic alphabetical sort if utility fails
         sorted_segment_paths = sorted(segment_paths)


    print(f"Applying effects {effects_list} to {len(sorted_segment_paths)} segments...")
    print(f"Available effects: {list(EFFECT_REGISTRY.keys())}")

    # --- Instantiate Effects Once ---
    active_effects = []
    for effect_name in effects_list:
         effect_name_lower = effect_name.lower()
         if effect_name_lower in EFFECT_REGISTRY:
              effect_cls = EFFECT_REGISTRY[effect_name_lower]
              active_effects.append(effect_cls()) # Instantiate
         else:
              print(f" Warning: Effect '{effect_name}' not found in registry. Skipping.")

    if not active_effects:
        print("Error: No valid effects selected or found. Aborting effect application.")
        # Decide if this should return False or just process without effects
        # Let's return True as no processing was *needed* according to the (empty) valid list
        # Copying files without effects might be desired? Or maybe False is better. Let's stick with False.
        return False

    processed_count = 0
    # --- Process Segments ---
    for segment_idx, segment_path in enumerate(sorted_segment_paths):
        base_filename = os.path.basename(segment_path)
        output_path = os.path.join(output_effects_folder, base_filename)

        print(f"\nProcessing segment {segment_idx+1}/{len(sorted_segment_paths)}: {base_filename} (Index: {segment_idx})")

        cap = cv2.VideoCapture(segment_path)
        if not cap.isOpened():
            print(f" Warning: Could not open segment {base_filename}. Skipping.")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Basic validation for video properties
        if fps <= 0 or width <= 0 or height <= 0:
            print(f" Warning: Invalid video properties (fps:{fps}, w:{width}, h:{height}) for {base_filename}. Skipping.")
            cap.release()
            continue

        # Use a common codec like mp4v (H.264)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            print(f" Warning: Could not create VideoWriter for {output_path}. Check codec and permissions. Skipping segment.")
            cap.release()
            continue

        frame_num = 0
        while True:
            ret, original_frame = cap.read()
            if not ret:
                break # End of segment
            # Ensure frame is valid
            if original_frame is None or original_frame.size == 0:
                 print(f"  Warning: Read invalid frame (num {frame_num}) in {base_filename}. Skipping frame.")
                 continue


            processed_frame = original_frame.copy() # Work on a copy

            # --- Apply effects sequentially ---
            for effect_instance in active_effects:
                try:
                    # Prepare arguments for the process method
                    process_args = {
                        "frame": processed_frame,
                        "segment_index": segment_idx
                    }
                    if effect_instance.requires_original_frame:
                        process_args["original_frame"] = original_frame

                    # Call the effect's process method
                    processed_frame = effect_instance.process(**process_args)

                    # Handle case where effect might mistakenly return None
                    if processed_frame is None:
                         print(f"  Warning: Effect {type(effect_instance).__name__} returned None for frame {frame_num}. Reverting to frame state before this effect.")
                         # Fallback: Use the frame state *before* this effect was called.
                         # This requires getting the frame from process_args again.
                         processed_frame = process_args["frame"]


                except Exception as e:
                    print(f" Error applying effect {type(effect_instance).__name__} on frame {frame_num}: {e}")
                    import traceback
                    traceback.print_exc() # Print detailed traceback for debugging
                    # Continue with the frame state before the error occurred
                    processed_frame = process_args.get("frame", original_frame) # Fallback robustly
                    pass # Allow processing to continue to next effect/frame if desired


            # --- Write the final frame ---
            if processed_frame is not None and processed_frame.size > 0:
                out.write(processed_frame)
            else:
                 print(f"  Error: Final processed frame for frame {frame_num} is invalid. Writing original frame instead.")
                 out.write(original_frame) # Fallback write if processing failed badly

            frame_num += 1

        # --- Release resources for this segment ---
        cap.release()
        out.release()
        print(f"  Finished processing {frame_num} frames -> {os.path.basename(output_path)}")
        processed_count += 1

    print(f"\nEffect application complete. Processed {processed_count} segments saved in: {output_effects_folder}")
    return True


# --- Example Usage Guard (Moved Up) ---
# This block only *runs* if the script is executed directly.
# By the time it runs, Python will have processed the entire file,
# including all effect class definitions and registrations below.
if __name__ == "__main__":
    print("--- Running effects.py directly for testing ---")

    # --- Test Setup Code ---
    # Use unique names to avoid conflicts if other tests run
    test_input_dir = "temp_effects_module_input"
    test_output_dir = "temp_effects_module_output"
    print(f"Test Input Directory: {os.path.abspath(test_input_dir)}")
    print(f"Test Output Directory: {os.path.abspath(test_output_dir)}")

    os.makedirs(test_input_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    # Clean output directory before test
    print(f"Cleaning test output directory: {test_output_dir}")
    for f in os.listdir(test_output_dir):
        try:
            os.remove(os.path.join(test_output_dir, f))
        except Exception as e:
            print(f"  Warning: Could not remove file {f}: {e}")


    # Create a few dummy small video files (e.g., 10 frames each)
    width, height, fps = 128, 72, 30 # Slightly larger test videos
    num_segments = 3
    frames_per_segment = 10

    print(f"Creating {num_segments} dummy video segments...")
    for i in range(num_segments):
        # Use file_utils naming convention if possible, otherwise basic
        try:
            # Assuming timestamp is just sequence for testing
            dummy_vid_path = os.path.join(test_input_dir, file_utils.get_segment_filename(i, float(i), float(i+1)))
        except AttributeError: # Fallback if get_segment_filename isn't in file_utils or signature changed
             dummy_vid_path = os.path.join(test_input_dir, f"segment_{i:04d}.mp4")

        if not os.path.exists(dummy_vid_path) or True: # Overwrite for fresh test data
             # print(f"  Creating dummy video: {os.path.basename(dummy_vid_path)}")
             fourcc = cv2.VideoWriter_fourcc(*'mp4v') # H.264
             writer = cv2.VideoWriter(dummy_vid_path, fourcc, fps, (width, height))
             if not writer.isOpened():
                 print(f"  ERROR: Failed to open VideoWriter for {dummy_vid_path}")
                 continue

             for j in range(frames_per_segment):
                 # Create a simple frame that changes slightly
                 frame = np.zeros((height, width, 3), dtype=np.uint8)
                 # Vary color based on segment and frame index
                 color = (
                     (i * 70 + j * 15 + 50) % 256,
                     (i * 50 + j * 5 + 100) % 256,
                     (i * 90 + j * 10 + 30) % 256
                 )
                 # Add text indicating segment and frame
                 cv2.putText(frame, f"Seg:{i} Frm:{j}", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                 # Add a moving rectangle
                 rect_x = (j * (width // frames_per_segment)) % (width - 20)
                 cv2.rectangle(frame, (rect_x, 10), (rect_x + 20, 30), (255, 255, 255), -1)

                 writer.write(frame)
             writer.release()
             # print(f"  Finished writing {dummy_vid_path}")


    # --- Test Execution ---
    # Select effects to test:
    # test_effects_to_run = ['duotone', 'hue_rotate']
    # test_effects_to_run = ['colored_edges_only']
    # test_effects_to_run = ['hue_rotate', 'edge_color_cycle']
    test_effects_to_run = ['duotone', 'edge_color_cycle'] # Test combining standard effects


    # Check if input dir exists and has files before running the test
    segment_files = []
    if os.path.isdir(test_input_dir):
        segment_files = [os.path.join(test_input_dir, f) for f in os.listdir(test_input_dir) if f.lower().endswith('.mp4')]

    if segment_files: # Check if the list is not empty
        print(f"\n--- Running Effects Test ---")
        print(f"Input Dir: {test_input_dir}")
        print(f"Output Dir: {test_output_dir}")
        print(f"Effects to Apply: {test_effects_to_run}")
        print(f"Found {len(segment_files)} segment files.")

        # *** Crucially, apply_effects_to_segments is CALLED here ***
        # By this point, all effect classes below have been defined and registered.
        success = apply_effects_to_segments(test_input_dir, test_output_dir, test_effects_to_run)

        if success:
            print(f"\nTest effects finished successfully. Check folder: {test_output_dir}")
        else:
            print("\nTest effects application reported failure.")
    else:
        print(f"Input directory for effects test ('{test_input_dir}') not found or is empty. Cannot run example.")

    print("--- Finished effects.py direct execution test ---")


# ==============================================================
# --- Constants ---
# (Defined after the __main__ block, but before effect classes that use them)
# ==============================================================

# Color Lists (BGR Format)
DUOTONE_PAIRS = [
    ((0, 255, 255), (255, 0, 255)),   # Yellow / Magenta
    ((255, 0, 255), (0, 255, 0)),     # Magenta / Lime
    ((0, 0, 255), (0, 255, 255)),     # Red / Yellow
    ((255, 0, 0), (0, 255, 255)),     # Blue / Yellow
    ((0, 255, 0), (255, 0, 255)),     # Lime / Magenta
    ((255, 0, 0), (0, 255, 0)),       # Blue / Lime
    ((0, 255, 255), (255, 0, 0)),     # Yellow / Blue
    ((255, 0, 255), (255, 255, 0)),   # Magenta / Cyan
    ((0, 128, 255), (0, 255, 0)),     # Orange / Lime
    ((255, 0, 128), (0, 255, 255)),   # Pink / Yellow
    ((0, 0, 255), (0, 255, 0)),       # Red / Lime
    ((255, 0, 255), (255, 255, 0)),   # Magenta / Cyan (Repeat?) -> Use different Cyan if needed
    ((0, 165, 255), (0, 255, 255)),   # Orange-Red / Yellow
    ((255, 0, 0), (0, 255, 128)),     # Blue / Spring Green
    ((128, 0, 255), (0, 255, 0)),     # Purple / Lime
    ((0, 0, 255), (255, 255, 0)),     # Red / Cyan
    ((255, 50, 0), (0, 255, 255)),    # Dodger Blue / Yellow
    ((255, 0, 255), (0, 255, 128)),   # Magenta / Spring Green
    ((0, 255, 255), (255, 0, 128)),   # Yellow / Pink
    ((255, 0, 0), (0, 255, 191)),     # Blue / Turquoise
    ((0, 127, 255), (0, 255, 127)),   # Orange / Aquamarine
    ((140, 255, 0), (0, 0, 255)),     # Chartreuse / Red
    ((255, 140, 0), (0, 255, 255))    # Dark Orange / Yellow
]

EDGE_COLORS = [
    (0, 255, 255),   # Yellow
    (255, 0, 255),   # Magenta
    (0, 255, 0),     # Lime
    (255, 255, 0),   # Cyan
    (0, 0, 255),     # Red
    (255, 0, 0),     # Blue
    (0, 165, 255),   # Orange
    (255, 255, 255), # White
    (0, 128, 255),   # Orange-Red
    (128, 0, 128),   # Purple
    (0, 128, 0),     # Green
    (128, 128, 0),   # Teal
]


# ==============================================================
# --- Specific Effect Class Definitions ---
# You can keep adding new effects below this line indefinitely.
# ==============================================================

@register_effect("duotone")
class DuotoneEffect(Effect):
    """Applies a two-tone color effect based on grayscale intensity."""
    def process(self, frame, segment_index, original_frame=None):
        if frame is None: return None
        try:
            # Determine colors based on segment index, cycling through the list
            pair_index = segment_index % len(DUOTONE_PAIRS)
            color1_bgr, color2_bgr = DUOTONE_PAIRS[pair_index]

            # Apply effect logic
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Normalize grayscale image to 0.0 - 1.0 for interpolation
            normalized = gray.astype(np.float32) / 255.0
            # Create an empty float frame for results
            result_float = np.zeros_like(frame, dtype=np.float32)
            # Interpolate between color1 (for light areas) and color2 (for dark areas)
            for i in range(3): # BGR channels
                result_float[:,:,i] = normalized * color1_bgr[i] + (1.0 - normalized) * color2_bgr[i]
            # Clip values to 0-255 and convert back to uint8
            result_uint8 = np.clip(result_float, 0, 255).astype(np.uint8)
            return result_uint8
        except cv2.error as e:
            print(f" OpenCV Error applying duotone (frame shape: {frame.shape}): {e}")
            return frame # Return input frame on specific OpenCV errors
        except Exception as e:
            print(f" General Error applying duotone: {e}")
            return frame # Return input frame on general errors

@register_effect("hue_rotate")
class HueRotateEffect(Effect):
    """Rotates the hue component of the image in HSV space."""
    HUE_STEP_ANGLE = 30 # Degrees of hue shift per segment

    def process(self, frame, segment_index, original_frame=None):
        if frame is None: return None
        try:
            # Determine total hue angle shift based on segment index
            total_angle_degrees = (segment_index * self.HUE_STEP_ANGLE) % 360
            # Convert degrees (0-360) to OpenCV's HSV hue range (0-179)
            # Ensure modulo arithmetic works correctly even for negative steps if HUE_STEP_ANGLE changes
            hue_shift_opencv = int(round((total_angle_degrees / 360.0) * 180)) % 180

            # Convert frame to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Apply hue shift (handling potential uint8 overflow)
            # Convert hue channel to a wider type for safe addition
            hue = hsv[:, :, 0].astype(np.int32)
            hue = (hue + hue_shift_opencv) % 180 # Modulo 180 ensures wrap-around
            # Convert back to uint8
            hsv[:, :, 0] = hue.astype(np.uint8)

            # Convert back to BGR
            rotated_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            return rotated_frame
        except cv2.error as e:
            print(f" OpenCV Error applying hue rotation (frame shape: {frame.shape}): {e}")
            return frame
        except Exception as e:
            print(f" General Error applying hue rotation: {e}")
            return frame

@register_effect("edge_color_cycle")
class EdgeColorCycleEffect(Effect):
    """Detects edges, colors them, and darkens non-edge areas."""
    # Canny edge detection thresholds
    CANNY_THRESHOLD1 = 50
    CANNY_THRESHOLD2 = 150
    # Factor to darken non-edge areas (0.0 to 1.0)
    DARKEN_FACTOR = 0.5

    def process(self, frame, segment_index, original_frame=None):
        if frame is None: return None
        try:
            # Determine edge color based on segment index
            color_index = segment_index % len(EDGE_COLORS)
            edge_color_bgr = EDGE_COLORS[color_index]

            # Detect edges
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Apply Gaussian blur slightly to reduce noise before Canny
            gray_blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(gray_blurred, self.CANNY_THRESHOLD1, self.CANNY_THRESHOLD2)

            # Create a mask of non-edge areas
            non_edge_mask = (edges == 0)

            # Create an image with only colored edges on black background
            color_edges_img = np.zeros_like(frame)
            color_edges_img[edges > 0] = edge_color_bgr

            # Darken the original frame where there are no edges
            darkened_frame = frame.copy().astype(np.float32) # Use float for multiplication
            darkened_frame[non_edge_mask] *= self.DARKEN_FACTOR
            darkened_frame = np.clip(darkened_frame, 0, 255).astype(np.uint8) # Convert back

            # Combine the darkened frame with the colored edges
            # Using add ensures colors are added, potentially brightening edges
            combined = cv2.add(darkened_frame, color_edges_img)
            # Alternatively, use addWeighted for blending:
            # combined = cv2.addWeighted(darkened_frame, 1.0, color_edges_img, 1.0, 0)

            return combined
        except cv2.error as e:
            print(f" OpenCV Error applying edge color cycle (frame shape: {frame.shape}): {e}")
            return frame
        except Exception as e:
            print(f" General Error applying edge color cycle: {e}")
            return frame

@register_effect("colored_edges_only", requires_original_frame=True)
class ColoredEdgesOnlyEffect(Effect):
    """Generates a frame with only colored edges on a black background."""
    # Canny edge detection thresholds
    CANNY_THRESHOLD1 = 50
    CANNY_THRESHOLD2 = 150

    def process(self, frame, segment_index, original_frame):
        # Note: 'frame' argument (potentially processed) is ignored here.
        if original_frame is None:
            print(" Error: ColoredEdgesOnlyEffect requires original_frame, but it was not provided.")
            # Return a black frame matching input 'frame' size if possible
            return np.zeros_like(frame) if frame is not None else None

        try:
            # Determine edge color based on segment index
            color_index = segment_index % len(EDGE_COLORS)
            edge_color_bgr = EDGE_COLORS[color_index]

            # Detect edges on the ORIGINAL frame
            gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
            gray_blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(gray_blurred, self.CANNY_THRESHOLD1, self.CANNY_THRESHOLD2)

            # Create a black background image
            color_edges_only_img = np.zeros_like(original_frame)
            # Apply the selected color only where edges were detected
            color_edges_only_img[edges > 0] = edge_color_bgr

            return color_edges_only_img
        except cv2.error as e:
            print(f" OpenCV Error generating colored edges (frame shape: {original_frame.shape}): {e}")
            return np.zeros_like(original_frame) # Return black frame on error
        except Exception as e:
            print(f" General Error generating colored edges: {e}")
            return np.zeros_like(original_frame) # Return black frame on error

@register_effect("smooth_colored_edges", requires_original_frame=True)
class SmoothColoredEdgesEffect(Effect):
    """
    Generates a frame with smoother, thicker colored edges on a black background.
    Aims to be less 'glitchy' than the standard colored_edges_only effect.
    """
    # --- Configurable Parameters ---
    PRE_BLUR_KERNEL_SIZE = (7, 7)  # Larger blur before Canny (must be odd)
    CANNY_THRESHOLD1 = 80         # Adjusted Canny thresholds (may need tuning)
    CANNY_THRESHOLD2 = 180
    DILATE_KERNEL_SIZE = (3, 3)   # Kernel for thickening edges (must be odd)
    DILATE_ITERATIONS = 1         # How many times to thicken
    POST_BLUR_KERNEL_SIZE = (5, 5) # Kernel for final smoothing 'glow' (must be odd, set to (0,0) or None to disable)
    # --- End Parameters ---


    def process(self, frame, segment_index, original_frame):
        # Note: input 'frame' (potentially processed) is ignored. Works on original_frame.
        if original_frame is None:
            print(" Error: SmoothColoredEdgesEffect requires original_frame, but it was not provided.")
            return np.zeros_like(frame) if frame is not None else None

        try:
            # 1. Determine edge color based on segment index (same as before)
            color_index = segment_index % len(EDGE_COLORS)
            edge_color_bgr = EDGE_COLORS[color_index]

            # 2. Pre-processing: Grayscale and more aggressive blur
            gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
            gray_blurred = cv2.GaussianBlur(gray, self.PRE_BLUR_KERNEL_SIZE, 0)

            # 3. Edge Detection: Canny with potentially adjusted thresholds
            edges = cv2.Canny(gray_blurred, self.CANNY_THRESHOLD1, self.CANNY_THRESHOLD2)

            # 4. Thicken Edges: Dilate the detected edge map
            if self.DILATE_KERNEL_SIZE and self.DILATE_KERNEL_SIZE[0] > 0 and self.DILATE_ITERATIONS > 0:
                dilate_kernel = np.ones(self.DILATE_KERNEL_SIZE, np.uint8)
                dilated_edges = cv2.dilate(edges, dilate_kernel, iterations=self.DILATE_ITERATIONS)
            else:
                dilated_edges = edges # Skip dilation if kernel size/iterations invalid

            # 5. Create Output Image: Black background, apply color to dilated edges
            # Start with black background matching original frame size
            smooth_color_edges_img = np.zeros_like(original_frame)
            # Apply the selected color only where dilated edges were detected
            smooth_color_edges_img[dilated_edges > 0] = edge_color_bgr

            # 6. Optional Post-Blur: Apply a final blur for a 'glow' effect
            if self.POST_BLUR_KERNEL_SIZE and self.POST_BLUR_KERNEL_SIZE[0] > 0:
                 smooth_color_edges_img = cv2.GaussianBlur(smooth_color_edges_img, self.POST_BLUR_KERNEL_SIZE, 0)

            return smooth_color_edges_img

        except cv2.error as e:
            print(f" OpenCV Error in SmoothColoredEdgesEffect (frame shape: {original_frame.shape}): {e}")
            return np.zeros_like(original_frame) # Return black frame on error
        except Exception as e:
            print(f" General Error in SmoothColoredEdgesEffect: {e}")
            # Optionally print more detailed traceback for debugging
            # import traceback
            # traceback.print_exc()
            return np.zeros_like(original_frame) # Return black frame on error


