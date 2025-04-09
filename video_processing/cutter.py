# audio_responsive_editor/video_processing/cutter.py

import cv2
import os

def cut_video_at_times(video_path, cut_times_seconds, output_folder):
    """
    Splits a video into segments based on a list of timestamps.

    Args:
        video_path (str): Path to the input video file.
        cut_times_seconds (list[float]): Sorted list of timestamps (in seconds) where cuts should occur.
        output_folder (str): Directory to save the output video segments.

    Returns:
        bool: True if cutting was successful (or partially successful), False otherwise.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found for cutting: {video_path}")
        return False

    os.makedirs(output_folder, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    print(f"Loading video for cutting: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return False

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps > 0 else 0

    print(f"Video properties: {frame_width}x{frame_height} @ {fps:.2f} FPS, Duration: {video_duration:.3f}s")

    if fps <= 0:
        print("Error: Invalid FPS detected. Cannot process video.")
        cap.release()
        return False

    # Prepare segment boundaries, including start and end of video
    # Filter out times beyond video duration
    valid_cut_times = sorted([t for t in cut_times_seconds if 0 < t < video_duration])
    segment_boundaries = [0.0] + valid_cut_times + [video_duration]
    # Remove duplicates that might arise if a cut time is exactly 0 or video_duration
    segment_boundaries = sorted(list(set(segment_boundaries)))


    print(f"Planned cuts create {len(segment_boundaries) - 1} segments.")
    segment_count = 0

    # Process each segment
    for i in range(len(segment_boundaries) - 1):
        start_time = segment_boundaries[i]
        end_time = segment_boundaries[i + 1]

        # Ensure end_time is strictly greater than start_time to avoid zero-length segments
        if end_time <= start_time:
            print(f"Skipping zero-duration segment at {start_time:.3f}s")
            continue

        segment_count += 1
        output_filename = os.path.join(
            output_folder,
            f"{base_name}_segment_{i:04d}_{start_time:.3f}s-{end_time:.3f}s.mp4"
        )

        print(f"Creating segment {segment_count}: {start_time:.3f}s to {end_time:.3f}s -> {os.path.basename(output_filename)}")

        # Calculate start and end frames
        start_frame = int(start_time * fps)
        # Ensure end_frame doesn't exceed total_frames
        end_frame = min(int(end_time * fps), total_frames)

        # Ensure we don't try to read past the actual end frame
        if start_frame >= total_frames:
            print(f"  Warning: Start frame ({start_frame}) is beyond total frames ({total_frames}). Skipping segment.")
            continue
        if start_frame >= end_frame:
             print(f"  Warning: Start frame ({start_frame}) is >= end frame ({end_frame}). Skipping segment.")
             continue


        # Set up video writer
        # Use a reliable codec like 'mp4v' for .mp4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        try:
            out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
            if not out.isOpened():
                 print(f"  Error: Could not open VideoWriter for {output_filename}. Check permissions and disk space.")
                 continue # Skip this segment

            # Set frame position to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Read and write frames for the current segment
            frames_written = 0
            for current_frame_idx in range(start_frame, end_frame):
                # Check if we are still reading correctly
                # Sometimes cap.read() can fail before the expected end_frame
                if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != current_frame_idx:
                     # print(f"  Frame position mismatch: Expected {current_frame_idx}, Got {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}. Attempting reset...")
                     cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
                     if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != current_frame_idx:
                           print(f"  Error: Failed to seek to frame {current_frame_idx}. Stopping segment.")
                           break # Exit inner loop for this segment

                ret, frame = cap.read()
                if not ret:
                    # Reached end of video unexpectedly or read error
                    print(f"  Warning: Could not read frame {current_frame_idx} (expected end: {end_frame}). Stopping segment.")
                    break # Exit inner loop for this segment

                out.write(frame)
                frames_written += 1

            out.release()
            print(f"  Segment saved with {frames_written} frames.")

        except Exception as e:
            print(f"  Error processing segment {segment_count}: {e}")
            # Ensure the writer is released if an error occurred mid-write
            if 'out' in locals() and out.isOpened():
                out.release()
            continue # Move to the next segment

    # Clean up
    cap.release()
    print(f"\nVideo cutting complete. Segments saved in: {output_folder}")
    return True

# --- Example Usage ---
if __name__ == "__main__":
    test_video = "path/to/your/test_video.mp4" # <--- Set a real video path
    test_cuts = [2.5, 5.1, 8.0, 15.9] # Example cut times in seconds
    test_output_dir = "temp_cut_test"

    if os.path.exists(test_video):
        success = cut_video_at_times(test_video, test_cuts, test_output_dir)
        if success:
            print(f"\nTest cutting finished. Check folder: {test_output_dir}")
            # Clean up dummy files/folders if desired
            # import shutil
            # shutil.rmtree(test_output_dir)
        else:
            print("\nTest cutting failed.")
    else:
        print(f"Test video not found: {test_video}. Cannot run example.")