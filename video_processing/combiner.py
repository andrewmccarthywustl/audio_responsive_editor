# audio_responsive_editor/video_processing/combiner.py

import os
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip
from utils import file_utils # Import the utility for sorting

def combine_segments_with_audio(processed_segments_folder, audio_path, output_video_path):
    """
    Combines processed video segments and sets the final audio track.

    Args:
        processed_segments_folder (str): Folder containing the video segments
                                         with effects applied.
        audio_path (str): Path to the final audio file (extracted or external).
        output_video_path (str): Path to save the final combined video file.

    Returns:
        bool: True if combination is successful, False otherwise.
    """
    if not os.path.isdir(processed_segments_folder):
        print(f"Error: Processed segments folder not found: {processed_segments_folder}")
        return False

    if not os.path.exists(audio_path):
        print(f"Error: Final audio file not found: {audio_path}")
        return False

    segment_files = [f for f in os.listdir(processed_segments_folder) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    if not segment_files:
        print(f"Error: No video segments found in {processed_segments_folder}")
        return False

    # --- Sort segments chronologically ---
    segment_paths = [os.path.join(processed_segments_folder, f) for f in segment_files]
    sorted_segment_paths = file_utils.sort_video_segments(segment_paths)

    print(f"Combining {len(sorted_segment_paths)} processed segments...")
    print("Segments order:")
    for i, p in enumerate(sorted_segment_paths):
        print(f"  {i+1}. {os.path.basename(p)}")

    try:
        # --- Load video clips ---
        print("Loading video clips...")
        clips = []
        total_duration = 0
        for segment_path in sorted_segment_paths:
             try:
                 clip = VideoFileClip(segment_path)
                 if clip.duration is None or clip.duration <= 0:
                      print(f"  Warning: Skipping segment with zero or invalid duration: {os.path.basename(segment_path)}")
                      continue
                 clips.append(clip)
                 total_duration += clip.duration
                 print(f"  Loaded: {os.path.basename(segment_path)} (Duration: {clip.duration:.3f}s)")
             except Exception as e:
                 print(f"  Warning: Error loading clip {os.path.basename(segment_path)}, skipping: {e}")
                 continue # Skip problematic clips

        if not clips:
             print("Error: No valid video clips could be loaded for concatenation.")
             return False

        # --- Concatenate video clips ---
        print("Concatenating video clips...")
        final_video_clip = concatenate_videoclips(clips, method="compose")
        print(f"Combined video duration: {final_video_clip.duration:.3f}s (Target based on segments: {total_duration:.3f}s)")

        # --- Load audio clip ---
        print(f"Loading final audio track: {audio_path}")
        final_audio_clip = AudioFileClip(audio_path)
        print(f"Audio duration: {final_audio_clip.duration:.3f}s")

        # --- Set audio ---
        # Trim audio to match video duration if necessary, or warn if video is longer
        if final_audio_clip.duration > final_video_clip.duration:
             print(f"Warning: Audio duration ({final_audio_clip.duration:.3f}s) is longer than video duration ({final_video_clip.duration:.3f}s). Trimming audio.")
             final_audio_clip = final_audio_clip.subclip(0, final_video_clip.duration)
        elif final_video_clip.duration > final_audio_clip.duration:
             print(f"Warning: Video duration ({final_video_clip.duration:.3f}s) is longer than audio duration ({final_audio_clip.duration:.3f}s). Final video will have trailing silence.")
             # Moviepy handles this automatically by default (padding with silence)

        print("Attaching final audio track...")
        final_clip_with_audio = final_video_clip.set_audio(final_audio_clip)

        # --- Write output file ---
        output_dir = os.path.dirname(output_video_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"Writing final video to: {output_video_path}")
        # Use appropriate codec, bitrate etc. Might need adjustment for quality/size.
        # 'libx264' is common for H.264/MP4. preset='medium' is a balance.
        # audio_codec='aac' is standard for MP4.
        final_clip_with_audio.write_videofile(
            output_video_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a', # Helps manage temporary audio files
            remove_temp=True, # Remove the temp audio file afterwards
            preset='medium', # Faster presets: ultrafast, superfast. Slower: slow, veryslow
            logger='bar' # Show progress bar
        )

        # --- Cleanup: Close clips to release file handles ---
        print("Closing video/audio clips...")
        final_audio_clip.close()
        final_clip_with_audio.close() # Includes final_video_clip and all concatenated clips
        # Explicitly close original clips list just in case
        for clip in clips:
             clip.close()


        print("\nVideo combination complete!")
        return True

    except Exception as e:
        print(f"Error during video combination: {e}")
        # Attempt to close any clips that might be open
        try:
            if 'final_audio_clip' in locals() and final_audio_clip: final_audio_clip.close()
            if 'final_clip_with_audio' in locals() and final_clip_with_audio: final_clip_with_audio.close()
            if 'clips' in locals():
                for clip in clips: clip.close()
        except Exception as ce:
            print(f"  Additional error during cleanup: {ce}")
        return False


# --- Example Usage ---
if __name__ == "__main__":
    # Assumes you ran cutter.py and effects.py examples
    test_input_dir = "temp_effects_test" # Use output from effects test
    # Use the audio generated by audio_utils.py test or a known audio file
    test_audio = "temp_audio_test/processing_audio.mp3" # From audio_utils test
    test_output_video = "temp_final_output/final_video.mp4"

    if os.path.isdir(test_input_dir) and os.path.exists(test_audio):
        print("Testing final combination...")
        success = combine_segments_with_audio(test_input_dir, test_audio, test_output_video)
        if success:
            print(f"\nTest combination finished. Check file: {test_output_video}")
            # Optional cleanup
            # import shutil
            # shutil.rmtree(os.path.dirname(test_output_video)) # Remove temp_final_output folder
        else:
            print("\nTest combination failed.")
    else:
        print("Missing input for combination test:")
        if not os.path.isdir(test_input_dir): print(f"  - Processed segments folder: {test_input_dir}")
        if not os.path.exists(test_audio): print(f"  - Audio file: {test_audio}")
        print("(Run the cutter.py, effects.py, and audio_utils.py examples first)")