# audio_responsive_editor/audio_processing/audio_utils.py

import os
import shutil
from moviepy.editor import VideoFileClip

def prepare_audio(video_path, external_audio_path, temp_dir):
    """
    Ensures an audio file is ready for processing.
    Either copies the external audio or extracts audio from the video.

    Args:
        video_path (str): Path to the input video file.
        external_audio_path (str, optional): Path to an external audio file.
        temp_dir (str): Directory to store the extracted/copied audio file.

    Returns:
        str: The path to the audio file to be used for analysis and recombination.
             Returns None if audio preparation fails.
    """
    target_audio_filename = "processing_audio.mp3" # Use a consistent name
    target_audio_path = os.path.join(temp_dir, target_audio_filename)

    if external_audio_path:
        if not os.path.exists(external_audio_path):
            print(f"Error: Provided external audio file not found: {external_audio_path}")
            return None
        try:
            print(f"Using external audio: {external_audio_path}")
            # Copy the external audio to the temp dir for consistency
            shutil.copyfile(external_audio_path, target_audio_path)
            return target_audio_path
        except Exception as e:
            print(f"Error copying external audio file: {e}")
            return None
    else:
        # Extract audio from video
        print(f"Extracting audio from video: {video_path}")
        try:
            with VideoFileClip(video_path) as video_clip:
                if video_clip.audio is None:
                    print(f"Error: Input video '{video_path}' does not contain an audio track.")
                    return None
                video_clip.audio.write_audiofile(target_audio_path, codec='mp3')
            print(f"Audio extracted successfully to: {target_audio_path}")
            return target_audio_path
        except Exception as e:
            # Catch potential errors during extraction (file not found, format issues)
            print(f"Error extracting audio from video '{video_path}': {e}")
            return None

# --- Example Usage (for testing this module directly) ---
if __name__ == "__main__":
    # Create dummy paths/files for testing
    test_video = "path/to/your/test_video.mp4" # <--- Set a real video path
    test_audio_ext = "path/to/your/external_audio.mp3" # <--- Set a real audio path (optional)
    temp_directory = "temp_audio_test"
    os.makedirs(temp_directory, exist_ok=True)

    print("--- Testing with Video Audio Extraction ---")
    if os.path.exists(test_video):
         audio_path_extracted = prepare_audio(test_video, None, temp_directory)
         if audio_path_extracted:
             print(f"Resulting audio path: {audio_path_extracted}")
             # os.remove(audio_path_extracted) # Clean up dummy file
         else:
             print("Audio extraction failed.")
    else:
         print(f"Test video not found: {test_video}")


    print("\n--- Testing with External Audio ---")
    if os.path.exists(test_audio_ext):
        audio_path_external = prepare_audio("dummy_video.mp4", test_audio_ext, temp_directory)
        if audio_path_external:
            print(f"Resulting audio path: {audio_path_external}")
            # os.remove(audio_path_external) # Clean up dummy file
        else:
            print("External audio preparation failed.")
    else:
        print(f"Test external audio not found: {test_audio_ext}")

    # Clean up temp directory
    # shutil.rmtree(temp_directory)
    print(f"\n(Manual cleanup suggested for: {temp_directory})")