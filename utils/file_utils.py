# audio_responsive_editor/utils/file_utils.py

import re
import os

def extract_timestamp_from_filename(filename):
    """
    Extracts the starting timestamp (in seconds) from standard segment filenames.
    Expected format: "..._segment_XXX_STARTs-ENDs.ext"
    Falls back to alphabetical sort if no pattern matches.

    Args:
        filename (str): The filename to parse.

    Returns:
        float or str: The starting timestamp as a float, or the original
                      filename for fallback sorting.
    """
    # Strongest pattern first: explicit start time marker
    match = re.search(r'_(\d+\.?\d*)s-\d+\.?\d*s\.(mp4|avi|mov|mkv)$', filename, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass # Should not happen with this regex, but good practice

    # Fallback: look for any number followed by 's' near the end
    match = re.search(r'_(\d+\.?\d*)s\.(mp4|avi|mov|mkv)$', filename, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    # Fallback: look for segment number if present
    match = re.search(r'_segment_(\d+)', filename)
    if match:
        try:
            # Return as int to ensure segments are ordered numerically
            return int(match.group(1))
        except ValueError:
            pass

    # Ultimate fallback: alphabetical sort
    return filename

def sort_video_segments(segment_paths):
    """
    Sorts a list of video segment file paths based on timestamps extracted
    from their filenames.

    Args:
        segment_paths (list[str]): A list of full paths to video segment files.

    Returns:
        list[str]: The sorted list of video segment file paths.
    """
    # Create a list of tuples (sort_key, path)
    sortable_list = []
    for path in segment_paths:
        filename = os.path.basename(path)
        sort_key = extract_timestamp_from_filename(filename)
        sortable_list.append((sort_key, path))

    # Sort the list based on the extracted key
    # Handles both numeric keys (timestamps, segment numbers) and string keys (fallback)
    sortable_list.sort(key=lambda item: item[0])

    # Return just the sorted paths
    return [path for key, path in sortable_list]

# --- Example Usage ---
if __name__ == "__main__":
    test_files = [
        "path/to/video_segment_002_10.500s-15.000s.mp4",
        "path/to/video_segment_000_0.000s-5.123s.mp4",
        "path/to/video_segment_001_5.123s-10.500s.mp4",
        "path/to/another_video_part_1.mp4", # Fallback example
        "path/to/another_video_part_10.mp4" # Fallback example
    ]
    print("Original files:", test_files)
    sorted_files = sort_video_segments(test_files)
    print("Sorted files:", sorted_files)

    print("\nTesting key extraction:")
    for f in test_files:
        key = extract_timestamp_from_filename(os.path.basename(f))
        print(f"  {os.path.basename(f)} -> Key: {key} (Type: {type(key)})")