# Audio Responsive Video Editor

This script processes an input video, detects audio events (peaks or onsets), cuts the video at these event times, applies visual effects to the segments (changing effect parameters per segment), and recombines the video with audio.

## Features

- Detects audio events using overall amplitude peaks or spectral onsets in harmonic/percussive components (via HPSS).
- Optionally uses external audio instead of the video's original audio.
- applies effects based on
- Command-line interface for controlling input, output, audio source, detection method, and effects.

## Usage

[Demo Output](https://www.youtube.com/watch?v=iaTsmtOMX7w)

Usage:

```bash
python main.py --input ../test.MOV --output ../output_edges_only.mp4 -e colored_edges_only --detect-mode percussive
```
