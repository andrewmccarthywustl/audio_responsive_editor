# Audio Responsive Video Editor

This script processes an input video, detects audio events (peaks or onsets), cuts the video at these event times, applies visual effects to the segments (changing effect parameters per segment), and recombines the video with audio.

## Features

- Detects audio events using overall amplitude peaks or spectral onsets in harmonic/percussive components (via HPSS).
- Optionally uses external audio instead of the video's original audio.
- Applies effects like duotone, hue rotation, and colored edges, cycling parameters based on detected audio events.
- Command-line interface for controlling input, output, audio source, detection method, and effects.

## Setup

1.  **Clone/Download:** Get the project files onto your computer.
2.  **Create Virtual Environment:** It's highly recommended to use a virtual environment. Navigate to the project's root directory (the one containing `audio_responsive_editor` and `requirements.txt`) in your terminal and create an environment (using Python 3.11 or 3.12 recommended due to potential dependency issues with 3.13):
    ```bash
    # Using Python's built-in venv
    python3.11 -m venv .venv # Or python3.12
    # Or using virtualenv if installed
    # virtualenv .venv -p python3.11
    ```
3.  **Activate Environment:**
    - macOS/Linux: `source .venv/bin/activate`
    - Windows: `.venv\Scripts\activate`
4.  **Install Dependencies:** Use `uv` (if installed) or `pip`:

    ```bash
    # Using uv
    uv pip install -r requirements.txt

    # OR Using pip
    pip install -r requirements.txt
    ```

    _(See `requirements.txt` for the list of dependencies)._

## Usage

There are two main ways to run the script from your terminal, **make sure your virtual environment is active**:

**Method 1: Running as a Module (Recommended if it works)**

Navigate to the directory _containing_ the `audio_responsive_editor` folder (the project root). Run using `python -m`:

```bash
python -m audio_responsive_editor.main [OPTIONS]
```
