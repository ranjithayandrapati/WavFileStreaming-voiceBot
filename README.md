# WavFileStreaming-voiceBot - SegmentStreamTest

This script demonstrates how to stream audio from a WAV file, segment speech using Silero VAD (Voice Activity Detection), and transcribe each speech segment using an external ASR (Automatic Speech Recognition) HTTP API.

## Features

- Reads an input WAV file (mono or stereo, auto-converts to mono).
- Segments speech using Silero VAD via the LiveKit plugin.
- Streams audio in small frames (default: 20ms).
- Sends detected speech segments to a user-specified ASR HTTP endpoint for transcription.
- Prints out segment timestamps and transcribed text.

## Requirements

- Python 3.7+
- [NumPy](https://numpy.org/)
- [SoundFile](https://pysoundfile.readthedocs.io/en/latest/)
- [LiveKit Python SDK](https://github.com/livekit/client-sdk-python) (with Silero plugin)
- [Requests](https://requests.readthedocs.io/en/latest/)
- An ASR (Automatic Speech Recognition) HTTP endpoint

Install dependencies:
```bash
pip install numpy soundfile requests livekit
```

## Usage

1. **Edit the configuration at the top of `SegmentStreamTest.py`:**
    ```python
    INPUT_FILE = Path(r"Here Your Wav file")      # e.g. Path("sample.wav")
    API_URL = HERE YOUR ASR API                   # e.g. "http://localhost:5000/asr"
    FRAME_MS = 20                                 # (Optional) frame size in ms, default is 20
    ```

2. **Run the script:**
    ```bash
    python SegmentStreamTest.py
    ```

3. **Output:**
    - For each detected speech segment, the script prints the segment index, start and end time (in seconds), and the ASR-transcribed text.

    Example:
    ```
    [01]     0.00s →     1.24s | Hello, how are you?
    [02]     2.15s →     3.67s | I am fine, thank you.
    ```

## How It Works

- The WAV file is loaded and (if needed) converted to mono.
- Audio is split into frames (default: 20ms) and fed into Silero VAD.
- The VAD stream emits events for speech start, end, and inference.
- When a speech segment ends, its audio frames are concatenated, converted to WAV, and sent to the ASR endpoint.
- The ASR's response is printed with timing information.

## Configuration

- **`INPUT_FILE`**: Path to the input WAV audio file.
- **`API_URL`**: Your ASR HTTP endpoint (must accept file uploads in WAV PCM16 format and return JSON with a "text" or "transcript" field).
- **`FRAME_MS`**: (Optional) Frame size in milliseconds. 20ms is recommended for Silero VAD.

## Notes

- Silero VAD settings (min speech duration, silence duration, thresholds, etc.) can be tuned in the script.
- This script is intended for demonstration and batch-processing use. For real-time streaming, integrate with microphone/audio devices and asynchronous pipelines.
- The ASR endpoint is assumed to accept HTTP POST requests with a file field.

## References

- [LiveKit Python SDK](https://github.com/livekit/client-sdk-python)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [SoundFile](https://pysoundfile.readthedocs.io/en/latest/)
- [NumPy](https://numpy.org/)

## License

This script is provided for educational purposes. Check the licenses of included dependencies.
