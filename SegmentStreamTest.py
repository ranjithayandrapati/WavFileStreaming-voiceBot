# livekit_silero_vad_stream_to_asr.py
from pathlib import Path
import asyncio
import io
import json
import math
import requests
import numpy as np
import soundfile as sf

from livekit import rtc
from livekit.plugins import silero

# ---------- your config ----------
INPUT_FILE = Path(r"C:\Users\Dhanaranjitha\Downloads\AUDIO_user_8657862907_1756912432.854027.wav")
API_URL = "http://35.200.216.143:8000/v1/audio/transcriptions"  # your ASR HTTP endpoint
FRAME_MS = 20  # 20ms frames work well with Silero VAD
# ---------------------------------

def _wav_bytes_from_int16_mono(int16_pcm: np.ndarray, sr: int) -> bytes:
    """Make a small WAV (PCM16) in-memory for HTTP upload."""
    buf = io.BytesIO()
    sf.write(buf, int16_pcm.astype(np.int16), sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()

def _post_to_asr(wav_bytes: bytes) -> str:
    files = {"file": ("segment.wav", io.BytesIO(wav_bytes), "audio/wav")}
    # Add extra form fields if your server expects them: data={"language":"en", ...}
    r = requests.post(API_URL, files=files, timeout=120)
    r.raise_for_status()
    try:
        payload = r.json()
        return payload.get("text") or payload.get("transcript") or str(payload)
    except json.JSONDecodeError:
        return r.text

async def run():
    assert INPUT_FILE.exists(), f"File not found: {INPUT_FILE}"
    pcm, sr = sf.read(str(INPUT_FILE), dtype="int16", always_2d=False)
    # ensure mono (LiveKit frames are interleaved if >1ch)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1).astype(np.int16)
    num_channels = 1

    # Create Silero VAD via LiveKit plugin (16k infer by default; will auto-resample internally)
    # You can tune these knobs if needed.
    vad = silero.VAD.load(
        min_speech_duration=0.15,       # need at least 150ms of speech to start
        min_silence_duration=0.35,      # end after 350ms silence
        prefix_padding_duration=0.25,   # include 250ms of pre-roll
        activation_threshold=0.5,       # speech prob threshold
        sample_rate=8000,              # model inference SR (8k/16k only)
        force_cpu=True,
    )  # loads ONNX model, returns a VAD you can stream to. :contentReference[oaicite:1]{index=1}

    stream = vad.stream()  # returns an async VADStream iterator that yields VADEvents. :contentReference[oaicite:2]{index=2}

    # Helper to push frames into the VAD stream
    samples_per_frame = int(sr * FRAME_MS / 1000)
    total_samples = len(pcm)
    total_frames = math.ceil(total_samples / samples_per_frame)

    async def feeder():
        """Push 20ms AudioFrame chunks into the VAD stream."""
        offset = 0
        for _ in range(total_frames):
            chunk = pcm[offset: offset + samples_per_frame]
            if len(chunk) == 0:
                break
            # pad last frame if short
            if len(chunk) < samples_per_frame:
                pad = np.zeros(samples_per_frame - len(chunk), dtype=np.int16)
                chunk = np.concatenate([chunk, pad])

            # LiveKit AudioFrame takes interleaved int16 bytes
            frame = rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=sr,
                num_channels=num_channels,
                samples_per_channel=len(chunk),
            )  # AudioFrame is 16-bit PCM, interleaved; includes sr/ch/samples. :contentReference[oaicite:3]{index=3}

            stream.push_frame(frame)
            offset += samples_per_frame

        # Signal no more input
        stream.end_input()

    # Collect transcriptions in timestamp order
    async def consumer():
        seg_idx = 0
        async for ev in stream:
            # VAD emits START_OF_SPEECH, INFERENCE_DONE, END_OF_SPEECH events carrying frames & timing
            # We only act on END_OF_SPEECH to get the full utterance. :contentReference[oaicite:4]{index=4}
            if ev.type.value == "end_of_speech":
                seg_idx += 1
                # Combine all frames' PCM back-to-back (all frames are int16 mono)
                frames = ev.frames
                if not frames:
                    continue
                # Concatenate raw int16 data
                seg_pcm = np.concatenate([np.frombuffer(f.data, dtype=np.int16) for f in frames])

                # The frames’ sample_rate is the *input* sample rate we pushed (sr)
                # Convert to micro wav bytes for your ASR
                wav_bytes = _wav_bytes_from_int16_mono(seg_pcm, sr)

                # fire to ASR and print
                try:
                    text = _post_to_asr(wav_bytes)
                except Exception as e:
                    text = f"[ASR error] {e}"

                # ev.timestamp is seconds; we don’t get absolute start/end directly, but you can
                # approximate end time with ev.timestamp and duration with ev.speech_duration. :contentReference[oaicite:5]{index=5}
                end_t = ev.timestamp
                dur = ev.speech_duration
                start_t = max(0.0, end_t - dur)

                print(f"[{seg_idx:02d}] {start_t:8.2f}s → {end_t:8.2f}s | {text}")

    await asyncio.gather(feeder(), consumer())

if __name__ == "__main__":
    asyncio.run(run())
