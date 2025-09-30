# livekit_silero_vad_stream_to_asr_save_chunks.py
from pathlib import Path
import os
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
INPUT_FILE = Path(r"Wave or Audio file path here")
API_URL = "Your ASR API Link here"  # your ASR HTTP endpoint
FRAME_MS = 20  # 20ms frames work well with Silero VAD

# Where to save the VAD chunks (Downloads/vad_segments)
OUTPUT_DIR = Path(os.path.expanduser("~/Downloads"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# ---------------------------------

def _wav_bytes_from_int16_mono(int16_pcm: np.ndarray, sr: int) -> bytes:
    """Make a small WAV (PCM16) in-memory for HTTP upload."""
    buf = io.BytesIO()
    sf.write(buf, int16_pcm.astype(np.int16), sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()

def _save_chunk_to_disk(int16_pcm, sr, seg_idx, start_t, end_t):
    base = OUTPUT_DIR / f"segment_{seg_idx:03d}_{start_t:06.2f}s-{end_t:06.2f}s"
    out_path = base.with_suffix(".wav")
    n = 1
    while out_path.exists():
        out_path = OUTPUT_DIR / f"{base.name}__{n}.wav"
        out_path = out_path.with_suffix(".wav")
        n += 1
    sf.write(str(out_path), int16_pcm.astype(np.int16), sr, format="WAV", subtype="PCM_16")
    return out_path


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

    # Create Silero VAD via LiveKit plugin (8k/16k supported; plugin will resample internally if needed)
    vad = silero.VAD.load(
        min_speech_duration=0.15,       # need at least 150ms of speech to start
        min_silence_duration=0.35,      # end after 350ms silence
        prefix_padding_duration=0.25,   # include 250ms of pre-roll
        activation_threshold=0.5,       # speech prob threshold
        sample_rate=8000,               # model inference SR (8k/16k only)
        force_cpu=True,
    )

    stream = vad.stream()  # async iterator yielding VADEvents

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
            )
            stream.push_frame(frame)
            offset += samples_per_frame

        # Signal no more input
        stream.end_input()

    # Collect, save, and transcribe each detected speech segment
    async def consumer():
        seg_idx = 0
        async for ev in stream:
            # Only act on END_OF_SPEECH to get the full utterance
            if ev.type.value == "end_of_speech":
                seg_idx += 1
                frames = ev.frames
                if not frames:
                    continue

                # Concatenate raw int16 data from all frames for this segment
                seg_pcm = np.concatenate([np.frombuffer(f.data, dtype=np.int16) for f in frames])

                # Timestamps
                end_t = ev.timestamp            # seconds
                dur = ev.speech_duration        # seconds
                start_t = max(0.0, end_t - dur)

                # --- Save the chunk to your Downloads/vad_segments folder ---
                out_path = _save_chunk_to_disk(seg_pcm, sr, seg_idx, start_t, end_t)

                # --- Send to ASR and print result ---
                try:
                    wav_bytes = _wav_bytes_from_int16_mono(seg_pcm, sr)
                    text = _post_to_asr(wav_bytes)
                except Exception as e:
                    text = f"[ASR error] {e}"

                print(f"[{seg_idx:02d}] {start_t:8.2f}s → {end_t:8.2f}s | saved: {out_path} | {text}")

    await asyncio.gather(feeder(), consumer())

if __name__ == "__main__":
    asyncio.run(run())

