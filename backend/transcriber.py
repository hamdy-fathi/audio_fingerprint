"""
Speech-to-text transcription using OpenAI Whisper (local).
Supports CUDA acceleration and real-time streaming transcription.
"""
import whisper
import torch
import os
import io
import tempfile
import numpy as np
import asyncio

_model = None
_device = None

# ──────────────────────────────────────────────────────────────
# Device helpers
# ──────────────────────────────────────────────────────────────

def get_device():
    """Detect and return the best available device (CUDA > CPU)."""
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"✅ CUDA available — using GPU: {gpu_name} ({vram:.1f} GB VRAM)")
        else:
            _device = "cpu"
            print("⚠️  CUDA not available — falling back to CPU (transcription will be slower)")
    return _device


def get_model(model_size="large-v3"):
    """Load Whisper model onto the best available device (cached singleton)."""
    global _model
    if _model is None:
        device = get_device()
        print(f"Loading Whisper '{model_size}' model on {device.upper()}...")
        _model = whisper.load_model(model_size, device=device)
        print(f"✅ Whisper '{model_size}' model loaded on {device.upper()}.")
    return _model


# ──────────────────────────────────────────────────────────────
# Full-file transcription
# ──────────────────────────────────────────────────────────────

def transcribe_file(file_path, language="ar"):
    """Full transcription of an audio file. Returns segments with timestamps."""
    model = get_model()
    device = get_device()

    result = model.transcribe(
        file_path,
        language=language,
        task="transcribe",
        fp16=(device == "cuda"),       # half-precision on GPU for speed
        condition_on_previous_text=True,
        word_timestamps=True,          # enable word-level timing
    )

    # Build word-by-word segments — each word is its own segment
    segments = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            word_text = w["word"].strip()
            if word_text:
                segments.append({
                    "start": round(w["start"], 2),
                    "end": round(w["end"], 2),
                    "text": word_text,
                })

    return {
        "full_text": result.get("text", "").strip(),
        "language": result.get("language", "ar"),
        "segments": segments
    }


# ──────────────────────────────────────────────────────────────
# Segment-by-segment generator (yields as Whisper decodes)
# ──────────────────────────────────────────────────────────────

def transcribe_segments_generator(file_path, language="ar"):
    """Generator that yields transcription segments one by one."""
    model = get_model()
    device = get_device()

    result = model.transcribe(
        file_path,
        language=language,
        task="transcribe",
        fp16=(device == "cuda"),
        condition_on_previous_text=True,
        word_timestamps=True,
    )

    for seg in result.get("segments", []):
        yield {
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip()
        }


# ──────────────────────────────────────────────────────────────
# Real-time streaming transcription
# ──────────────────────────────────────────────────────────────

CHUNK_DURATION_SEC = 5        # transcribe in 5-second windows
SAMPLE_RATE = 16_000          # Whisper expects 16 kHz mono audio
OVERLAP_SEC = 1               # 1-second overlap for context continuity

def _audio_bytes_to_numpy(raw_bytes: bytes) -> np.ndarray:
    """Convert raw 16-bit PCM bytes → float32 numpy array normalised to [-1, 1]."""
    audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def transcribe_audio_chunk(audio_np: np.ndarray, language="ar"):
    """
    Transcribe a single numpy audio chunk (float32, 16 kHz mono).
    Returns the decoded text or empty string if nothing useful detected.
    """
    model = get_model()
    device = get_device()

    # Whisper requires exactly 30 s of audio; pad shorter chunks
    audio_padded = whisper.pad_or_trim(audio_np)

    mel = whisper.log_mel_spectrogram(audio_padded).to(model.device)

    options = whisper.DecodingOptions(
        language=language,
        task="transcribe",
        fp16=(device == "cuda"),
        without_timestamps=True,
    )

    result = whisper.decode(model, mel, options)
    text = result.text.strip()
    return text


async def realtime_transcribe_stream(websocket, language="ar"):
    """
    Real-time streaming transcription over WebSocket.

    Protocol (client → server):
      • Binary frames  : raw 16-bit PCM audio chunks (16 kHz, mono)
      • Text   frame   : JSON  {"action": "stop"}  to end the session

    Protocol (server → client):
      • JSON  {"text": "...", "is_partial": bool, "chunk_index": int}
      • JSON  {"done": true}  when finished
    """
    model = get_model()  # ensure model is loaded before we start
    audio_buffer = bytearray()
    chunk_bytes = CHUNK_DURATION_SEC * SAMPLE_RATE * 2  # 2 bytes per int16 sample
    overlap_bytes = OVERLAP_SEC * SAMPLE_RATE * 2
    chunk_index = 0

    try:
        while True:
            message = await websocket.receive()

            # ── text frame → control message ──
            if "text" in message:
                import json
                msg = json.loads(message["text"])
                if msg.get("action") == "stop":
                    break
                continue

            # ── binary frame → audio data ──
            if "bytes" in message:
                audio_buffer.extend(message["bytes"])

            # Process complete chunks
            while len(audio_buffer) >= chunk_bytes:
                chunk_data = bytes(audio_buffer[:chunk_bytes])
                # keep overlap for next iteration
                audio_buffer = audio_buffer[chunk_bytes - overlap_bytes:]

                audio_np = _audio_bytes_to_numpy(chunk_data)

                # Run Whisper in a thread so we don't block the event loop
                text = await asyncio.to_thread(
                    transcribe_audio_chunk, audio_np, language
                )

                if text:
                    await websocket.send_json({
                        "text": text,
                        "is_partial": False,
                        "chunk_index": chunk_index,
                    })
                chunk_index += 1

        # ── Flush remaining audio ──
        if len(audio_buffer) > 0:
            remaining = bytes(audio_buffer)
            audio_np = _audio_bytes_to_numpy(remaining)
            text = await asyncio.to_thread(
                transcribe_audio_chunk, audio_np, language
            )
            if text:
                await websocket.send_json({
                    "text": text,
                    "is_partial": False,
                    "chunk_index": chunk_index,
                })

        await websocket.send_json({"done": True})

    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
