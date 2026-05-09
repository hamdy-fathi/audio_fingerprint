"""
Speech-to-text transcription using OpenAI Whisper (local).
"""
import whisper
import os
import numpy as np

_model = None

def get_model(model_size="base"):
    global _model
    if _model is None:
        print(f"Loading Whisper model '{model_size}'...")
        _model = whisper.load_model(model_size)
        print("Whisper model loaded.")
    return _model


def transcribe_file(file_path, language="ar"):
    """Full transcription of an audio file. Returns segments with timestamps."""
    model = get_model()
    result = model.transcribe(file_path, language=language, task="transcribe")
    
    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip()
        })
    
    return {
        "full_text": result.get("text", "").strip(),
        "language": result.get("language", "ar"),
        "segments": segments
    }


def transcribe_segments_generator(file_path, language="ar"):
    """Generator that yields transcription segments one by one."""
    model = get_model()
    result = model.transcribe(file_path, language=language, task="transcribe")
    
    for seg in result.get("segments", []):
        yield {
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip()
        }
