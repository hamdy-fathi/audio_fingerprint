"""
Dialect conversion routes.
"""
import os, base64
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from transcriber import transcribe_file
from dialect_converter import (
    convert_text_to_dialect, synthesize_speech,
    DIALECT_VOICES, DIALECT_DICTIONARIES, DIALECT_SENTENCES
)
from routes.upload import get_file_path

router = APIRouter(prefix="/api", tags=["conversion"])


class ConvertRequest(BaseModel):
    file_id: str
    target_dialect: str
    source_dialect: Optional[str] = None
    gender: Optional[str] = 'male'


@router.post("/convert-dialect")
async def convert_dialect(req: ConvertRequest):
    """Convert audio to a different dialect."""
    path = get_file_path(req.file_id)
    if not path or not os.path.exists(path):
        raise HTTPException(404, "File not found")

    if req.target_dialect not in DIALECT_VOICES:
        raise HTTPException(400, f"Invalid dialect. Choose from: {list(DIALECT_VOICES.keys())}")

    # Step 1: Transcribe original audio
    transcription = transcribe_file(path)
    original_text = transcription['full_text']

    # Step 2: Detect source dialect if not provided
    source = req.source_dialect or 'egyptian'

    # Step 3: Convert text to target dialect
    converted_text = convert_text_to_dialect(original_text, source, req.target_dialect)

    # Step 4: Synthesize speech in target dialect
    audio_bytes = await synthesize_speech(converted_text, req.target_dialect, req.gender or 'male')
    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

    return {
        'original_text': original_text,
        'converted_text': converted_text,
        'source_dialect': source,
        'target_dialect': req.target_dialect,
        'audio_base64': audio_b64,
        'voice_used': DIALECT_VOICES[req.target_dialect][req.gender or 'male']
    }


@router.get("/dialect-voices")
async def list_dialect_voices():
    """List available dialect voice options."""
    return {
        'voices': DIALECT_VOICES,
        'sample_sentences': DIALECT_SENTENCES,
        'dictionaries': {k: dict(list(v.items())[:10]) for k, v in DIALECT_DICTIONARIES.items()}
    }
