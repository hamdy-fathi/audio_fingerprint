"""
Transcription routes using Whisper.
"""
import os, json
from fastapi import APIRouter, WebSocket, HTTPException
from pydantic import BaseModel
from transcriber import transcribe_file, transcribe_segments_generator
from routes.upload import get_file_path

router = APIRouter(prefix="/api", tags=["transcription"])


class TranscribeRequest(BaseModel):
    file_id: str


@router.post("/transcribe")
async def transcribe(req: TranscribeRequest):
    """Full transcription of an audio file."""
    path = get_file_path(req.file_id)
    if not path or not os.path.exists(path):
        raise HTTPException(404, "File not found")
    
    result = transcribe_file(path)
    return result


@router.websocket("/transcribe/ws")
async def transcribe_ws(websocket: WebSocket):
    """WebSocket endpoint for streaming transcription segments."""
    await websocket.accept()
    try:
        data = await websocket.receive_text()
        msg = json.loads(data)
        file_id = msg.get('file_id')

        path = get_file_path(file_id)
        if not path or not os.path.exists(path):
            await websocket.send_json({'error': 'File not found'})
            await websocket.close()
            return

        # Send segments one by one
        for segment in transcribe_segments_generator(path):
            await websocket.send_json(segment)

        await websocket.send_json({'done': True})
    except Exception as e:
        try:
            await websocket.send_json({'error': str(e)})
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass
