"""
Upload and sample listing routes.
"""
import os, uuid, shutil, json
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import soundfile as sf

router = APIRouter(prefix="/api", tags=["upload"])

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'temp')
SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'audio_samples')

# In-memory file registry
uploaded_files = {}


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload an audio file and return a file ID."""
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1] or '.wav'
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    with open(save_path, 'wb') as f:
        content = await file.read()
        f.write(content)

    # Get file info
    try:
        info = sf.info(save_path)
        duration = info.duration
        sample_rate = info.samplerate
    except:
        duration = 0
        sample_rate = 0

    uploaded_files[file_id] = {
        'path': save_path,
        'filename': file.filename,
        'duration': duration,
        'sample_rate': sample_rate
    }

    return {
        'file_id': file_id,
        'filename': file.filename,
        'duration': round(duration, 2),
        'sample_rate': sample_rate
    }


@router.get("/samples")
async def list_samples():
    """List all available pre-loaded sample files organized by dialect."""
    samples = {}
    dialects = ['egyptian', 'gulf', 'levantine', 'maghrebi']
    dialect_labels = {
        'egyptian': 'Egyptian (مصري)',
        'gulf': 'Gulf (خليجي)',
        'levantine': 'Levantine (شامي)',
        'maghrebi': 'Maghrebi (مغربي)'
    }

    for dialect in dialects:
        dialect_dir = os.path.join(SAMPLES_DIR, dialect)
        if not os.path.exists(dialect_dir):
            continue
        files = []
        for fname in sorted(os.listdir(dialect_dir)):
            if fname.endswith(('.wav', '.mp3', '.ogg')):
                fpath = os.path.join(dialect_dir, fname)
                file_id = f"sample_{dialect}_{fname}"
                uploaded_files[file_id] = {
                    'path': fpath,
                    'filename': fname,
                    'dialect': dialect
                }
                try:
                    info = sf.info(fpath)
                    dur = round(info.duration, 2)
                except:
                    dur = 0
                files.append({
                    'file_id': file_id,
                    'filename': fname,
                    'dialect': dialect,
                    'duration': dur
                })
        samples[dialect] = {
            'label': dialect_labels.get(dialect, dialect),
            'files': files
        }

    return samples


@router.get("/file/{file_id}")
async def get_file(file_id: str):
    """Serve an audio file for playback."""
    if file_id not in uploaded_files:
        raise HTTPException(404, "File not found")
    path = uploaded_files[file_id]['path']
    if not os.path.exists(path):
        raise HTTPException(404, "File not found on disk")
    return FileResponse(path, media_type="audio/wav")


def get_file_path(file_id: str):
    """Helper to get file path from ID."""
    if file_id in uploaded_files:
        return uploaded_files[file_id]['path']
    return None
