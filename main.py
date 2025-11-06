from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil
import subprocess

app = FastAPI()

UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

@app.get("/ping")
def ping():
    return {"message": "pong"}

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...), prompt: str = ""):
    # Save uploaded file
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    output_file = PROCESSED_DIR / f"processed_{file.filename}"

    # Apply delay if requested
    if "add delay" in prompt.lower():
        subprocess.run([
            "ffmpeg",
            "-y",
            "-i", str(file_path),
            "-filter_complex", "adelay=500|500",
            str(output_file)
        ])
    else:
        # Copy original if no effect
        shutil.copy(file_path, output_file)

    return {
        "original_filename": file.filename,
        "processed_filename": output_file.name,
        "prompt": prompt,
        "message": "Audio processed successfully!"
    }
