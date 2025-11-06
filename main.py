from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import os
import shutil
import subprocess
import uuid

app = FastAPI()

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.get("/ping")
def ping():
    return {"message": "pong"}

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...), prompt: str = ""):
    # Save the uploaded file
    input_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{file.filename}")
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Prepare processed file path
    output_path = os.path.join(PROCESSED_FOLDER, f"processed_{file.filename}")

    # Apply 1000ms delay using ffmpeg
    # aevalsrc=0 generates silence if needed; "adelay=1000|1000" adds 1000ms delay to both channels
    command = [
        "ffmpeg",
        "-y",  # overwrite output if exists
        "-i", input_path,
        "-af", "adelay=1000|1000",
        output_path
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        return {"error": f"FFmpeg processing failed: {e.stderr.decode()}"}

    return {
        "filename": file.filename,
        "prompt": prompt,
        "message": f"File '{file.filename}' uploaded and processed with 1000ms delay!",
        "download_url": f"/download/{os.path.basename(output_path)}"
    }

@app.get("/download/{filename}")
def download_file(filename: str):
    file_path = os.path.join(PROCESSED_FOLDER, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/m4a", filename=filename)
    return {"error": "File not found"}
