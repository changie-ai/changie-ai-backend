from fastapi import FastAPI, UploadFile, File
import shutil
import subprocess
import os

app = FastAPI()

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...), prompt: str = ""):
    # Save the uploaded file
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Determine output path
    output_path = os.path.join(PROCESSED_FOLDER, f"processed_{file.filename}")

    # Simple FFmpeg delay effect (echo)
    if "delay" in prompt.lower():
        # Add 0.5s echo delay
        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-filter_complex", "aecho=0.8:0.9:500|500:0.3|0.25",
            output_path
        ]
        subprocess.run(cmd, check=True)

    # Return info
    return {
        "filename": file.filename,
        "prompt": prompt,
        "message": f"File '{file.filename}' uploaded successfully!",
        "processed_file": output_path
    }
