from fastapi import FastAPI, UploadFile, File
import imageio_ffmpeg as ffmpeg

app = FastAPI()

# Simple ping endpoint to check if the server is alive
@app.get("/ping")
def ping():
    return {"message": "pong"}

# Test ffmpeg and ffprobe versions
@app.get("/test_ffmpeg")
def test_ffmpeg():
    return {
        "ffmpeg_version": ffmpeg.get_ffmpeg_version(),
        "ffprobe_version": ffmpeg.get_ffprobe_version()
    }

# Upload endpoint for audio files
@app.post("/upload")
async def upload_audio(file: UploadFile = File(...), prompt: str = ""):
    # Currently just returns filename and prompt
    return {
        "filename": file.filename,
        "prompt": prompt,
        "message": f"File '{file.filename}' uploaded successfully!"
    }
