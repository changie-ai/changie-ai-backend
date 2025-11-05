from fastapi import FastAPI, UploadFile, File
import imageio_ffmpeg as ffmpeg

app = FastAPI()

@app.get("/test_ffmpeg")
def test_ffmpeg():
    return {
        "ffmpeg_version": ffmpeg.get_ffmpeg_version(),
        "ffprobe_version": ffmpeg.get_ffprobe_version()
    }

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...), prompt: str = ""):
    # Just returns the file name and prompt for now
    return {
        "filename": file.filename,
        "prompt": prompt
    }
