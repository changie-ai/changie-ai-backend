from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import librosa
import uuid
import os
import soundfile as sf
import numpy as np

from effects_engine import apply_effect_chain
from prompt_parser import parse_prompt_to_plan

app = FastAPI()

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
FRONTEND_DIR = "frontend"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ✅ CORS (required for browser)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Serve index.html
app.mount("/app", StaticFiles(directory="frontend", html=True), name="frontend")


@app.post("/process")
async def process_audio(
    file: UploadFile = File(...),
    prompt: str = Form(...)
):
    uid = str(uuid.uuid4())

    input_path = f"{UPLOAD_DIR}/{uid}_{file.filename}"
    output_path = f"{OUTPUT_DIR}/{uid}_out.wav"

    with open(input_path, "wb") as f:
        f.write(await file.read())

    audio, sr = librosa.load(input_path, sr=None, mono=True)

    chain = parse_prompt_to_plan(prompt)
    try:
        processed = apply_effect_chain(audio, sr, chain)
    except Exception as e:
        print("❌ EFFECT ENGINE ERROR:", e)
        raise

    processed = np.clip(processed, -1.0, 1.0)
    processed_int16 = (processed * 32767).astype(np.int16)

    sf.write(output_path, processed_int16, sr, subtype="PCM_16")

    return FileResponse(output_path, media_type="audio/wav")
