# main.py
import os
import soundfile as sf
import librosa
import json
import datetime
from prompt_parser import parse_prompt_to_plan
from effects_engine import apply_effect_chain

UPLOADS_DIR = "uploads"
PROCESSED_DIR = "processed"
LOGS_DIR = "logs"

MODEL_USED = "gpt-4o-mini"  # for logging purposes

def run_interactive():
    while True:
        user_prompt = input("\nEnter audio prompt (or 'q' to quit):\n> ")

        if user_prompt.lower() == "q":
            print("Goodbye!")
            break

        # Generate unique filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        input_file = os.path.join(UPLOADS_DIR, "audiofile.wav")
        output_file = os.path.join(PROCESSED_DIR, f"audio_processed_{timestamp}.wav")
        log_file = os.path.join(LOGS_DIR, f"audio_log_{timestamp}.json")

        print("🎛️ Applying effect chain...")

        # Parse prompt
        try:
            plan = parse_prompt_to_plan(user_prompt)
        except Exception as e:
            print(f"⚠️ GPT parser error: {e}")
            plan = []

        # Load audio
        try:
            audio, sr = librosa.load(input_file, sr=None)
        except Exception as e:
            print(f"⚠️ Could not load input file {input_file}: {e}")
            continue

        # Apply effects
        try:
            output = apply_effect_chain(audio, sr, plan)
            sf.write(output_file, output, sr)
            print(f"✅ Processing complete! Saved as {output_file}")
        except Exception as e:
            print(f"⚠️ Error processing audio: {e}")
            continue

        # Save log
        try:
            log_data = {
                "prompt": user_prompt,
                "plan": plan,
                "input_file": input_file,
                "output_file": output_file,
                "timestamp": timestamp,
                "model_used": MODEL_USED
            }
            with open(log_file, "w") as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save log: {e}")

if __name__ == "__main__":
    # Ensure folders exist
    for d in [UPLOADS_DIR, PROCESSED_DIR, LOGS_DIR]:
        os.makedirs(d, exist_ok=True)

    run_interactive()
