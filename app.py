from flask import Flask, request, jsonify, send_file
import os

app = Flask(__name__)

# Folders for original uploads and processed files
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return "Changie.ai backend online and ready to process audio!"

# Upload + prompt route
@app.route('/upload', methods=['POST'])
def upload_audio():
    file = request.files.get('file')
    prompt = request.form.get('prompt', '')

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # Save the uploaded file
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Placeholder for audio processing
    # Here you will later use ffmpeg / ffprobe to apply changes
    processed_path = os.path.join(PROCESSED_FOLDER, file.filename)
    # For now, just copy the uploaded file as "processed"
    from shutil import copyfile
    copyfile(filepath, processed_path)

    # Return JSON with next steps and a link to download
    return jsonify({
        "message": f"File '{file.filename}' uploaded successfully!",
        "prompt": prompt,
        "next_step": "Audio processing would happen here",
        "download_url": f"/download/{file.filename}"
    })

# Route to download processed audio
@app.route('/download/<filename>', methods=['GET'])
def download_audio(filename):
    processed_path = os.path.join(PROCESSED_FOLDER, filename)
    if not os.path.exists(processed_path):
        return jsonify({"error": "File not found"}), 404
    return send_file(processed_path, as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
