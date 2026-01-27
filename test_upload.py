# test_upload.py
import requests

url = "http://127.0.0.1:8255/upload"
file_path = "uploads/audiofile.wav"

with open(file_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print(response.json())
