import requests

url = "https://changie-ai-backend.onrender.com/upload"
file_path = "uploads/audiofile.m4a"  # Your test file
prompt_text = "delay 0.5"  # Adjust seconds if needed

with open(file_path, "rb") as f:
    files = {"file": f}
    data = {"prompt": prompt_text}
    response = requests.post(url, files=files, data=data)

print(response.json())
