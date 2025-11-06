import requests

# URL of your deployed backend (replace with your Render URL if needed)
url = "https://changie-ai-backend.onrender.com/upload"

# Path to the audio file you want to upload
file_path = "uploads/audiofile.m4a"

# Prompt for the effect(s)
prompt_text = "add delay"

# Open the file and send the POST request
with open(file_path, "rb") as f:
    files = {"file": f}
    data = {"prompt": prompt_text}
    response = requests.post(url, files=files, data=data)

# The response should include the processed file path or download URL
resp_json = response.json()
print(resp_json)

# Optionally, download the processed file automatically if your API returns a download path
if "processed_filename" in resp_json:
    download_url = f"https://changie-ai-backend.onrender.com/{resp_json['processed_filename']}"
    r = requests.get(download_url)
    with open("processed/downloaded_audio.m4a", "wb") as out_file:
        out_file.write(r.content)
    print("Processed audio downloaded to 'processed/downloaded_audio.m4a'")
