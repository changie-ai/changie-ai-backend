import requests

# The URL of your local Flask server
url = "http://127.0.0.1:5000/upload"

# Path to a test audio file in the uploads folder
file_path = "uploads/audiofile.m4a"  # Make sure this exists

# Example prompt
prompt_text = "Increase volume and add slight reverb"

# Open the file in binary mode and send the POST request
with open(file_path, "rb") as f:
    files = {"file": f}
    data = {"prompt": prompt_text}
    response = requests.post(url, files=files, data=data)

print(response.json())
