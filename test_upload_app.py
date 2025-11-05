import requests

# The URL points to your local Flask server (port 5001)
url = "https://changie-ai-backend.onrender.com/upload"

# Replace 'audiofile.m4a' with a real file in your 'uploads' folder
files = {'file': open('uploads/audiofile.m4a', 'rb')}
data = {'prompt': 'Increase volume and add slight reverb'}

response = requests.post(url, files=files, data=data)

print(response.json())
