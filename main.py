# main.py
from api import app
import uvicorn
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Fly sets this automatically
    uvicorn.run(app, host="0.0.0.0", port=port)