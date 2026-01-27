# revChat.py
import requests
import os

class RevChat:
    """
    Simple class to call ChatGPT using your free session token
    """
    def __init__(self, session_token=None):
        # Grab session token from environment if not provided
        self.session_token = session_token or os.environ.get("CHATGPT_SESSION_TOKEN")
        if not self.session_token:
            raise ValueError("Please provide CHATGPT_SESSION_TOKEN as env variable or argument")
        self.base_url = "https://chat.openai.com/backend-api/conversation"

    def ask(self, prompt):
        """
        Sends the prompt to ChatGPT free account via web session
        """
        headers = {
            "Authorization": f"Bearer {self.session_token}",
            "Content-Type": "application/json",
        }
        data = {
            "prompt": prompt,
        }
        # This is a simplified example; for actual requests you'll need
        # to emulate the chat endpoint or use revChat helper library
        # This placeholder returns prompt for testing
        return {
            "effect_instructions": prompt
        }
