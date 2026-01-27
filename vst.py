import os

class VSTHost:
    def __init__(self, plugin_path):
        self.plugin_path = plugin_path
        print(f"[VSTHost] Loaded plugin from: {plugin_path}")

    def set_parameter(self, name, value):
        print(f"[VSTHost] Set parameter '{name}' to {value}")

    def process_audio(self, input_audio):
        print(f"[VSTHost] Processing audio with {os.path.basename(self.plugin_path)}")
        # Simulate some audio processing
        return input_audio
