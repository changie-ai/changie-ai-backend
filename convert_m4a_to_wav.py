from pydub import AudioSegment

# Paths
input_path = "uploads/audiofile.m4a"
output_path = "uploads/audiofile.wav"

# Load the M4A file
audio = AudioSegment.from_file(input_path, format="m4a")

# Convert to standard WAV (stereo, 44.1kHz)
audio = audio.set_channels(2)
audio = audio.set_frame_rate(44100)

# Export as WAV
audio.export(output_path, format="wav")

print(f"✅ Converted {input_path} to {output_path}")
