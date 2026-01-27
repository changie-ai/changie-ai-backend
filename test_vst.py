import os
import traceback
import numpy as np
import soundfile as sf
from pedalboard import Pedalboard, load_plugin

# === optional python dsp ===
import librosa

PLUGIN_DIR = "/Library/Audio/Plug-Ins/VST3"

PLUGINS = {
    "changie_autotune": os.path.join(PLUGIN_DIR, "Graillon 3.vst3"),
    "changie_hype_delay": os.path.join(PLUGIN_DIR, "ValhallaDelay.vst3"),
    "changie_sick_verb": os.path.join(PLUGIN_DIR, "ValhallaRoom.vst3"),
    "changie_space_mod": os.path.join(PLUGIN_DIR, "ValhallaSpaceModulator.vst3"),
    "changie_timewarp": os.path.join(PLUGIN_DIR, "PaulXStretch.vst3"),
}

def ensure_stereo(arr):
    if arr.ndim == 1:
        arr = np.stack([arr, arr], axis=1)
    elif arr.shape[1] == 1:
        arr = np.repeat(arr, 2, axis=1)
    return arr.astype(np.float32)

def process_with_plugin(plugin, audio, sr):
    board = Pedalboard([plugin])
    return board(audio, sr)

# === tempo tweak python dsp ===
def tempo_tweak(audio, sr, speed=1.25, pitch_shift=0):
    """Simple tempo & pitch adjustment using librosa."""
    print(f"🎚️ Tempo Tweak → speed {speed}, pitch {pitch_shift} semitones")
    # convert to mono for librosa, process, then return stereo
    mono = librosa.to_mono(audio.T)
    stretched = librosa.effects.time_stretch(mono, speed)
    shifted = librosa.effects.pitch_shift(stretched, sr, n_steps=pitch_shift)
    stereo = np.stack([shifted, shifted], axis=1)
    return stereo.astype(np.float32)

def apply_plugin(input_file, plugin_name, plugin_path, params):
    try:
        print(f"\n🎛️ Running {plugin_name}...")

        # special case: tempo tweak
        if plugin_name == "changie_tempo_tweak":
            audio, sr = sf.read(input_file, always_2d=True)
            audio = ensure_stereo(audio)
            processed = tempo_tweak(audio, sr,
                                    speed=params.get("speed", 1.25),
                                    pitch_shift=params.get("pitch_shift", 0))
        else:
            plugin = load_plugin(plugin_path)
            for k, v in params.items():
                try:
                    if k in plugin.parameters:
                        plugin.parameters[k].value = v
                        print(f"  set {k} = {v}")
                except Exception:
                    pass

            audio, sr = sf.read(input_file, always_2d=True)
            audio = ensure_stereo(audio)
            processed = process_with_plugin(plugin, audio, sr)

        os.makedirs("processed", exist_ok=True)
        out_path = os.path.join("processed", f"{plugin_name}_audiofile.wav")
        sf.write(out_path, processed, sr)
        print(f"✅ Saved processed file: {out_path}")

    except Exception as e:
        print(f"❌ Error running {plugin_name}: {e}")
        traceback.print_exc()

def process_all_effects(input_file):
    plugin_params = {
        "changie_autotune": {
            "bypass": False,
            "corr_amount": 100.0,
            "inertia": 0.0,
            "dry_mix_db": -10.0,
            "wet_mix_db": 6.0,
        },
        "changie_hype_delay": {
            "bypass": False,
            "mix": 0.6,
            "feedback": 0.7,
            "delayl_ms": 300.0,
            "delayr_ms": 600.0,
        },
        "changie_sick_verb": {
            "bypass": False,
            "mix": 0.5,
            "decay": 3.5,
            "diffusion": 1.0,
        },
        "changie_space_mod": {
            "bypass": False,
            "wetdry": 1.0,
            "depth": 0.8,
            "rate": 1.2,
        },
        "changie_timewarp": {
            "bypass": False,
            "stretch_amount": 3.0,
            "main_volume": 0.8,
        },
        "changie_tempo_tweak": {
            "speed": 1.25,
            "pitch_shift": 0,
        },
    }

    for name, path in PLUGINS.items():
        params = plugin_params.get(name, {})
        apply_plugin(input_file, name, path, params)

    # run the Python DSP last
    apply_plugin(input_file, "changie_tempo_tweak", None, plugin_params["changie_tempo_tweak"])
