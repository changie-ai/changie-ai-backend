from pedalboard import load_plugin

# Paths to your 5 VST3 plugins
vst_paths = {
    "changie_autotune": "/Library/Audio/Plug-Ins/VST3/Graillon 3.vst3",
    "changie_hype_delay": "/Library/Audio/Plug-Ins/VST3/ValhallaDelay.vst3",
    "changie_sick_verb": "/Library/Audio/Plug-Ins/VST3/ValhallaRoom.vst3",
    "changie_space_mod": "/Library/Audio/Plug-Ins/VST3/ValhallaSpaceModulator.vst3",
    "changie_timewarp": "/Library/Audio/Plug-Ins/VST3/PaulXStretch.vst3",
}

print("==== VST Plugin Parameters ====")
for name, path in vst_paths.items():
    try:
        plugin = load_plugin(path)
        print(f"\n{name} parameters:")
        for key in plugin.parameters.keys():
            print(f"  - {key}")
    except Exception as e:
        print(f"❌ Could not load {name} ({path}): {e}")

# For the Python tempo tweak (librosa-based), we can just list intended parameters manually
tempo_tweak_params = [
    "target_bpm",      # target BPM to adjust to
    "preserve_pitch",  # True/False, whether pitch stays the same
    "slow_factor",     # optional, for slowing down
    "fast_factor",     # optional, for speeding up
]
print("\nchangie_temp_tweak parameters:")
for param in tempo_tweak_params:
    print(f"  - {param}")
