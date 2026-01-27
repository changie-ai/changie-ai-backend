#!/usr/bin/env python3
"""
test_vst.py — improved test runner that:
 - loads VST3s via pedalboard.load_plugin
 - tries to set sensible, audible parameter values
 - confirms which writes succeeded (prints current param value)
 - writes debug_* test tones AND the processed real file outputs
 - integrates a tempo tweak using librosa (with fallback)
"""

import os
import traceback
import numpy as np
import soundfile as sf
from pedalboard import Pedalboard, load_plugin
import librosa

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
UPLOADS = os.path.join(PROJECT_ROOT, "uploads")
PROCESSED = os.path.join(PROJECT_ROOT, "processed")
os.makedirs(PROCESSED, exist_ok=True)

PLUGIN_DIR = "/Library/Audio/Plug-Ins/VST3"
PLUGINS = {
    "changie_autotune": os.path.join(PLUGIN_DIR, "Graillon 3.vst3"),
    "changie_hype_delay": os.path.join(PLUGIN_DIR, "ValhallaDelay.vst3"),
    "changie_sick_verb": os.path.join(PLUGIN_DIR, "ValhallaRoom.vst3"),
    "changie_space_mod": os.path.join(PLUGIN_DIR, "ValhallaSpaceModulator.vst3"),
    "changie_timewarp": os.path.join(PLUGIN_DIR, "PaulXStretch.vst3"),
    # tempo tweak is Python-only and handled later
}

# ---------- utilities ----------
def ensure_stereo(arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = np.stack([arr, arr], axis=1)
    elif arr.ndim == 2 and arr.shape[1] == 1:
        arr = np.repeat(arr, 2, axis=1)
    return arr.astype(np.float32)

def write_file(path, arr, sr):
    sf.write(path, arr, sr)
    print(f"  -> wrote {path}")

def gen_sine(freq=440.0, sr=44100, dur=2.0):
    t = np.linspace(0, dur, int(sr*dur), endpoint=False)
    s = 0.5 * np.sin(2 * np.pi * freq * t)
    return ensure_stereo(s)

def gen_noise(sr=44100, dur=2.0):
    s = 0.35 * np.random.normal(size=int(sr*dur))
    return ensure_stereo(s)

def process_with_plugin(plugin, audio, sr):
    board = Pedalboard([plugin])
    out = board(audio, sr)
    return out

def dump_param_value(plugin, key):
    # Try to read a current value for diagnostic
    try:
        if key in plugin.parameters:
            return plugin.parameters[key].value
    except Exception:
        pass
    try:
        return getattr(plugin, key)
    except Exception:
        return "<unreadable>"

def try_set(plugin, key, val):
    """
    Attempt several strategies to set a plugin parameter and then read back current value.
    Returns tuple (ok:bool, current_value).
    """
    ok = False
    current = None
    # 1) try setattr
    try:
        setattr(plugin, key, val)
        ok = True
    except Exception:
        pass

    # 2) try plugin.parameters raw_value if present
    try:
        if key in plugin.parameters:
            plugin.parameters[key].raw_value = val
            ok = True
    except Exception:
        pass

    # read back current
    try:
        if key in plugin.parameters:
            current = plugin.parameters[key].value
        else:
            current = getattr(plugin, key)
    except Exception:
        current = None

    # normalize boolean-like values for print readability
    print(f"    set {key} = {val!r} -> {'OK' if ok else 'FAILED'} (current={current!r})")
    return ok, current

# ---------- per-plugin "safe" parameter sets ----------
SAFE_TESTS = {
    "changie_autotune": [
        ("bypass", False),
        ("correction", True),
        ("corr_amount", 100.0),
        ("snap_max_st", 0.1),
        ("inertia", 0.0),
        ("wet_mix_db", 6.0),   # push wet up
        ("dry_mix_db", 0.0),   # keep dry audible but allow effect
    ],
    "changie_hype_delay": [
        ("bypass", False),
        ("mix", 1.0),          # full wet to guarantee audible
        ("feedback", 0.95),
        ("delayl_ms", 350.0),
        ("delayr_ms", 700.0),
        ("moddepth", 0.6),
        ("modrate", 0.8),
    ],
    "changie_sick_verb": [
        ("bypass", False),
        ("mix", 1.0),
        # decay may be discrete; we'll attempt numeric but expect validation feedback
        ("predelay", 0.02),
        ("diffusion", 1.0),
    ],
    "changie_space_mod": [
        ("bypass", False),
        ("wetdry", 1.0),
        ("rate", 1.2),
        ("depth", 1.0),
        ("feedback", 0.8),
        ("manual", 1.0),
    ],
    "changie_timewarp": [
        ("bypass", False),
        ("stretch_amount", 4.0),      # large stretch to make effect audible
        ("num_harmonics", 8),
        ("pass_input_through", False), # force plugin to output processed audio
    ],
}

# ---------- main routine ----------
def process_all_effects(input_path):
    if not os.path.exists(input_path):
        print("ERROR: input not found:", input_path)
        return

    # read real input
    try:
        real, sr = sf.read(input_path, always_2d=True)
        # if sf.read returns shape (frames, channels) we're good; ensure float32 stereo
        if real.ndim == 1:
            real = ensure_stereo(real)
        else:
            # if mono -> duplicate; if stereo already ok
            if real.shape[1] == 1:
                real = np.repeat(real, 2, axis=1)
            real = real.astype(np.float32)
    except Exception as e:
        print("Failed reading input:", e)
        return

    print("Processing all effects from:", input_path)
    results = {}

    for logical_name, plugin_path in PLUGINS.items():
        print("\nProcessing", logical_name, "->", plugin_path)
        if not os.path.exists(plugin_path):
            print("  Plugin file NOT FOUND at path:", plugin_path)
            results[logical_name] = "MISSING_PLUGIN"
            continue

        try:
            plugin = load_plugin(plugin_path)
            print("  Plugin loaded.")
        except Exception as e:
            print("  Failed to load plugin:", e)
            traceback.print_exc()
            results[logical_name] = "LOAD_FAILED"
            continue

        # Dump a short readable parameter sample (first 40)
        try:
            keys = list(plugin.parameters.keys())
            print("  Sample parameters (first 40):")
            for i,k in enumerate(keys[:40]):
                try:
                    print(f"   - {k}: {plugin.parameters[k]}")
                except Exception:
                    try:
                        print(f"   - {k}: {getattr(plugin, k)}")
                    except Exception:
                        print(f"   - {k}: <unreadable>")
        except Exception:
            print("  (could not enumerate plugin.parameters)")

        # apply safe test values
        tests = SAFE_TESTS.get(logical_name, [("bypass", False)])
        for k,v in tests:
            try_set(plugin, k, v)

        # After setting, explicitly check wet/dry possibilities and ensure not muted
        # If plugin exposes wet/dry/dry_mix_db etc, make sure wet is audible and dry not -inf
        # (We already set common fields in SAFE_TESTS; here we just print current snapshot)
        print("  Post-set snapshot (sample):")
        for k,v in tests[:10]:
            current = dump_param_value(plugin, k)
            print(f"   * {k} -> {current!r}")

        # Create a short test signal that is likely to reveal effect:
        # - autotune/timewarp: use sine so pitch processing is obvious
        # - time-based effects: use noise
        if "autotune" in logical_name or "timewarp" in logical_name:
            test_signal = gen_sine(freq=440.0, sr=sr, dur=2.0)
        else:
            test_signal = gen_noise(sr=sr, dur=2.0)

        # process test tone
        try:
            processed_test = process_with_plugin(plugin, test_signal, sr)
            out_test = os.path.join(PROCESSED, f"debug_{logical_name}.wav")
            write_file(out_test, processed_test, sr)
        except Exception as e:
            print("  ERROR processing test tone:", e)
            traceback.print_exc()

        # process real file
        try:
            processed_real = process_with_plugin(plugin, real, sr)
            out_real = os.path.join(PROCESSED, f"{logical_name}_audiofile.wav")
            write_file(out_real, processed_real, sr)
            results[logical_name] = out_real
        except Exception as e:
            print("  ERROR processing real file:", e)
            traceback.print_exc()
            results[logical_name] = "PROCESS_FAILED"

    # Tempo tweak (python-only)
    print("\nProcessing changie_temp_tweak (librosa tempo analysis + adjust) ...")
    try:
        y, sr2 = librosa.load(input_path, sr=None, mono=True)
        # detect tempo
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr2)
        except Exception:
            # fallback to tempo() alias
            tempo = float(librosa.beat.tempo(y=y, sr=sr2)[0])
        print(f"  Detected tempo: {tempo:.2f} BPM")
        target_bpm = 120.0  # default target
        rate = target_bpm / float(tempo)
        print(f"  Adjusting tempo -> {target_bpm} BPM (rate {rate:.3f})")

        # try time_stretch; librosa.effects.time_stretch(y, rate) is the expected call
        try:
            y_stretched = librosa.effects.time_stretch(y, rate)
            out_t = os.path.join(PROCESSED, f"changie_temp_tweak_audiofile_adjusted_{int(round(target_bpm))}bpm.wav")
            sf.write(out_t, y_stretched, sr2)
            print("  wrote adjusted tempo file:", out_t)
            results["changie_temp_tweak"] = out_t
        except TypeError as te:
            print("  time_stretch call failed (TypeError), attempting resample fallback:", te)
            # fallback: naive resample then write (changes pitch)
            new_sr = int(sr2 * rate)
            y_resampled = librosa.resample(y, sr2, new_sr)
            # write resampled but keep sample rate at original to change playback speed
            out_t = os.path.join(PROCESSED, f"changie_temp_tweak_audiofile_adjusted_{int(round(target_bpm))}bpm_resample.wav")
            sf.write(out_t, y_resampled, new_sr)
            print("  wrote resample fallback file:", out_t)
            results["changie_temp_tweak"] = out_t

    except Exception as e:
        print("❌ Error processing changie_temp_tweak:", e)
        traceback.print_exc()
        results["changie_temp_tweak"] = "ERROR"

    # Summary
    print("\nALL RESULTS:")
    for k,v in results.items():
        print(f" - {k} : {v}")
    print("\nDiagnostic run complete. Check processed/ for debug_*.wav (test tones) and *_audiofile.wav (real outputs).")
    print("Paste the full terminal output here (especially post-set snapshots) and tell me whether debug_*.wav sounds affected.")

# convenience main
def main():
    test_input = os.path.join(UPLOADS, "audiofile.wav")
    process_all_effects(test_input)

if __name__ == "__main__":
    main()
