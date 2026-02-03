# effects_engine.pymix = float(params.get("mix", 1.0))

import numpy as np
import librosa
import soundfile as sf
import warnings
from scipy import signal

# Optional high-quality tools (wrapped)
try:
    import pyrubberband as pyrb
    RUBBERBAND_OK = True
except Exception:
    RUBBERBAND_OK = False

try:
    import pyworld as pw
    PYWORLD_OK = True
except Exception:
    PYWORLD_OK = False

# Pedalboard plugins optional
try:
    from pedalboard import (
        Pedalboard, Chorus, Compressor, Delay, Distortion, Gain, Limiter,
        NoiseGate, Phaser, Reverb
    )
    PEDALBOARD_OK = True
except Exception:
    PEDALBOARD_OK = False

from scipy.signal import butter, sosfiltfilt

# ---------------------------
# Utilities
# ---------------------------
def ensure_mono(audio):
    if audio is None:
        return audio
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)

def ensure_stereo(audio):
    if audio is None:
        return audio
    if audio.ndim == 1:
        return np.column_stack([audio, audio])
    return audio


def normalize_audio(audio):
    """
    Engine-safe audio normalization.
    Ensures:
    - numpy array
    - mono (n,) or stereo (n,2)
    """
    if audio is None:
        return audio

    # unwrap accidental tuples
    if isinstance(audio, tuple):
        audio = audio[0]

    audio = np.asarray(audio, dtype=np.float32)

    # collapse weird shapes
    if audio.ndim > 2:
        audio = audio.mean(axis=0)

    return audio


def to_float32_for_pedal(audio):
    a = audio.astype(np.float32) if audio.dtype != np.float32 else audio
    return ensure_stereo(a)

def rms(x, eps=1e-9):
    if x is None:
        return 0.0
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    return np.sqrt(np.mean(x.astype(np.float64)**2) + eps)

def match_level(reference, proc, min_gain_db=-60.0, max_gain_db=24.0):
    if reference is None or proc is None:
        return proc
    ref_r = rms(reference)
    p_r = rms(proc)
    if p_r <= 1e-9 or ref_r <= 1e-9:
        return proc
    gain = ref_r / p_r
    gain_db = 20.0 * np.log10(gain + 1e-12)
    gain_db = np.clip(gain_db, min_gain_db, max_gain_db)
    lin = 10.0 ** (gain_db / 20.0)
    return (proc * lin).astype(np.float32)

# ---------------------------
# Filters (zero-phase butterworth)
# ---------------------------
def _sos_filter(audio, sr, cutoff, btype="low", order=8):
    if audio is None:
        return audio
    if cutoff <= 0 or cutoff >= sr/2:
        return audio
    try:
        sos = butter(order, cutoff/(sr/2), btype=btype, output="sos")
    except Exception:
        return audio
    if audio.ndim == 1:
        return sosfiltfilt(sos, audio)
    out = np.zeros_like(audio)
    out[:,0] = sosfiltfilt(sos, audio[:,0])
    out[:,1] = sosfiltfilt(sos, audio[:,1])
    return out

def apply_lowpass(audio, sr, cutoff_hz=800.0, order=8):
    out = _sos_filter(audio, sr, cutoff_hz, btype="low", order=order)
    # extra smoothing for extreme low cutoff
    if cutoff_hz <= 400:
        try:
            from scipy.ndimage import gaussian_filter1d
            if out.ndim == 1:
                out = gaussian_filter1d(out, sigma=3)
            else:
                out[:,0] = gaussian_filter1d(out[:,0], sigma=3)
                out[:,1] = gaussian_filter1d(out[:,1], sigma=3)
        except Exception:
            pass
    return out.astype(np.float32)

def apply_highpass(audio, sr, cutoff_hz=80.0, order=8):
    out = _sos_filter(audio, sr, cutoff_hz, btype="high", order=order)
    return out.astype(np.float32)

# ---------------------------
# Pitch / time helpers
# ---------------------------
def pitch_shift_audio(audio, sr, semitones=0.0):
    if audio is None:
        return audio
    # prefer rubberband if available
    if RUBBERBAND_OK:
        try:
            return pyrb.pitch_shift(audio, sr, semitones).astype(np.float32)
        except Exception:
            pass
    try:
        if audio.ndim == 1:
            return librosa.effects.pitch_shift(audio.astype(np.float32), sr, semitones).astype(np.float32)
        left = librosa.effects.pitch_shift(audio[:,0].astype(np.float32), sr, semitones)
        right = librosa.effects.pitch_shift(audio[:,1].astype(np.float32), sr, semitones)
        minlen = min(len(left), len(right))
        return np.column_stack([left[:minlen], right[:minlen]]).astype(np.float32)
    except Exception:
        return audio

def time_stretch_audio(audio, sr, rate=1.0, preserve_pitch=True):
    if audio is None:
        return audio
    if preserve_pitch and RUBBERBAND_OK:
        try:
            return pyrb.time_stretch(audio, sr, rate).astype(np.float32)
        except Exception:
            pass
    try:
        if audio.ndim == 1:
            return librosa.effects.time_stretch(audio.astype(np.float32), rate).astype(np.float32)
        l = librosa.effects.time_stretch(audio[:,0].astype(np.float32), rate)
        r = librosa.effects.time_stretch(audio[:,1].astype(np.float32), rate)
        m = min(len(l), len(r))
        return np.column_stack([l[:m], r[:m]]).astype(np.float32)
    except Exception:
        return audio

# ---------------------------
# Key detection (simple chroma)
# ---------------------------
_MAJOR_TEMPLATE = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
_MINOR_TEMPLATE = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
def detect_key(y, sr):
    mono = ensure_mono(y)
    if mono is None or len(mono) < 1024:
        return (0, "major")
    try:
        C = librosa.feature.chroma_cqt(y=mono.astype(np.float32), sr=sr)
        v = C.mean(axis=1)
        v = v / (v.sum() + 1e-9)
    except Exception:
        v = np.ones(12)/12.0
    best = (0,"major"); best_score = -1e9
    for root in range(12):
        maj = np.roll(_MAJOR_TEMPLATE, root)
        min_ = np.roll(_MINOR_TEMPLATE, root)
        smaj = np.dot(v, maj)
        smin = np.dot(v, min_)
        if smaj > best_score:
            best_score = smaj; best = (root, "major")
        if smin > best_score:
            best_score = smin; best = (root, "minor")
    return best

# ---------------------------
# PyWorld autotune helpers (HARDENED)
# ---------------------------
def quantize_f0_to_scale(f0, amount=1.0, root=0, mode="major"):
    if mode == "major":
        scale = np.array([0,2,4,5,7,9,11])
    else:
        scale = np.array([0,2,3,5,7,8,10])
    f0s = f0.copy()
    f0s[f0s<=0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        midi = 69 + 12*np.log2(f0s / 440.0)
    qm = np.full_like(midi, np.nan)
    for i,m in enumerate(midi):
        if np.isnan(m): continue
        semitone = int(round(m))
        base = (semitone//12)*12
        candidates = base + ((root + scale) % 12)
        candidates = np.concatenate([candidates-12, candidates, candidates+12])
        idx = np.argmin(np.abs(candidates - m))
        target = candidates[idx]
        qm[i] = (target * amount) + (m * (1.0 - amount))
    fq = 440.0 * 2**((qm - 69)/12.0)
    fq[np.isnan(fq)] = 0.0
    return fq

def _align_and_mix_original(mono_orig, y_synth, amount):
    """
    Ensure y_synth length matches mono_orig and mix according to amount.
    mono_orig: 1-D numpy (original mono)
    y_synth: 1-D numpy (synthesized by pyworld)
    amount: 0..1 (1 => full synth)
    Returns mixed mono float32, length == len(mono_orig)
    """
    n_orig = mono_orig.shape[0]
    n_y = y_synth.shape[0]
    if n_y < n_orig:
        # pad synth with zeros (safer than repeating)
        y_synth = np.pad(y_synth, (0, n_orig - n_y), mode='constant', constant_values=0.0)
    elif n_y > n_orig:
        # trim synth to original
        y_synth = y_synth[:n_orig]
    # mix
    mixed = (float(amount) * y_synth.astype(np.float32) + (1.0 - float(amount)) * mono_orig.astype(np.float32))
    # level-match to original to avoid envelopes disappearing
    mixed = match_level(mono_orig, mixed)
    return mixed.astype(np.float32)

def apply_pyworld_autotune(audio, sr, amount=1.0, key=None, mode=None, robotic=False):
    """
    amount: 0..1
    key: 0-11 or string like "C"
    robotic: if True, zero aperiodicity -> robotic tone
    """
    if not PYWORLD_OK:
        warnings.warn("pyworld not installed — autotune skipped")
        return audio
    mono_orig = ensure_mono(audio)
    if mono_orig is None or mono_orig.size == 0:
        return audio
    mono = mono_orig.astype(np.float64)

    # detect key if missing
    if key is None or mode is None:
        try:
            detected = detect_key(mono, sr)
            key, mode = detected
        except Exception:
            key, mode = 0, "major"

    if isinstance(key, str):
        NOTES = {"C":0,"C#":1,"DB":1,"D":2,"D#":3,"EB":3,"E":4,"F":5,"F#":6,"GB":6,"G":7,"G#":8,"AB":8,"A":9,"A#":10,"BB":10,"B":11}
        key = NOTES.get(key.upper(), 0)

    # analysis
    try:
        # harvest returns f0 (Hz) per frame and time array t
        f0, t = pw.harvest(mono, sr)
        sp = pw.cheaptrick(mono, f0, t, sr)
        ap = pw.d4c(mono, f0, t, sr)
    except Exception as e:
        warnings.warn(f"pyworld analysis failed: {e}")
        return audio

    # robotic: flatten aperiodicity
    if robotic:
        try:
            ap = np.zeros_like(ap)
        except Exception:
            pass

    # quantize f0 (snap to scale)
    f0_q = quantize_f0_to_scale(f0, amount=float(amount), root=int(key), mode=mode)

    # Synthesis: attempt to synthesize; if shapes mismatch, align safely
    try:
        y = pw.synthesize(f0_q.astype(np.float64), sp.astype(np.float64), ap.astype(np.float64), sr)
    except Exception as e:
        warnings.warn(f"pyworld synthesis failed: {e}")
        return audio

    # y is mono float64. Now align lengths with original mono and mix according to amount.
    try:
        mixed_mono = _align_and_mix_original(mono_orig, y, amount)
    except Exception as e:
        warnings.warn(f"autotune mixing/align failed: {e}")
        # as fallback, just return original
        return audio

    # return stereo version if original was stereo
    if audio.ndim == 2:
        stereo_mixed = np.column_stack([mixed_mono, mixed_mono]).astype(np.float32)
        return stereo_mixed
    return mixed_mono.astype(np.float32)

# ---------------------------
# Formant shift (pyworld-based if possible)
# ---------------------------
def formant_shift_audio(audio, sr, semitones=0.0):
    """
    True formant shift using pyworld spectral envelope warping.
    Pitch is preserved. Timbre changes only.
    """
    if audio is None:
        return audio

    # allow percentage-style input mapped externally
    semitones = float(semitones)
    if abs(semitones) < 0.05:
        return audio

    if not PYWORLD_OK:
        warnings.warn("pyworld not available — formant shift skipped")
        return audio

    mono = ensure_mono(audio).astype(np.float64)

    try:
        f0, t = pw.harvest(mono, sr)
        sp = pw.cheaptrick(mono, f0, t, sr)
        ap = pw.d4c(mono, f0, t, sr)

        factor = 2 ** (semitones / 12.0)

        n_frames, n_bins = sp.shape
        freqs = np.linspace(0, sr / 2.0, n_bins)

        new_sp = np.zeros_like(sp)

        for i in range(n_frames):
            frame = sp[i]

            # shift formants by remapping spectral envelope
            src_freqs = freqs / factor

            # clamp to valid range
            src_freqs = np.clip(src_freqs, freqs[0], freqs[-1])

            # sample original envelope at shifted positions
            new_sp[i] = np.interp(src_freqs, freqs, frame)

        y = pw.synthesize(f0, new_sp, ap, sr)

        if audio.ndim == 2:
            y = np.column_stack([y, y])

        return match_level(audio, y.astype(np.float32))

    except Exception as e:
        warnings.warn(f"formant_shift failed: {e}")
        return audio



# ---------------------------
# Harmonizer (multi-voice DSP)
# ---------------------------
def harmonizer(audio, sr, voices=None, key=None, intervals=None, mix=0.8):
    import numpy as np
    from librosa.effects import pitch_shift

    if audio is None:
        return audio

    # Ensure mono for processing
    mono = ensure_mono(audio)
    if mono is None:
        return audio

    # --- BUILD INTERVALS ---
    if intervals is None:
        if voices == 3:
            intervals = [0, 4, 7]
        elif voices == 4:
            intervals = [0, 4, 7, 12]
        elif voices == 5:
            intervals = [-12, -7, 0, 7, 12]
        else:
            intervals = [0, 7, 12]  # default 3-voice: root, 5th, octave

    # --- RENDER HARMONY LAYERS SAFELY ---
    voices_out = []
    for semis in intervals:
        try:
            shifted = pitch_shift_audio(mono, sr, semitones=semis)
            voices_out.append(shifted)
        except Exception as e:
            print(f"[HARMONIZER ERROR] interval {semis}: {e}")

    if not voices_out:
        return audio

    # --- SAFE STACK: force 1D layers, match length ---
    import numpy as np
    min_len = min([len(v) for v in voices_out])
    voices_out = [v[:min_len].reshape(-1) for v in voices_out]

    # Stack voices (shape = samples x voices)
    harmony_stack = np.stack(voices_out, axis=1)

    # Mix harmonies
    harmony_mix = np.sum(harmony_stack, axis=1) / np.sqrt(harmony_stack.shape[1])

    # Match original mono length
    mono = mono[:min_len]

    # Blend with original
    blend = ((1 - mix) * mono + mix * harmony_mix).astype(np.float32)
    
    # -----------------------------
    # FINAL OUTPUT SHAPE — DO NOT MOVE
    # -----------------------------
    
    # Always return consistent float32 array
    return np.asarray(blend, dtype=np.float32)


# ---------------------------
# Simple fallbacks and small effects
# ---------------------------
def reverse(audio): return audio[::-1]

def apply_gain(audio, gain_db=0.0):
    """
    Apply gain in decibels.
    +6 dB ≈ 2x louder
    -6 dB ≈ half volume
    """
    if audio is None:
        return audio

    gain = 10 ** (gain_db / 20.0)
    return (audio * gain).astype(np.float32)

def simple_limiter(audio, threshold_db=-6.0):
    if audio is None:
        return audio
    th = 10**(threshold_db/20.0)
    a = audio.copy().astype(np.float32)
    return np.clip(a, -th, th)

def simple_compressor(audio, threshold_db=-18.0, ratio=4.0, attack=0.01, release=0.05, sr=44100):
    if audio is None:
        return audio
    a = ensure_stereo(audio).astype(np.float32)
    thresh = 10**(threshold_db/20.0)
    alpha_a = np.exp(-1.0/(sr*attack))
    alpha_r = np.exp(-1.0/(sr*release))
    env_prev = 0.0
    out = np.zeros_like(a)
    for i in range(a.shape[0]):
        s = max(abs(a[i,0]), abs(a[i,1]))
        env = (alpha_a*env_prev + (1-alpha_a)*s) if s > env_prev else (alpha_r*env_prev + (1-alpha_r)*s)
        env_prev = env
        if env > thresh:
            gain = 1.0 / (1.0 + (env/thresh - 1.0)*(ratio-1.0))
        else:
            gain = 1.0
        out[i,0] = a[i,0]*gain
        out[i,1] = a[i,1]*gain
    return out.astype(np.float32)

def simple_distortion(audio, drive_db=20.0, mix=1.0):
    if audio is None:
        return audio

    a = ensure_stereo(audio).astype(np.float32)

    # Convert dB drive into aggressive linear gain
    drive = 10 ** (drive_db / 20.0)

    # Harder waveshaping
    driven = a * drive

    # Clip + tanh combo (much more audible)
    clipped = np.clip(driven, -1.0, 1.0)
    shaped = np.tanh(clipped * 2.5)

    # Wet / dry blend
    out = mix * shaped + (1.0 - mix) * a

    return out.astype(np.float32)


def simple_chorus(audio, sr, depth=0.7, rate_hz=1.5, mix=0.5):
    if audio is None:
        return audio
    mono = ensure_mono(audio).astype(np.float32)
    n = len(mono)
    delay = int(0.02*sr)
    v1 = np.roll(mono, delay//2)
    v2 = np.roll(mono, delay)
    try:
        v1s = librosa.effects.pitch_shift(v1, sr, 0.2)
        v2s = librosa.effects.pitch_shift(v2, sr, -0.2)
    except Exception:
        v1s, v2s = v1, v2
    mix_total = (1.0-mix)*mono[:v1s.shape[0]] + (mix/2.0)*v1s[:v1s.shape[0]] + (mix/2.0)*v2s[:v2s.shape[0]]
    if audio.ndim==2:
        return np.column_stack([mix_total,mix_total]).astype(np.float32)
    return mix_total.astype(np.float32)

def simple_phaser(audio, sr, depth=0.7, rate_hz=0.8, mix=0.5):
    if audio is None:
        return audio
    mono = ensure_mono(audio).astype(np.float32)
    n = len(mono)
    lfo = 0.5 + 0.5*np.sin(2*np.pi*rate_hz*np.arange(n)/sr)
    max_delay = int(0.005*sr)
    out = np.zeros_like(mono)
    for i in range(n):
        d = int(lfo[i]*max_delay)
        j = i-d
        out[i] = mono[i] - 0.5*(mono[j] if j>=0 else 0.0)
    mixed = (1.0-mix)*mono + mix*out
    if audio.ndim==2:
        return np.column_stack([mixed,mixed]).astype(np.float32)
    return mixed.astype(np.float32)


def bitcrush(audio, bits=6, sample_rate=8000, mix=1.0, original_sr=44100):
    """
    Simple, correct retro bitcrusher:
    - sample & hold downsampling
    - integer amplitude quantization
    """

    if audio is None:
        return audio

    x = audio.astype(np.float32)

    # Stereo-safe
    if x.ndim == 2:
        out = np.zeros_like(x)
        for ch in range(x.shape[1]):
            out[:, ch] = bitcrush(
                x[:, ch],
                bits=bits,
                sample_rate=sample_rate,
                mix=mix,
                original_sr=original_sr
            )
        return out

    # ------------------------------------------------
    # 1. PRE-GAIN (FORCE EVERYTHING INTO THE CRUSHER)
    # ------------------------------------------------
    x = np.clip(x * 10.0, -1.0, 1.0)

    # ------------------------------------------------
    # 2. SAMPLE & HOLD (HARD DECIMATION)
    # ------------------------------------------------
    ratio = max(1, int(original_sr // sample_rate))
    held = x[::ratio]
    crushed = np.repeat(held, ratio)[:len(x)]

    # ------------------------------------------------
    # 3. HARD QUANTIZATION (NO ROUNDING)
    # ------------------------------------------------
    levels = 2 ** bits
    crushed = np.floor(crushed * levels) / levels

    # ------------------------------------------------
    # 4. FULL WET — NO CLEAN SIGNAL EVER
    # ------------------------------------------------
    return np.clip(crushed, -1.0, 1.0).astype(np.float32)


def stereo_widen(audio, width=1.0):
    """
    Stereo widening using Mid/Side processing.
    width = 0.0  -> mono
    width = 1.0  -> original
    width > 1.0  -> wider
    """

    if audio is None:
        return audio

    # Must be stereo to widen
    if audio.ndim != 2 or audio.shape[1] != 2:
        return audio

    left = audio[:, 0]
    right = audio[:, 1]

    # Mid / Side
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)

    # Apply width
    side *= width

    # Back to L/R
    out_left = mid + side
    out_right = mid - side

    return np.stack([out_left, out_right], axis=1).astype(np.float32)



def reverse_audio(audio):
    """
    Reverse audio in time.
    Stereo-safe.
    """
    if audio is None:
        return audio

    # Mono
    if audio.ndim == 1:
        return audio[::-1].copy()

    # Stereo
    if audio.ndim == 2:
        return audio[::-1, :].copy()

    return audio

# ---------------------------
# Pedalboard helper
# ---------------------------
def run_pedalboard_plugins(audio, sr, plugin_list):
    if not PEDALBOARD_OK or not plugin_list:
        return audio
    try:
        board = Pedalboard(plugin_list)
        inp = to_float32_for_pedal(audio)
        out = board(inp, sr)
        return ensure_stereo(out).astype(np.float32)
    except Exception as e:
        warnings.warn(f"Pedalboard error: {e}")
        return audio

# ---------------------------
# Interpreter helper: if plugin chain is raw string you can optionally interpret here.
# (Your prompt_parser handles GPT-based mapping; keep this minimal fallback.)
# ---------------------------
def interpret_prompt_to_chain(prompt):
    p = prompt.lower()
    chain = []


    # ---------- AUTOTUNE ----------
    if "t-pain" in p or "t pain" in p or "robot" in p or "robotic" in p:
        amt = 1.0 if ("max" in p or "100%" in p) else 0.9 if "heavy" in p else 0.7
        chain.append({"effect":"autotune","params":{"amount":amt,"robotic":True}})

    # ---------- HARMONIZER ----------
    if "harmon" in p or "choir" in p or "voices" in p:
        if "5" in p or "five" in p:
            voices = [-12, -5, 0, 5, 12]
        elif "3" in p or "three" in p:
            voices = [-12, 0, 12]
        else:
            voices = [-12, -7, 0, 7, 12]
        chain.append({"effect":"harmonizer","params":{"intervals":voices,"mix":0.8}})

    # ---------- PITCH SHIFT ----------
    if "chipmunk" in p or ("pitch" in p and "up" in p):
        chain.append({"effect":"pitch_shift","params":{"semitones":12}})
    if "monster" in p or ("pitch" in p and "down" in p):
        chain.append({"effect":"pitch_shift","params":{"semitones":-12}})

    # ---------- FILTERS ----------
    if "underwater" in p or "muffled" in p:
        chain.append({"effect":"lp_filter","params":{"cutoff_freq":300}})
    if "telephone" in p or "tiny radio" in p:
        chain.append({"effect":"hp_filter","params":{"cutoff_freq":3000}})
    
    
    # ---------- DELAY ----------
    if "delay" in p or "echo" in p:
        chain.append({"effect":"hype_delay","params":{"delay_ms":1200,"feedback":0.8,"mix":0.7}})

    # ---------- REVERB ----------
    if "reverb" in p or "cathedral" in p:
        chain.append({"effect":"sick_verb","params":{"room_size":0.98,"mix":0.9}})
    
    # ------- DISTORTION --------
    if ("distort" in p or "overdrive" in p or "saturat" in p or 
        "megaphone" in p or "blown" in p or "destroy" in p or "gritty" in p
       and "bitcrush" not in p
    ):
    
        if "light" in p or "subtle" in p:
           drive = 8
           mix = 0.5
        elif "destroy" in p or "blown" in p or "extreme" in p:
           drive = 35
           mix = 1.0
        else:
           drive = 20
           mix = 0.8

        chain.append({
           "effect": "distortion",
           "params": {"drive_db": drive, "mix": mix}
        })

        # ---------- NATURAL LANGUAGE HARMONY WITH KEY ----------
    import re

    # e.g. "3 part harmony in the key of F minor"
    m = re.search(r"(\d+)\s*part harmony(?: in the key of ([a-g][b#]?(?:\s*major|\s*minor)?))?", p)
    if m:
        voices = int(m.group(1))
        key = m.group(2).strip() if m.group(2) else None
        chain.append({
            "effect": "harmonizer",
            "params": {
                "voices": voices,
                "key": key,
                "mix": 0.9
            }
        })

    # ---------- NATURAL LANGUAGE KEY CHANGES ----------
    # e.g. "change to the key of C major"
    m = re.search(r"(?:change|shift|move) to the key of ([a-g][b#]?(?:\s*major|\s*minor)?)", p)
    if m:
        target_key = m.group(1).strip()
        chain.append({
            "effect": "key_change",
            "params": {"target_key": target_key}
        })

    return chain



# ---------------------------
# Main apply_effect_chain
# ---------------------------
def normalize_bitcrush_params(params):
    """
    Normalize GPT params for the NEW bitcrush DSP
    """

    intensity = (
        str(params.get("intensity", ""))
        or str(params.get("amount", ""))
        or str(params.get("strength", ""))
    ).lower()

    # Defaults (retro but usable)
    bits = 4
    sample_rate = 6000
    mix = float(params.get("mix", 0.6))

    if any(x in intensity for x in ["subtle", "light", "barely"]):
        bits = 6
        sample_rate = 11025
        mix = 0.6
    elif "medium" in intensity:
        bits = 5
        sample_rate = 8000
        mix = 0.8
    elif any(x in intensity for x in ["heavy", "hard"]):
        bits = 4
        sample_rate = 6000
        mix = 1.0
    elif any(x in intensity for x in ["extreme", "destroy", "max"]):
        bits = 2
        sample_rate = 3000
        mix = 1.0
    elif any(x in intensity for x in ["nes", "gameboy", "8-bit", "video game"]):
        bits = 3
        sample_rate = 4000
        mix = 1.0

    # Explicit GPT overrides ALWAYS win
    bits = int(params.get("bits", params.get("bit_depth", bits)))
    sample_rate = int(params.get("sample_rate", sample_rate))

    return bits, sample_rate, mix


def apply_effect_chain(orig_audio, sr, chain):
    """
    orig_audio: np.ndarray (n,) or (n,2) or a prompt string (handled by prompt_parser normally)
    sr: sample rate
    chain: list-of-steps OR a raw prompt string
    """
 
    import shutil, sys, time

    print("🔥 APPLY_EFFECT_CHAIN ENTERED 🔥", flush=True)
    print("AUTOTUNE CHECK — rubberband path:", shutil.which("rubberband"), flush=True)
    sys.stdout.flush()
    time.sleep(0.2)
    
    if orig_audio is None:
        return orig_audio

    # If chain is a raw string, interpret it (fallback)
    if isinstance(chain, str):
        chain = interpret_prompt_to_chain(chain)

    audio = orig_audio.copy()
    pending = []

    
    def flush():
        nonlocal audio, pending
        if pending:
            audio = run_pedalboard_plugins(audio, sr, pending)
            pending = []

    if not isinstance(chain, list):
        return audio

 
    # FORCE bitcrush to run last so it stays crunchy
    chain = sorted(chain, key=lambda s: 1 if s.get("effect") == "bitcrush" else 0)

    for step in chain:
        if isinstance(step, str):
            continue
        eff = (step.get("effect") or "").lower()
        params = step.get("params", {}) or {}

        # AUTOTUNE
        if eff in ("autotune","robotic_autotune","robot_autotune"):
            flush()
            amt = float(params.get("amount", 1.0))
            key = params.get("key", None)
            mode = params.get("mode", None)
            robotic = bool(params.get("robotic", params.get("robot", False)))
            try:
                audio = apply_pyworld_autotune(audio, sr, amount=amt, key=key, mode=mode, robotic=robotic)
            except Exception as e:
                warnings.warn(f"autotune failed: {e}")
        
        # REVERSE
        elif eff in ("reverse", "reverse_audio", "backwards"):
            flush()
            audio = reverse_audio(audio)
 
       
        # FORMANT SHIFT
        elif eff in ("formant_shift","formant"):
            flush()
            semis = float(params.get("semitones", params.get("semitone", 0.0)))
            try:
                audio = formant_shift_audio(audio, sr, semitones=semis)
            except Exception as e:
                warnings.warn(f"formant shift failed: {e}")


        # ---------------------------
        # HARMONIZER (multi-voice)
        # ---------------------------
        elif eff in ("harmony","harmonizer"):
            flush()
            key = params.get("key", None)
            voices = params.get("voices", params.get("v", 1))

            print(f"[HARMONIZER] requested | key={key} voices={voices}")

            # --- natural language voice detection ---
            spoken_to_number = {
                "one": 1, "two": 2, "three": 3, "third": 3, "3": 3,
                "four": 4, "quad": 4, "4": 4,
                "five": 5, "5": 5
            }

            # Try to detect numbers in prompt
            for word in str(params).lower().split():
                if word in spoken_to_number:
                    voices = spoken_to_number[word]            

            default_intervals = {
                1: [0],
                2: [+4],
                3: [+4, +7],
                4: [+4, +7, +12],
                5: [+4, +7, +12, +19]
            }

            try:
                voices = int(voices)
                intervals = default_intervals.get(voices, default_intervals[3])
            except:
                intervals = default_intervals[3]

            # Key mapping (root-shift)
            KEY_MAP = {
                "c": 0, "c#": 1, "db": 1,
                "d": 2, "d#": 3, "eb": 3,
                "e": 4,
                "f": 5, "f#": 6, "gb": 6,
                "g": 7, "g#": 8, "ab": 8,
                "a": 9, "a#": 10, "bb": 10,
                "b": 11,
            }

            if key:
                key_name = key.lower().split()[0]
                root = KEY_MAP.get(key_name, None)
                if root is not None:
                    intervals = [i + root for i in intervals]
                print(f"[HARMONIZER] mapped intervals: {intervals}")

            try:
                h = harmonizer(
                    audio=audio,
                    sr=sr,
                    voices=voices,
                    key=key,
                    intervals=intervals,
                    mix=0.8
                )

                audio = h.astype(np.float32)
                print("[HARMONIZER] SUCCESS")

            except Exception as e:
                print(f"[HARMONIZER] FAILED: {e}")

            
        # PITCH SHIFT
        elif eff in ("pitch_shift","pitch"):
            flush()
            semis = float(params.get("semitones", params.get("semitone", 0.0)))
            try:
                audio = pitch_shift_audio(audio, sr, semis)
            except Exception as e:
                warnings.warn(f"pitch shift failed: {e}")

        # TIME / TEMPO
        elif eff in ("tempo_tweak","time","speed","timestretch"):
            flush()
            factor = float(params.get("factor", params.get("speed", 1.0)))
            preserve = bool(params.get("preserve_pitch", True))
            try:
                audio = time_stretch_audio(audio, sr, factor, preserve_pitch=preserve)
            except Exception as e:
                warnings.warn(f"tempo failed: {e}")

        # DELAY
        elif eff in ("hype_delay","delay","pingpong_delay","slap_delay"):
            delay_ms = float(params.get("delay_ms", params.get("delay", 300)))
            fb = float(params.get("feedback", params.get("fb", 0.5)))
            mixv = float(params.get("mix", 0.5))
            try:
                if PEDALBOARD_OK:
                    pending.append(Delay(delay_seconds=delay_ms/1000.0, feedback=fb, mix=mixv))
                else:
                    # simple echo fallback (single tap)
                    mono = ensure_mono(audio).astype(np.float32)
                    d = int(sr*(delay_ms/1000.0))
                    echoes = np.zeros_like(ensure_stereo(mono))
                    if d < len(mono):
                        e = np.roll(mono, d) * fb
                        echoes[:e.shape[0],0] += e; echoes[:e.shape[0],1] += e
                        audio = (audio*(1.0-mixv) + echoes*mixv).astype(np.float32)
            except Exception as e:
                warnings.warn(f"delay failed: {e}")

        # REVERB
        elif eff in ("sick_verb","reverb","big_verb","small_room"):
            room = float(params.get("room_size", params.get("room", 0.8)))
            wet = float(params.get("mix", 0.5))
            damping = float(params.get("damping", 0.5))
            try:
                if PEDALBOARD_OK:
                    pending.append(Reverb(room_size=room, damping=damping, wet_level=wet, dry_level=1.0, width=1.0))
                else:
                    mono = ensure_mono(audio).astype(np.float32)
                    tail = np.concatenate([np.zeros(int(0.01*sr)), np.linspace(1.0,0.0,int(0.3*sr))])
                    conv = np.convolve(mono, tail, mode='full')[:mono.shape[0]]
                    combined = (1.0-wet)*mono + wet*conv[:mono.shape[0]]
                    audio = ensure_stereo(combined).astype(np.float32)
            except Exception as e:
                warnings.warn(f"reverb failed: {e}")

        # PHASER / SPACE MOD
        elif eff in ("space_mod","phaser","flanger"):
            rate = float(params.get("rate", 0.5))
            depth = float(params.get("depth", 0.7))
            mixv = float(params.get("mix", 0.5))
            try:
                if PEDALBOARD_OK:
                    pending.append(Phaser(rate_hz=rate, depth=depth, feedback=params.get("feedback",0.2), mix=mixv))
                else:
                    audio = simple_phaser(audio, sr, depth=depth, rate_hz=rate, mix=mixv)
            except Exception as e:
                warnings.warn(f"phaser failed: {e}")

        # CHORUS
        elif eff == "chorus":
            rate = float(params.get("rate", 1.5))
            depth = float(params.get("depth", 0.7))
            mixv = float(params.get("mix", 0.5))
            try:
                if PEDALBOARD_OK:
                    pending.append(Chorus(rate_hz=rate, depth=depth, feedback=0.0, mix=mixv))
                else:
                    audio = simple_chorus(audio, sr, depth=depth, rate_hz=rate, mix=mixv)
            except Exception as e:
                warnings.warn(f"chorus failed: {e}")

        # STEREO WIDEN
        elif eff in ("stereo_widen", "widen", "stereo_spread"):
            flush()
            width = float(params.get("width", params.get("amount", 1.3)))
            try:
                audio = stereo_widen(audio, width=width)
            except Exception as e:
                warnings.warn(f"stereo widen failed: {e}")
        

        # COMPRESSOR
        elif eff == "compressor":
            thresh = float(params.get("threshold_db", params.get("threshold", -18)))
            ratio = float(params.get("ratio", 4.0))
            attack = float(params.get("attack", 0.01))
            release = float(params.get("release", 0.05))
            try:
                if PEDALBOARD_OK:
                    pending.append(Compressor(threshold_db=thresh, ratio=ratio, attack_ms=attack*1000, release_ms=release*1000))
                else:
                    audio = simple_compressor(ensure_stereo(audio), threshold_db=thresh, ratio=ratio, attack=attack, release=release, sr=sr)
            except Exception as e:
                warnings.warn(f"compressor failed: {e}")

        # DISTORTION / OVERDRIVE
        elif eff in ("distortion","overdrive","saturation"):
            drive = float(params.get("drive", params.get("drive_db", 20)))
            mixv = float(params.get("mix", 1.0))
            try:
                if PEDALBOARD_OK:
                    pending.append(Distortion(drive_db=drive))
                    # force pedalboard render immediately so mix applies
                    flush()
                    audio = simple_distortion(audio, drive_db=drive, mix=mixv)
                else:
                    audio = simple_distortion(ensure_stereo(audio), drive_db=drive, mix=mixv)
            except Exception as e:
                warnings.warn(f"distortion failed: {e}")

        # BITCRUSH
        elif eff == "bitcrush":
            bits, sample_rate, mix = normalize_bitcrush_params(params)
            audio = bitcrush(
                audio,
                bits=bits,
                sample_rate=sample_rate,
                mix=mix,
                original_sr=sr
            )


        # GAIN
        elif eff in ("gain", "volume", "boost"):
            flush()
            gain_db = float(params.get("db", params.get("gain_db", params.get("amount", 0.0))))
            try:
                audio = apply_gain(audio, gain_db=gain_db)
            except Exception as e:
                warnings.warn(f"gain failed: {e}")


        # HPF / LPF
        elif eff in ("hp_filter","highpass","hp"):
            cutoff = float(params.get("cutoff_freq", params.get("cutoff", 80)))
            audio = apply_highpass(audio, sr, cutoff_hz=cutoff)
        elif eff in ("lp_filter","lowpass","lp"):
            cutoff = float(params.get("cutoff_freq", params.get("cutoff", 800)))
            audio = apply_lowpass(audio, sr, cutoff_hz=cutoff)

        # LIMITER
        elif eff == "limiter":
            try:
                if PEDALBOARD_OK:
                    pending.append(Limiter())
                else:
                    audio = simple_limiter(audio, threshold_db=float(params.get("threshold_db",-6.0)))
            except Exception:
                warnings.warn("limiter failed")

        # KEY DETECTION
        elif eff in ("key_detect","key","detect_key"):
            flush()
            try:
                root, scale = detect_key(audio, sr)
                # Store result for logs / prompt response
                results["detected_key"] = (int(root), str(scale))
                print(f"[KEY DETECTION] → Root: {root}, Scale: {scale}")
            except Exception as e:
                warnings.warn(f"key detection failed: {e}")
       

        # NOISE GATE
        elif eff == "noise_gate":
            th_db = float(params.get("threshold_db", params.get("threshold", -50)))
            try:
                if PEDALBOARD_OK:
                    pending.append(NoiseGate(threshold_db=th_db))
                else:
                    a = audio.copy()
                    env = np.abs(ensure_mono(a))
                    mask = env > (10**(th_db/20.0))
                    if a.ndim==1:
                        a = a * mask
                    else:
                        a[:,0] *= mask; a[:,1] *= mask
                    audio = a.astype(np.float32)
            except Exception as e:
                warnings.warn(f"noise_gate failed: {e}")

        # Unknown effect: ignore silently
        else:
            # silent ignore
            pass

        # 🔒 SAFETY: enforce valid audio shape
        audio = normalize_audio(audio)

    # flush pending pedalboard plugins at end
    try:
        flush()
    except Exception:
        pass

    return ensure_stereo(audio).astype(np.float32)
# end of file
