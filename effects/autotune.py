# effects/autotune.py
import numpy as np
import pyrubberband as pyrb
import librosa

# midi math
def hz_to_midi(hz):
    return 69 + 12 * np.log2(hz / 440.0) if hz > 0 else None

def midi_to_hz(m):
    return 440.0 * (2.0 ** ((m - 69) / 12.0))

def quantize_hz_to_nearest(hz, allowed_notes=None):
    """
    allowed_notes: list of MIDI note numbers allowed (if None, use all chromatic)
    """
    if hz is None or hz <= 0:
        return None
    m = hz_to_midi(hz)
    if allowed_notes is None:
        return midi_to_hz(round(m))
    # find nearest allowed note
    diffs = [abs(m - n) for n in allowed_notes]
    idx = int(np.argmin(diffs))
    return midi_to_hz(allowed_notes[idx])

def get_standard_scale_midi(scale_name="chromatic"):
    # returns list of MIDI ints.
    if scale_name == "chromatic":
        return list(range(0, 128))
    # major scale (C major) as default template across octaves
    if scale_name == "major":
        notes = [0,2,4,5,7,9,11]
        mids = []
        for octave in range(0,11):
            base = 12*octave
            for n in notes:
                mids.append(base + n)
        return mids
    # fallback to chromatic
    return list(range(0,128))

def process_autotune(in_audio, sr, mode="hard", scale="chromatic", strength=1.0, hop_sec=0.02):
    """
    mode: 'soft' or 'hard' — hard snaps to quantized pitch more aggressively
    scale: 'chromatic' or 'major' etc.
    strength: 0..1 how aggressive
    hop_sec: frame hop for pitch detection (smaller = more accuracy but slower)
    """
    audio = in_audio.copy().astype(np.float32)
    if audio.ndim == 2 and audio.shape[1] > 1:
        mono = np.mean(audio, axis=1)
    else:
        mono = audio.flatten()
    hop_length = max(64, int(sr * hop_sec))
    f0, voiced_flag, voiced_probs = librosa.pyin(mono, fmin=librosa.note_to_hz('C2'),
                                                 fmax=librosa.note_to_hz('C7'),
                                                 sr=sr, hop_length=hop_length)
    # f0 is an array length ~ frames
    allowed = get_standard_scale_midi(scale)
    # Build per-frame target semitone shift
    frames = len(f0)
    # Convert framewise target frequency to semitone shift relative to current pitch
    semitone_shifts = np.zeros(frames)
    for i in range(frames):
        cur_hz = f0[i]
        if np.isnan(cur_hz):
            semitone_shifts[i] = 0.0
            continue
        target_hz = quantize_hz_to_nearest(cur_hz, allowed_notes=allowed)
        if target_hz is None:
            semitone_shifts[i] = 0.0
            continue
        cur_m = hz_to_midi(cur_hz)
        targ_m = hz_to_midi(target_hz)
        semitone_shifts[i] = (targ_m - cur_m) * float(strength)
    # Now apply block-wise pitch shift using pyrubberband
    # We'll process in chunks corresponding to frames then overlap-add
    frame_len = int(hop_length * 4)  # a multiple for overlap
    if frame_len < 1024:
        frame_len = 1024
    step = hop_length
    n = len(mono)
    out = np.zeros_like(mono, dtype=np.float32)
    weight = np.zeros_like(mono, dtype=np.float32)
    for i, frame_idx in enumerate(range(0, n, step)):
        start = frame_idx
        end = min(frame_idx + frame_len, n)
        chunk = mono[start:end]
        # find nearest f0 frame index for semitone shift
        fidx = min(frames-1, int(frame_idx // hop_length))
        semis = semitone_shifts[fidx]
        if abs(semis) < 0.01:
            # no shift
            out[start:end] += chunk
            weight[start:end] += 1.0
            continue
        try:
            shifted = pyrb.pitch_shift(chunk, sr, semis)
        except Exception:
            # fallback: no shift
            shifted = chunk
        L = len(shifted)
        out[start:start+L] += shifted
        weight[start:start+L] += 1.0
    # avoid div by zero
    weight[weight == 0] = 1.0
    mono_out = out / weight
    # Reconstruct stereo by copying to two channels (or if input was stereo try to preserve channels by same transform)
    if in_audio.ndim == 2 and in_audio.shape[1] == 2:
        stereo = np.column_stack([mono_out, mono_out])
    else:
        stereo = mono_out[:,None]
    # gentle normalization
    peak = np.max(np.abs(stereo)) or 1.0
    if peak > 0.99:
        stereo = stereo * (0.99 / peak)
    return stereo.astype(np.float32)
