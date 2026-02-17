# effects/aa_fx.py
import numpy as np
import librosa


def _stutter(chunk, repeats=4):
    """Repeat a tiny slice of the chunk rhythmically."""
    slice_len = max(1, len(chunk) // repeats)
    grain = chunk[:slice_len]
    return np.tile(grain, repeats + 1)[:len(chunk)]


def _tape_stop(chunk, sr):
    """Simulate a turntable/tape slowing to a halt."""
    n = len(chunk)
    # rate curve from 1.0 down to ~0.05
    rate_curve = np.linspace(1.0, 0.05, n)
    out = np.zeros(n, dtype=np.float32)
    read_pos = 0.0
    for i in range(n):
        idx = int(read_pos)
        if idx >= n - 1:
            break
        frac = read_pos - idx
        out[i] = chunk[idx] * (1 - frac) + chunk[idx + 1] * frac
        read_pos += rate_curve[i]
    # fade volume down with the slowdown
    out *= np.linspace(1.0, 0.0, n) ** 0.5
    return out


def _pitch_warp(chunk, sr):
    """Warp pitch up then back down across the slice."""
    n = len(chunk)
    half = n // 2
    # accelerate then decelerate
    rate_up = np.linspace(1.0, 1.8, half)
    rate_down = np.linspace(1.8, 1.0, n - half)
    rate_curve = np.concatenate([rate_up, rate_down])
    out = np.zeros(n, dtype=np.float32)
    read_pos = 0.0
    for i in range(n):
        idx = int(read_pos)
        if idx >= n - 1:
            break
        frac = read_pos - idx
        out[i] = chunk[idx] * (1 - frac) + chunk[idx + 1] * frac
        read_pos += rate_curve[i]
    return out


def _granular_scatter(chunk, sr, grain_ms=30.0):
    """Scramble tiny grains within the slice for textural chaos."""
    grain_len = max(1, int(sr * grain_ms / 1000.0))
    n = len(chunk)
    num_grains = max(1, n // grain_len)
    grains = [chunk[i * grain_len:(i + 1) * grain_len] for i in range(num_grains)]
    # shuffle grain order
    rng_state = np.random.get_state()
    np.random.shuffle(grains)
    # apply tiny fade to each grain to avoid clicks
    for i, g in enumerate(grains):
        if len(g) > 8:
            fade = min(len(g) // 4, 64)
            g[:fade] *= np.linspace(0, 1, fade)
            g[-fade:] *= np.linspace(1, 0, fade)
    out = np.concatenate(grains)
    # pad or trim to original length
    if len(out) < n:
        out = np.pad(out, (0, n - len(out)))
    return out[:n]


def _gate_chop(chunk, sr, divisions=8):
    """Rhythmic volume gate — hard on/off pattern."""
    n = len(chunk)
    step = max(1, n // divisions)
    out = chunk.copy()
    for i in range(divisions):
        start = i * step
        end = min(start + step, n)
        # silence odd-numbered divisions
        if i % 2 == 1:
            out[start:end] *= 0.0
    return out


def process_aa_fx(in_audio, sr,
                  chaos=0.5, density=2.0, stutter_weight=1.0,
                  tapestop_weight=1.0, reverse_weight=1.0,
                  pitchwarp_weight=1.0, scatter_weight=0.7,
                  gate_weight=0.5, mix=0.85, seed=None):
    """
    Beat-synced glitch machine.

    Detects tempo, slices audio at beat subdivisions, then randomly
    mutates each slice with stutters, tape-stops, reverses, pitch warps,
    granular scatter, and gate chops. Reassembles into controlled chaos.

    Params:
      chaos:       0..1 probability any given slice gets glitched
      density:     beat subdivisions (1=whole beats, 2=eighth, 4=sixteenth)
      *_weight:    relative probability for each glitch type
      mix:         wet/dry blend
      seed:        random seed for reproducibility
    """
    audio = in_audio.copy().astype(np.float32)
    if audio.ndim == 2 and audio.shape[1] > 1:
        mono = np.mean(audio, axis=1)
    else:
        mono = audio.flatten()

    n = len(mono)
    rng = np.random.RandomState(seed)

    # --- detect tempo and beats ---
    try:
        tempo = float(librosa.beat.tempo(y=mono, sr=sr)[0])
    except Exception:
        tempo = 120.0

    # samples per beat, then subdivide
    samples_per_beat = int(60.0 / tempo * sr)
    slice_len = max(256, int(samples_per_beat / max(0.25, density)))

    # --- build the glitch menu with weights ---
    effects_menu = []
    weights = []

    if stutter_weight > 0:
        effects_menu.append("stutter")
        weights.append(stutter_weight)
    if tapestop_weight > 0:
        effects_menu.append("tapestop")
        weights.append(tapestop_weight)
    if reverse_weight > 0:
        effects_menu.append("reverse")
        weights.append(reverse_weight)
    if pitchwarp_weight > 0:
        effects_menu.append("pitchwarp")
        weights.append(pitchwarp_weight)
    if scatter_weight > 0:
        effects_menu.append("scatter")
        weights.append(scatter_weight)
    if gate_weight > 0:
        effects_menu.append("gate")
        weights.append(gate_weight)

    if not effects_menu:
        return in_audio

    weights = np.array(weights, dtype=np.float64)
    weights /= weights.sum()

    # --- slice and glitch ---
    out = mono.copy()
    num_slices = max(1, n // slice_len)

    for i in range(num_slices):
        start = i * slice_len
        end = min(start + slice_len, n)
        chunk = mono[start:end].copy()

        if len(chunk) < 64:
            continue

        # roll the dice
        if rng.random() > chaos:
            continue  # this slice stays clean

        # pick a glitch type
        fx = rng.choice(effects_menu, p=weights)

        if fx == "stutter":
            repeats = rng.choice([2, 3, 4, 6, 8])
            chunk = _stutter(chunk, repeats=repeats)
        elif fx == "tapestop":
            chunk = _tape_stop(chunk, sr)
        elif fx == "reverse":
            chunk = chunk[::-1].copy()
        elif fx == "pitchwarp":
            chunk = _pitch_warp(chunk, sr)
        elif fx == "scatter":
            grain_ms = rng.choice([15.0, 30.0, 50.0, 80.0])
            chunk = _granular_scatter(chunk, sr, grain_ms=grain_ms)
        elif fx == "gate":
            divs = rng.choice([4, 6, 8])
            chunk = _gate_chop(chunk, sr, divisions=divs)

        # crossfade edges to avoid clicks (4ms)
        fade = min(len(chunk) // 4, int(sr * 0.004))
        if fade > 1:
            chunk[:fade] *= np.linspace(0, 1, fade)
            chunk[-fade:] *= np.linspace(1, 0, fade)
            # also fade the dry signal at boundaries
            out[start:start + fade] *= np.linspace(1, 0, fade)
            if end - fade > start:
                out[end - fade:end] *= np.linspace(0, 1, fade)

        out[start:end] = chunk[:end - start]

    # --- wet/dry blend ---
    result = (1.0 - mix) * mono + mix * out

    # --- reconstruct stereo ---
    if in_audio.ndim == 2 and in_audio.shape[1] == 2:
        result = np.column_stack([result, result])
    else:
        result = result[:, None]

    # normalize
    peak = np.max(np.abs(result)) or 1.0
    if peak > 0.99:
        result = result * (0.99 / peak)

    return result.astype(np.float32)
