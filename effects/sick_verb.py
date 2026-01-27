# effects/sick_verb.py
import numpy as np

def _allpass(x, delay, feedback):
    # naive allpass via buffer
    n = len(x)
    buf = np.zeros(n + delay + 10, dtype=np.float32)
    out = np.zeros_like(x)
    buf[:n] = x
    for i in range(n):
        j = i - delay
        d = buf[j] if j >= 0 else 0.0
        out[i] = -feedback * buf[i] + d + feedback * (d if j>=0 else 0.0)
        buf[i] = x[i] + feedback * (buf[i - delay] if i - delay >= 0 else 0.0)
    return out

def process_sick_verb(in_audio, sr,
                      mix=0.5, predelay_ms=10.0, decay=2.0, room_size=0.5, diffusion=0.5):
    """
    Lightweight reverb: predelay + cascaded allpasses + feedback to simulate reverb tail.
    Params:
      mix: 0..1
      predelay_ms: in ms
      decay: seconds approximate
      room_size: 0..1 (affects reverb density)
      diffusion: 0..1 (affects repeated allpass count)
    """
    audio = in_audio.copy().astype(np.float32)
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    n = audio.shape[0]
    predelay = int(sr * (predelay_ms / 1000.0))
    # Create early reflections via short delays
    early = np.zeros_like(audio)
    delays = [int(sr * d) for d in (0.003, 0.007, 0.013)]
    for ch in range(2):
        for d in delays:
            if d < n:
                early[d:, ch] += audio[:n - d, ch] * (0.3 * room_size)
    # Create late reverb by feedback / filtered noise
    tail_len = int(sr * decay)
    tail = np.zeros((n, 2), dtype=np.float32)
    # simple feedback network
    comb_delays = [int(sr * dd) for dd in (0.029, 0.037, 0.043)]
    for ch in range(2):
        buf = np.zeros(n + max(comb_delays) + 10, dtype=np.float32)
        buf[:n] = audio[:, ch]
        for i in range(n):
            val = 0.0
            for d in comb_delays:
                j = i - d
                val += (buf[j] if j >= 0 else 0.0) * (0.3 * room_size)
            # feedback decays gradually
            tail[i, ch] = val * (0.9 ** (i / (sr*decay + 1)))
            buf[i] += tail[i, ch] * diffusion
    wet = early + tail
    out = (1.0 - mix) * audio + mix * wet
    # normalize
    peak = np.max(np.abs(out)) or 1.0
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32)
