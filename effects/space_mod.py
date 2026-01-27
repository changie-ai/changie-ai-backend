# effects/space_mod.py
import numpy as np

def process_space_mod(in_audio, sr,
                      rate=1.0, depth=0.4, feedback=0.2, mix=0.5, manual=0.0, weird=False):
    """
    Stereo modulated delay (spacey chorus/flanger style).
    rate: Hz of LFO
    depth: intensity (0..1)
    feedback: feedback coefficient (-1..1)
    mix: wet/dry 0..1
    manual: static offset in ms [-50..50]
    weird: toggles more complex LFO
    """
    audio = in_audio.copy().astype(np.float32)
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    n = audio.shape[0]
    t = np.arange(n) / sr
    if weird:
        lfo = 0.5 * (np.sin(2*np.pi*rate*t) + 0.5*np.sin(2*np.pi*(rate*3.7)*t))
    else:
        lfo = np.sin(2*np.pi*rate*t)
    base_delay_ms = 10.0 + (depth * 40.0) + manual
    max_delay = int(sr * (base_delay_ms/1000.0 + 0.05)) + 4
    out = np.zeros_like(audio, dtype=np.float32)
    buf_l = np.zeros(n + max_delay + 10, dtype=np.float32)
    buf_r = np.zeros(n + max_delay + 10, dtype=np.float32)
    buf_l[:n] = audio[:,0]
    buf_r[:n] = audio[:,1]
    for i in range(n):
        this_delay_s = (base_delay_ms/1000.0) + (lfo[i] * depth * 0.005)
        d_samps = int(np.clip(this_delay_s * sr, 1, max_delay-1))
        read_idx = i - d_samps
        if read_idx < 0:
            delayed_l = 0.0
            delayed_r = 0.0
        else:
            delayed_l = buf_l[read_idx]
            delayed_r = buf_r[read_idx]
        wet_l = delayed_l + feedback * (out[i-1,0] if i>0 else 0.0)
        wet_r = delayed_r + feedback * (out[i-1,1] if i>0 else 0.0)
        out[i,0] = (1.0 - mix) * audio[i,0] + mix * wet_l
        out[i,1] = (1.0 - mix) * audio[i,1] + mix * wet_r
    # normalize
    peak = np.max(np.abs(out)) or 1.0
    if peak > 0.98:
        out = out * (0.98 / peak)
    return out.astype(np.float32)
