# effects/hype_delay.py
import numpy as np

def process_hype_delay(in_audio, sr,
                       mix=0.6, feedback=0.7, delay_ms=500.0, taper=0.7,
                       repeats=8, feedback_smoothing=0.98):
    """
    Stereo delay with multiple repeats and taper/feedback control.
    Params:
      mix: 0..1 wet/dry
      feedback: 0..0.98 per repeat
      delay_ms: base ms for delay
      taper: per-repeat taper multiplier (0..1) lower = faster fade
      repeats: max repeats to generate
    """
    audio = in_audio.copy()
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    n = audio.shape[0]
    delay_samples = max(1, int(sr * (delay_ms / 1000.0)))
    max_len = n + delay_samples * repeats + 10
    buf_l = np.zeros(max_len, dtype=np.float32)
    buf_r = np.zeros(max_len, dtype=np.float32)
    buf_l[:n] += audio[:,0]
    buf_r[:n] += audio[:,1]

    for rep in range(1, repeats+1):
        start = rep * delay_samples
        gain = (feedback ** rep) * (taper ** rep)
        if start >= max_len:
            break
        end = min(start + n, max_len)
        seg_len = end - start
        buf_l[start:end] += audio[:seg_len,0] * gain
        buf_r[start:end] += audio[:seg_len,1] * gain

    wet = np.column_stack([buf_l[:n], buf_r[:n]])
    mixed = (1.0 - mix) * audio + mix * wet
    # mild normalization
    peak = np.max(np.abs(mixed)) or 1.0
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)
    return mixed.astype(np.float32)
