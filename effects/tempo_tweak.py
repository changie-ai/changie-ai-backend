# effects/tempo_tweak.py
import numpy as np
import librosa
import pyrubberband as pyrb

def process_tempo_tweak(in_audio, sr, target_bpm=None, aggressive=False):
    """
    Detect audio tempo and time-stretch to target_bpm preserving pitch.
    If target_bpm is None, use nearest standard tempo (e.g., 120) or leave unchanged.
    aggressive toggles whether we apply stronger tempo correction.
    """
    # convert stereo -> mono for tempo detection (mixdown)
    mono = np.mean(in_audio, axis=1)
    # librosa tempo detection
    try:
        tempo = float(librosa.beat.tempo(y=mono, sr=sr)[0])
    except Exception:
        # fallback
        tempo = 120.0
    if target_bpm is None:
        # choose nearest conventional tempo (optional) OR do small fix to 120
        # We will aim to slightly correct to nearest 4/4 multiple (optional)
        target_bpm = tempo  # if user didn't ask, keep tempo (no change)
    if target_bpm == 0 or abs(target_bpm - tempo) < 0.5:
        return in_audio  # nothing to do
    rate = float(target_bpm) / float(tempo)
    # pyrubberband.time_stretch expects 1D or 2D? We'll process channels independently
    chans = []
    for ch in range(in_audio.shape[1]):
        y = in_audio[:, ch]
        try:
            stretched = pyrb.time_stretch(y, sr, rate)
        except Exception:
            # fallback to librosa.phase_vocoder (librosa expects STFT)
            S = librosa.stft(y)
            S2 = librosa.feature.phase_vocoder(D=S, rate=rate)
            stretched = librosa.istft(S2, length=int(len(y)/rate))
        chans.append(stretched)
    # Resample channels to same length (pyrubberband may produce same length)
    minlen = min(map(len, chans))
    stacked = np.stack([c[:minlen] for c in chans], axis=1)
    # ensure stereo shape
    return stacked.astype(np.float32)
