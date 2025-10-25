# python -m dj_helpers.beatgrid data/mp3s/Dynamite.mp3

# beatgrid.py
from dataclasses import dataclass
from typing import Optional
import numpy as np
import librosa
import sys
import os

@dataclass
class BeatGrid:
    bpm: float
    beats: np.ndarray        # shape (N,) beat times in seconds
    bars: np.ndarray         # shape (M,) bar start times (downbeats) in seconds
    sr: int
    duration: float

def get_beatgrid(
    path: str,
    sr: Optional[int] = None,
    start_bpm: Optional[float] = None,
    tightness: float = 100.0,
) -> BeatGrid:
    """
    Build a beat grid + bar starts (downbeats) assuming 4/4.

    Heuristic for downbeat: choose the phase (0..3) that maximizes summed
    onset strength at those beats across the whole track.
    """
    # 1) load mono
    y, sr = librosa.load(path, mono=True, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)

    # 2) onset envelope (accent strength)
    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # 3) tempo + beats
    beat_track_kwargs = {
        "onset_envelope": onset_env,
        "sr": sr,
        "hop_length": hop_length,
        "tightness": tightness,   # higher = stick closer to constant tempo
        "units": "frames",
    }
    if start_bpm is not None:
        beat_track_kwargs["start_bpm"] = start_bpm
    
    tempo, beat_frames = librosa.beat.beat_track(**beat_track_kwargs)
    beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    # guard-rails
    if beats.size == 0:
        # fallback: no beats found -> fake a grid from tempo estimate
        if tempo <= 0:
            tempo = 120.0
        sec_per_beat = 60.0 / float(tempo)
        beats = np.arange(0.0, duration, sec_per_beat)
    bpm = float(np.asarray(tempo).squeeze())
    bpm = max(60.0, min(200.0, bpm))  # clamp to DJ-ish range

    # 4) pick downbeat phase (0..3) by maximizing accent strength at those beats
    #    (sample onset_env at beat frames and sum every 4 beats with an offset)
    beat_env = onset_env[np.clip(beat_frames, 0, len(onset_env)-1)]
    if beats.size < 4:
        bars = beats.copy()  # too short to estimate phase; treat every beat as a bar
    else:
        scores = []
        for phase in range(4):
            idx = np.arange(phase, len(beat_env), 4)
            scores.append(beat_env[idx].sum())
        best_phase = int(np.argmax(scores))
        bar_frames = beat_frames[best_phase::4]
        bars = librosa.frames_to_time(bar_frames, sr=sr, hop_length=hop_length)

    # optional cleanup: ensure bars are strictly increasing and within duration
    bars = bars[(bars >= 0.0) & (bars <= duration)]
    beats = beats[(beats >= 0.0) & (beats <= duration)]

    return BeatGrid(bpm=bpm, beats=beats, bars=bars, sr=sr, duration=duration)

# -----------------------------------------------------------
# Run from command line
# -----------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m calibrate.beatgrid <audio_path>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        sys.exit(1)

    grid = get_beatgrid(path)
    print(f"\n🎵 Analyzed: {os.path.basename(path)}")
    print(f"BPM: {grid.bpm:.1f}")
    print(f"Duration: {grid.duration:.2f}s")
    print(f"Beats detected: {len(grid.beats)}")
    print(f"Bars detected: {len(grid.bars)}")

    print("\n🕐 First 8 beats (s):", np.round(grid.beats[:8], 2))
    print("🎶 First 4 bars (s):", np.round(grid.bars[:4], 2))
    print()
