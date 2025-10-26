# crossfade.py
# DJ-style crossfading with bass and voice transition control
#
# USAGE EXAMPLES:
#
# Basic crossfade with defaults (16 beat fade, auto-detect start):
#   python -m dj_helpers.crossfade data/mp3s/Song1.mp3 data/mp3s/Song2.mp3
#
# Custom fade: start at bar 64, fade over 32 beats:
#   python -m dj_helpers.crossfade Song1.mp3 Song2.mp3 --fade-start-bar 64 --fade-beats 32
#
# Start track 2 from its 8th bar (skip intro):
#   python -m dj_helpers.crossfade Song1.mp3 Song2.mp3 --track2-start-bar 8
#
# Specify exact fade start time in seconds:
#   python -m dj_helpers.crossfade Song1.mp3 Song2.mp3 --fade-start-time 125.5
#
# Delay bass swap until 70% through the fade:
#   python -m dj_helpers.crossfade Song1.mp3 Song2.mp3 --bass-transition 0.7
#
# Enable voice transition at 60% through fade:
#   python -m dj_helpers.crossfade Song1.mp3 Song2.mp3 --voice-transition 0.6
#
# Restore to original tempo after crossfade (EXPERIMENTAL):
#   python -m dj_helpers.crossfade Song1.mp3 Song2.mp3 --restore-tempo --restore-after-bars 8
#
# Full advanced transition (PowerShell - use backticks):
#   python -m dj_helpers.crossfade Song1.mp3 Song2.mp3 `
#     --fade-start-time 180.0 `
#     --fade-beats 32 `
#     --bass-transition 0.5 `
#     --voice-transition 0.7 `
#     -o mixes/perfect_transition.wav
#
# Full advanced transition (Bash/Linux - use backslashes):
#   python -m dj_helpers.crossfade Song1.mp3 Song2.mp3 \
#     --fade-start-time 180.0 \
#     --fade-beats 32 \
#     --bass-transition 0.5 \
#     --voice-transition 0.7 \
#     -o mixes/perfect_transition.wav

import json
import os
import sys
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from typing import Optional, Tuple
import argparse


class DJCrossfader:
    """
    Handles DJ-style crossfading between two tracks with bass and voice transition control.
    
    Features:
    - Beat-synced crossfading
    - Bass frequency separation and transition
    - Voice/vocal separation and transition (optional)
    - Bar-aligned mixing points
    - Tempo matching via time-stretching
    - Return to original tempo after crossfade (optional, experimental)
    """
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
        # Bass frequency cutoff (typical DJ mixer: ~100-250 Hz)
        self.bass_cutoff = 150  # Hz
        # Voice frequency range (typical vocals: ~300-3000 Hz)
        self.voice_low = 300   # Hz
        self.voice_high = 3000  # Hz
        
    def load_barmap(self, audio_path: str, barmap_dir: str = "data/metadata") -> dict:
        """Load the barmap JSON for a given audio file."""
        song_name = os.path.splitext(os.path.basename(audio_path))[0]
        
        # Try both _barmap.json and .json naming conventions
        barmap_path = os.path.join(barmap_dir, f"{song_name}_barmap.json")
        if not os.path.exists(barmap_path):
            barmap_path = os.path.join(barmap_dir, f"{song_name}.json")
        
        if not os.path.exists(barmap_path):
            raise FileNotFoundError(
                f"Barmap not found for {song_name}\n"
                f"Tried: {barmap_dir}/{song_name}_barmap.json and {song_name}.json\n"
                f"Run: python -m calibrate.barmap {audio_path}"
            )
        
        with open(barmap_path, "r") as f:
            return json.load(f)
    
    def separate_bass(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separate audio into bass and non-bass (mids/highs) components.
        Returns: (bass, mids_highs)
        """
        # Design a lowpass filter for bass
        nyquist = self.sr / 2
        cutoff_norm = self.bass_cutoff / nyquist
        
        # Butterworth filter (steep rolloff, good for DJ mixing)
        sos_low = signal.butter(4, cutoff_norm, btype='low', output='sos')
        sos_high = signal.butter(4, cutoff_norm, btype='high', output='sos')
        
        # Apply filters
        bass = signal.sosfilt(sos_low, audio)
        mids_highs = signal.sosfilt(sos_high, audio)
        
        return bass, mids_highs
    
    def separate_voice(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separate audio into voice and non-voice components using bandpass filtering.
        Returns: (voice, non_voice)
        
        Note: This is a simple frequency-based approach. For better vocal separation,
        consider using more advanced models like Spleeter or Demucs.
        """
        nyquist = self.sr / 2
        low_norm = self.voice_low / nyquist
        high_norm = self.voice_high / nyquist
        
        # Bandpass filter for voice range
        sos_voice = signal.butter(4, [low_norm, high_norm], btype='band', output='sos')
        # Bandstop filter for everything else
        sos_non_voice = signal.butter(4, [low_norm, high_norm], btype='bandstop', output='sos')
        
        voice = signal.sosfilt(sos_voice, audio)
        non_voice = signal.sosfilt(sos_non_voice, audio)
        
        return voice, non_voice
    
    def get_bar_time(self, barmap: dict, bar_index: int) -> float:
        """Get the time (in seconds) of a specific bar."""
        bars = barmap["bars"]
        if bar_index < 0:
            bar_index = len(bars) + bar_index  # Handle negative indexing
        if bar_index >= len(bars):
            return barmap["duration"]
        return bars[bar_index]
    
    def beat_to_seconds(self, barmap: dict, num_beats: int) -> float:
        """Convert number of beats to seconds based on BPM."""
        bpm = barmap["bpm"]
        return (num_beats / bpm) * 60.0
    
    def create_fade_curve(self, num_samples: int, curve_type: str = "equal_power") -> np.ndarray:
        """
        Create a fade curve.
        
        Types:
        - 'linear': Simple linear fade
        - 'equal_power': Equal power crossfade (sounds more natural)
        - 'smooth': S-curve (smooth acceleration)
        """
        x = np.linspace(0, 1, num_samples)
        
        if curve_type == "linear":
            return x
        elif curve_type == "equal_power":
            # Equal power: sum of squares = constant
            return np.sqrt(x)
        elif curve_type == "smooth":
            # Smooth S-curve (smoothstep)
            return x * x * (3 - 2 * x)
        else:
            return x
    
    def crossfade(
        self,
        track1_path: str,
        track2_path: str,
        fade_start_bar: Optional[int] = None,  # Bar in track1 to start fade
        fade_start_time: Optional[float] = None,  # Or specify exact time in seconds
        fade_beats: int = 16,                   # Duration of crossfade in beats
        track2_start_bar: int = 0,              # Which bar of track2 to mix in
        fade_curve: str = "equal_power",
        bass_transition_ratio: float = 0.5,     # 0-1: how much to delay bass transition
        voice_transition_ratio: Optional[float] = None,  # 0-1: when to swap vocals (None = no voice handling)
        restore_tempo: bool = False,            # Return track2 to original tempo after fade
        restore_after_bars: int = 8,            # How many bars to wait before restoring tempo
        barmap_dir: str = "data/metadata",
    ) -> Tuple[np.ndarray, int]:
        """
        Perform DJ-style crossfade between two tracks.
        
        Args:
            track1_path: Path to first audio file
            track2_path: Path to second audio file
            fade_start_bar: Bar in track1 to start fade (None = auto-detect near end)
            fade_start_time: Exact time in seconds to start fade (overrides fade_start_bar)
            fade_beats: Number of beats to crossfade over
            track2_start_bar: Which bar of track2 to start from (0 = beginning)
            fade_curve: Type of fade curve ('linear', 'equal_power', 'smooth')
            bass_transition_ratio: 0-1, how far through fade to swap bass (0.5 = halfway)
            voice_transition_ratio: 0-1, when to swap vocals (None = no voice handling)
            restore_tempo: If True, gradually restore track2 to original BPM after crossfade
            restore_after_bars: Number of bars to wait before starting tempo restoration
            barmap_dir: Directory containing barmap JSON files
            
        Returns:
            (mixed_audio, sample_rate)
        """
        print("🎵 DJ Crossfader starting...")
        
        # Load barmaps
        barmap1 = self.load_barmap(track1_path, barmap_dir)
        barmap2 = self.load_barmap(track2_path, barmap_dir)
        
        print(f"Track 1: {os.path.basename(track1_path)} - {barmap1['bpm']:.1f} BPM, {len(barmap1['bars'])} bars")
        print(f"Track 2: {os.path.basename(track2_path)} - {barmap2['bpm']:.1f} BPM, {len(barmap2['bars'])} bars")
        
        # Load audio
        y1, sr1 = librosa.load(track1_path, sr=self.sr, mono=True)
        y2_original, sr2 = librosa.load(track2_path, sr=self.sr, mono=True)
        
        # Determine fade start point
        if fade_start_time is not None:
            # Use exact time if provided
            fade_start_time_final = fade_start_time
            # Find closest bar for display
            bars = barmap1["bars"]
            fade_start_bar = min(range(len(bars)), key=lambda i: abs(bars[i] - fade_start_time))
            print(f"📍 Using exact fade start time: {fade_start_time:.2f}s (near bar {fade_start_bar})")
        elif fade_start_bar is not None:
            # Use specified bar
            fade_start_time_final = self.get_bar_time(barmap1, fade_start_bar)
        else:
            # Auto-detect: start 16 bars before end
            fade_start_bar = len(barmap1["bars"]) - 16
            fade_start_bar = max(0, fade_start_bar)
            fade_start_time_final = self.get_bar_time(barmap1, fade_start_bar)
        
        # Calculate timing
        fade_duration = self.beat_to_seconds(barmap1, fade_beats)
        track2_start_time = self.get_bar_time(barmap2, track2_start_bar)
        
        print(f"\n⏱️  Crossfade timing:")
        print(f"   Track 1 fade starts at: {fade_start_time_final:.2f}s (bar {fade_start_bar})")
        print(f"   Fade duration: {fade_duration:.2f}s ({fade_beats} beats)")
        print(f"   Track 2 starts from: {track2_start_time:.2f}s (bar {track2_start_bar})")
        
        # Convert times to samples
        fade_start_sample = int(fade_start_time_final * self.sr)
        fade_samples = int(fade_duration * self.sr)
        track2_start_sample = int(track2_start_time * self.sr)
        
        # Tempo matching: stretch track2 to match track1's BPM
        bpm_ratio = barmap1["bpm"] / barmap2["bpm"]
        if abs(bpm_ratio - 1.0) > 0.01:  # Only stretch if significantly different
            print(f"🎛️  Tempo matching: stretching track 2 by {bpm_ratio:.3f}x")
            y2_stretched = librosa.effects.time_stretch(y2_original, rate=bpm_ratio)
        else:
            y2_stretched = y2_original
        
        # Trim track2 to start from desired bar
        y2_trimmed = y2_stretched[track2_start_sample:]
        
        # Separate frequencies
        print("🔊 Separating frequencies...")
        bass1, mids_highs1 = self.separate_bass(y1)
        bass2, mids_highs2 = self.separate_bass(y2_trimmed)
        
        # Voice separation if requested
        if voice_transition_ratio is not None:
            print("🎤 Separating vocals...")
            voice1, non_voice1 = self.separate_voice(mids_highs1)
            voice2, non_voice2 = self.separate_voice(mids_highs2)
        
        # Create crossfade curves
        fade_out = 1 - self.create_fade_curve(fade_samples, fade_curve)
        fade_in = self.create_fade_curve(fade_samples, fade_curve)
        
        # Calculate transition points
        bass_transition_sample = int(fade_samples * bass_transition_ratio)
        if voice_transition_ratio is not None:
            voice_transition_sample = int(fade_samples * voice_transition_ratio)
        
        # Build the mix
        # Part 1: Before crossfade (just track 1)
        mix_before = y1[:fade_start_sample]
        
        # Part 2: During crossfade
        # Check if we need to pad either track
        track1_remaining = len(y1) - fade_start_sample
        track2_available = len(y2_trimmed)
        
        if track1_remaining < fade_samples:
            # Track 1 ends before fade completes - pad with silence
            print(f"⚠️  Track 1 ends before fade completes, padding with {(fade_samples - track1_remaining) / self.sr:.2f}s silence")
            y1_padded = np.concatenate([y1, np.zeros(fade_samples - track1_remaining)])
            # Re-separate bass/mids from padded track
            bass1_padded, mids_highs1_padded = self.separate_bass(y1_padded)
            if voice_transition_ratio is not None:
                voice1_padded, non_voice1_padded = self.separate_voice(mids_highs1_padded)
        else:
            y1_padded = y1
            bass1_padded = bass1
            mids_highs1_padded = mids_highs1
            if voice_transition_ratio is not None:
                voice1_padded = voice1
                non_voice1_padded = non_voice1
        
        if track2_available < fade_samples:
            # Track 2 is too short - pad with silence
            print(f"⚠️  Track 2 is shorter than fade duration, padding with {(fade_samples - track2_available) / self.sr:.2f}s silence")
            y2_padded = np.concatenate([y2_trimmed, np.zeros(fade_samples - track2_available)])
            # Re-separate bass/mids from padded track
            bass2_padded, mids_highs2_padded = self.separate_bass(y2_padded)
            if voice_transition_ratio is not None:
                voice2_padded, non_voice2_padded = self.separate_voice(mids_highs2_padded)
        else:
            y2_padded = y2_trimmed
            bass2_padded = bass2
            mids_highs2_padded = mids_highs2
            if voice_transition_ratio is not None:
                voice2_padded = voice2
                non_voice2_padded = non_voice2
        
        # Now extract the full crossfade segments
        crossfade_len = fade_samples
        
        # Get segments for crossfade
        t1_bass = bass1_padded[fade_start_sample:fade_start_sample + crossfade_len]
        t2_bass = bass2_padded[:crossfade_len]
        
        if voice_transition_ratio is not None:
            # With voice handling
            t1_voice = voice1_padded[fade_start_sample:fade_start_sample + crossfade_len]
            t1_non_voice = non_voice1_padded[fade_start_sample:fade_start_sample + crossfade_len]
            t2_voice = voice2_padded[:crossfade_len]
            t2_non_voice = non_voice2_padded[:crossfade_len]
            
            # Crossfade non-voice mids with main curve
            non_voice_mixed = t1_non_voice * fade_out + t2_non_voice * fade_in
            
            # Voice transition: sharp swap at the transition point
            voice_fade = np.ones(crossfade_len)
            voice_fade[voice_transition_sample:] = 0
            voice_mixed = t1_voice * voice_fade + t2_voice * (1 - voice_fade)
            
            # Combine
            mids_mixed = non_voice_mixed + voice_mixed
        else:
            # Without voice handling (original behavior)
            t1_mids = mids_highs1_padded[fade_start_sample:fade_start_sample + crossfade_len]
            t2_mids = mids_highs2_padded[:crossfade_len]
            mids_mixed = t1_mids * fade_out + t2_mids * fade_in
        
        # Bass transition: sharper swap at the transition point
        bass_fade = np.ones(crossfade_len)
        bass_fade[bass_transition_sample:] = 0
        bass_mixed = t1_bass * bass_fade + t2_bass * (1 - bass_fade)
        
        # Combine bass and mids
        mix_crossfade = bass_mixed + mids_mixed
        
        # Part 3: After crossfade
        remaining_track2 = y2_padded[crossfade_len:] if track2_available < fade_samples else y2_trimmed[crossfade_len:]
        
        if restore_tempo and abs(bpm_ratio - 1.0) > 0.01 and len(remaining_track2) > 0:
            # SIMPLIFIED tempo restoration to avoid overlapping bugs
            print(f"🔄 Restoring tempo after {restore_after_bars} bars...")
            
            # Calculate when to start tempo restoration
            restore_start_samples = int(self.beat_to_seconds(barmap1, restore_after_bars * 4) * self.sr)
            
            if len(remaining_track2) > restore_start_samples:
                # Keep stretched version for a bit
                immediate_after = remaining_track2[:restore_start_samples]
                
                # Then switch to original tempo for the rest
                # Calculate where we are in the original track
                elapsed_in_stretched = (crossfade_len + restore_start_samples) / bpm_ratio
                original_position = int(track2_start_sample + elapsed_in_stretched)
                
                if original_position < len(y2_original):
                    # Continue with original tempo from this point
                    mix_after = np.concatenate([immediate_after, y2_original[original_position:]])
                else:
                    # Just use what we have
                    mix_after = remaining_track2
            else:
                mix_after = remaining_track2
        else:
            mix_after = remaining_track2
        
        # Concatenate all parts
        mix_final = np.concatenate([mix_before, mix_crossfade, mix_after])
        
        # Normalize to prevent clipping
        max_val = np.abs(mix_final).max()
        if max_val > 0:
            mix_final = mix_final * (0.95 / max_val)
        
        print(f"\n✅ Crossfade complete!")
        print(f"   Final mix duration: {len(mix_final) / self.sr:.2f}s")
        print(f"   Bass transition at: {(fade_start_time_final + bass_transition_sample/self.sr):.2f}s")
        if voice_transition_ratio is not None:
            print(f"   Voice transition at: {(fade_start_time_final + voice_transition_sample/self.sr):.2f}s")
        
        return mix_final, self.sr
    
    def save_mix(self, audio: np.ndarray, sr: int, output_path: str):
        """Save the mixed audio to a file."""
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        sf.write(output_path, audio, sr)
        print(f"💾 Saved mix to: {output_path}")


# -------------------------------------------------------------
# Command line interface
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="DJ-style crossfade between two tracks with bass and voice transition control",
        epilog="Example: python -m dj_helpers.crossfade data/mp3s/Dynamite.mp3 data/mp3s/Summer.mp3 --fade-beats 32 --voice-transition 0.6"
    )
    parser.add_argument("track1", help="Path to first audio file")
    parser.add_argument("track2", help="Path to second audio file")
    parser.add_argument(
        "--fade-start-bar",
        type=int,
        default=None,
        help="Bar in track1 to start fade (default: auto-detect 16 bars from end)"
    )
    parser.add_argument(
        "--fade-start-time",
        type=float,
        default=None,
        help="Exact time in seconds to start fade (overrides --fade-start-bar)"
    )
    parser.add_argument(
        "--fade-beats",
        type=int,
        default=16,
        help="Number of beats to crossfade over (default: 16)"
    )
    parser.add_argument(
        "--track2-start-bar",
        type=int,
        default=0,
        help="Which bar of track2 to start from (default: 0)"
    )
    parser.add_argument(
        "--fade-curve",
        choices=["linear", "equal_power", "smooth"],
        default="equal_power",
        help="Type of fade curve (default: equal_power)"
    )
    parser.add_argument(
        "--bass-transition",
        type=float,
        default=0.5,
        help="Bass transition point (0-1, default: 0.5 = halfway through fade)"
    )
    parser.add_argument(
        "--voice-transition",
        type=float,
        default=None,
        help="Voice transition point (0-1, enables voice handling, default: None = disabled)"
    )
    parser.add_argument(
        "--bass-cutoff",
        type=int,
        default=150,
        help="Bass frequency cutoff in Hz (default: 150)"
    )
    parser.add_argument(
        "--restore-tempo",
        action="store_true",
        help="Gradually restore track2 to its original tempo after crossfade (EXPERIMENTAL)"
    )
    parser.add_argument(
        "--restore-after-bars",
        type=int,
        default=8,
        help="How many bars to wait before restoring tempo (default: 8)"
    )
    parser.add_argument(
        "--barmap-dir",
        default="data/metadata",
        help="Directory containing barmap JSON files (default: data/metadata)"
    )
    parser.add_argument(
        "--out-path",
        "-o",
        default=None,
        help="Output file path (default: auto-generated as Song1_Song2.wav)"
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=44100,
        help="Sample rate (default: 44100)"
    )
    
    args = parser.parse_args()
    
    # Auto-generate output filename if not provided
    if args.out_path is None:
        song1_name = os.path.splitext(os.path.basename(args.track1))[0]
        song2_name = os.path.splitext(os.path.basename(args.track2))[0]
        args.out_path = f"data/mixes/{song1_name}_{song2_name}.wav"
        print(f"💾 Auto-generated output name: {args.out_path}")
    
    # Create crossfader
    crossfader = DJCrossfader(sr=args.sr)
    crossfader.bass_cutoff = args.bass_cutoff
    
    # Perform crossfade
    try:
        mixed_audio, sr = crossfader.crossfade(
            track1_path=args.track1,
            track2_path=args.track2,
            fade_start_bar=args.fade_start_bar,
            fade_start_time=args.fade_start_time,
            fade_beats=args.fade_beats,
            track2_start_bar=args.track2_start_bar,
            fade_curve=args.fade_curve,
            bass_transition_ratio=args.bass_transition,
            voice_transition_ratio=args.voice_transition,
            restore_tempo=args.restore_tempo,
            restore_after_bars=args.restore_after_bars,
            barmap_dir=args.barmap_dir,
        )
        
        # Save the mix
        crossfader.save_mix(mixed_audio, sr, args.out_path)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
