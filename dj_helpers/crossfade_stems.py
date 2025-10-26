# crossfade_stems.py
# DJ-style crossfading using actual Demucs stems with independent transition timing
#
# USAGE EXAMPLES:
#
# Basic crossfade with stem-specific transitions:
#   python -m dj_helpers.crossfade_stems data/mp3s/Song1.mp3 data/mp3s/Song2.mp3
#
# Custom transition points for each stem:
#   python -m dj_helpers.crossfade_stems Song1.mp3 Song2.mp3 `
#     --bass-transition 0.5 `
#     --drums-transition 0.4 `
#     --vocals-transition 0.7 `
#     --other-transition 0.6 `
#     --fade-beats 32
#
# The "other" stem includes all backing instruments (guitars, synths, piano, etc.)
# By transitioning it separately from vocals, you get smoother blends!

import json
import os
import sys
import numpy as np
import librosa
import soundfile as sf
from typing import Optional, Tuple, Dict
import argparse


class DJStemCrossfader:
    """
    DJ-style crossfading using actual Demucs stem separations.
    
    Stems:
    - vocals: Lead and backing vocals
    - drums: All percussion
    - bass: Bass guitar, synth bass, sub bass
    - other: Everything else (guitars, piano, synths, strings, etc.)
    
    Each stem can transition independently for maximum control!
    """
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
        
    def load_metadata(self, audio_path: str, metadata_dir: str = "data/metadata") -> dict:
        """Load the metadata JSON that contains barmap and stem paths."""
        song_name = os.path.splitext(os.path.basename(audio_path))[0]
        
        # Try to load barmap
        barmap_data = None
        barmap_files = [
            f"{song_name}_barmap.json",
            f"{song_name}.json",
        ]
        
        for filename in barmap_files:
            barmap_path = os.path.join(metadata_dir, filename)
            if os.path.exists(barmap_path):
                with open(barmap_path, "r") as f:
                    data = json.load(f)
                    if "bars" in data and "bpm" in data and "duration" in data:
                        barmap_data = data
                        break
        
        if barmap_data is None:
            raise FileNotFoundError(
                f"Barmap not found for {song_name}\n"
                f"Run: python -m calibrate.barmap {audio_path}"
            )
        
        # Try to load stem paths (might be in the same file or separate)
        if "stems" not in barmap_data:
            # Try separate metadata file
            metadata_files = [
                f"{song_name}.json",
                f"{song_name}_metadata.json",
            ]
            
            for filename in metadata_files:
                metadata_path = os.path.join(metadata_dir, filename)
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        stem_data = json.load(f)
                        if "stems" in stem_data:
                            barmap_data["stems"] = stem_data["stems"]
                            break
        
        if "stems" not in barmap_data:
            raise FileNotFoundError(
                f"Stem paths not found for {song_name}\n"
                f"Make sure your metadata JSON has a 'stems' field with paths to:\n"
                f"  - vocals.wav\n"
                f"  - drums.wav\n"
                f"  - bass.wav\n"
                f"  - other.wav\n"
                f"Or run: python -m calibrate.split_audio {audio_path}"
            )
        
        return barmap_data
    
    def load_stems(self, metadata: dict) -> Dict[str, np.ndarray]:
        """
        Load all stems from paths in metadata.
        Returns dict with keys: vocals, drums, bass, other
        """
        if "stems" not in metadata:
            raise ValueError("No stem paths found in metadata")
        
        stem_paths = metadata["stems"]
        
        print(f"📂 Loading stems...")
        
        stems = {}
        for stem_name in ["vocals", "drums", "bass", "other"]:
            if stem_name not in stem_paths:
                raise FileNotFoundError(f"Missing {stem_name} stem path in metadata")
            
            stem_path = stem_paths[stem_name]
            
            # Handle both absolute and relative paths
            if not os.path.exists(stem_path):
                # Try relative to current directory
                stem_path = os.path.join(os.getcwd(), stem_path)
            
            if not os.path.exists(stem_path):
                raise FileNotFoundError(f"Stem file not found: {stem_paths[stem_name]}")
            
            # Load audio (works with both .wav and .mp3)
            audio, _ = librosa.load(stem_path, sr=self.sr, mono=True)
            stems[stem_name] = audio
            print(f"   ✓ {stem_name}: {len(audio) / self.sr:.2f}s ({os.path.basename(stem_path)})")
        
        return stems
    
    def get_bar_time(self, metadata: dict, bar_index: int) -> float:
        """Get the time (in seconds) of a specific bar."""
        bars = metadata["bars"]
        if bar_index < 0:
            bar_index = len(bars) + bar_index
        if bar_index >= len(bars):
            return metadata["duration"]
        return bars[bar_index]
    
    def beat_to_seconds(self, metadata: dict, num_beats: int) -> float:
        """Convert number of beats to seconds based on BPM."""
        bpm = metadata["bpm"]
        return (num_beats / bpm) * 60.0
    
    def create_fade_curve(self, num_samples: int, curve_type: str = "equal_power") -> np.ndarray:
        """Create a fade curve."""
        x = np.linspace(0, 1, num_samples)
        
        if curve_type == "linear":
            return x
        elif curve_type == "equal_power":
            return np.sqrt(x)
        elif curve_type == "smooth":
            return x * x * (3 - 2 * x)
        else:
            return x
    
    def crossfade_stems(
        self,
        track1_path: str,
        track2_path: str,
        fade_start_bar: Optional[int] = None,
        fade_start_time: Optional[float] = None,
        fade_beats: int = 16,
        track2_start_bar: int = 0,
        fade_curve: str = "equal_power",
        bass_transition_ratio: float = 0.5,
        drums_transition_ratio: float = 0.4,
        vocals_transition_ratio: float = 0.7,
        other_transition_ratio: float = 0.6,
        vocals_blend_mode: str = "crossfade",  # "cut" or "crossfade"
        other_blend_mode: str = "crossfade",   # "cut" or "crossfade"
        metadata_dir: str = "data/metadata",
    ) -> Tuple[np.ndarray, int]:
        """
        Perform DJ-style crossfade using stems with independent transition timing.
        
        Args:
            track1_path: Path to first audio file
            track2_path: Path to second audio file
            fade_start_bar: Bar in track1 to start fade (None = auto-detect)
            fade_start_time: Exact time in seconds to start fade
            fade_beats: Number of beats to crossfade over
            track2_start_bar: Which bar of track2 to start from
            fade_curve: Type of fade curve
            bass_transition_ratio: When to swap bass (0-1)
            drums_transition_ratio: When to swap drums (0-1)
            vocals_transition_ratio: When to swap vocals (0-1)
            other_transition_ratio: When to swap other/backing (0-1)
            vocals_blend_mode: "cut" for sharp swap, "crossfade" for gradual blend
            other_blend_mode: "cut" for sharp swap, "crossfade" for gradual blend
            metadata_dir: Directory containing metadata JSON files
        """
        print("🎵 DJ Stem Crossfader starting...")
        
        # Load metadata (includes barmap and stem paths)
        metadata1 = self.load_metadata(track1_path, metadata_dir)
        metadata2 = self.load_metadata(track2_path, metadata_dir)
        
        print(f"\nTrack 1: {os.path.basename(track1_path)} - {metadata1['bpm']:.1f} BPM")
        print(f"Track 2: {os.path.basename(track2_path)} - {metadata2['bpm']:.1f} BPM")
        
        # Load stems
        stems1 = self.load_stems(metadata1)
        stems2 = self.load_stems(metadata2)
        
        # Determine fade start point
        if fade_start_time is not None:
            fade_start_time_final = fade_start_time
            bars = metadata1["bars"]
            fade_start_bar = min(range(len(bars)), key=lambda i: abs(bars[i] - fade_start_time))
        elif fade_start_bar is not None:
            fade_start_time_final = self.get_bar_time(metadata1, fade_start_bar)
        else:
            fade_start_bar = len(metadata1["bars"]) - 16
            fade_start_bar = max(0, fade_start_bar)
            fade_start_time_final = self.get_bar_time(metadata1, fade_start_bar)
        
        # Calculate timing
        fade_duration = self.beat_to_seconds(metadata1, fade_beats)
        track2_start_time = self.get_bar_time(metadata2, track2_start_bar)
        
        print(f"\n⏱️  Crossfade timing:")
        print(f"   Start: {fade_start_time_final:.2f}s (bar {fade_start_bar})")
        print(f"   Duration: {fade_duration:.2f}s ({fade_beats} beats)")
        print(f"   Track 2 from: {track2_start_time:.2f}s")
        
        # Convert to samples
        fade_start_sample = int(fade_start_time_final * self.sr)
        fade_samples = int(fade_duration * self.sr)
        track2_start_sample = int(track2_start_time * self.sr)
        
        # Tempo matching for track 2
        bpm_ratio = metadata1["bpm"] / metadata2["bpm"]
        if abs(bpm_ratio - 1.0) > 0.01:
            print(f"\n🎛️  Tempo matching: stretching track 2 by {bpm_ratio:.3f}x")
            stems2_stretched = {}
            for stem_name, audio in stems2.items():
                stems2_stretched[stem_name] = librosa.effects.time_stretch(audio, rate=bpm_ratio)
            stems2 = stems2_stretched
        
        # Trim track2 stems to start from desired bar
        stems2_trimmed = {
            name: audio[track2_start_sample:] 
            for name, audio in stems2.items()
        }
        
        # Create fade curves
        fade_out = 1 - self.create_fade_curve(fade_samples, fade_curve)
        fade_in = self.create_fade_curve(fade_samples, fade_curve)
        
        # Calculate transition points for each stem
        transition_samples = {
            "bass": int(fade_samples * bass_transition_ratio),
            "drums": int(fade_samples * drums_transition_ratio),
            "vocals": int(fade_samples * vocals_transition_ratio),
            "other": int(fade_samples * other_transition_ratio),
        }
        
        print(f"\n🎚️  Stem transition points:")
        print(f"   Drums:  {drums_transition_ratio*100:.0f}% ({transition_samples['drums']/self.sr:.2f}s)")
        print(f"   Bass:   {bass_transition_ratio*100:.0f}% ({transition_samples['bass']/self.sr:.2f}s)")
        print(f"   Other:  {other_transition_ratio*100:.0f}% ({transition_samples['other']/self.sr:.2f}s)")
        print(f"   Vocals: {vocals_transition_ratio*100:.0f}% ({transition_samples['vocals']/self.sr:.2f}s)")
        
        # Build the mix - Part 1: Before crossfade
        print("\n🔨 Building mix...")
        mix_before = sum(stems1.values())[:fade_start_sample]
        
        # Part 2: During crossfade
        # Determine actual crossfade length (handle padding if needed)
        track1_remaining = len(stems1["vocals"]) - fade_start_sample
        track2_available = len(stems2_trimmed["vocals"])
        
        # Pad if necessary
        if track1_remaining < fade_samples or track2_available < fade_samples:
            print(f"⚠️  Padding stems to complete full {fade_beats}-beat fade")
            
            for stem_name in ["vocals", "drums", "bass", "other"]:
                if track1_remaining < fade_samples:
                    pad_len = fade_samples - (len(stems1[stem_name]) - fade_start_sample)
                    if pad_len > 0:
                        stems1[stem_name] = np.concatenate([stems1[stem_name], np.zeros(pad_len)])
                
                if track2_available < fade_samples:
                    pad_len = fade_samples - len(stems2_trimmed[stem_name])
                    if pad_len > 0:
                        stems2_trimmed[stem_name] = np.concatenate([stems2_trimmed[stem_name], np.zeros(pad_len)])
        
        crossfade_len = fade_samples
        
        # Mix each stem independently with its own transition point
        mixed_stems = {}
        
        # Define blend mode for each stem (bass and drums always cut)
        blend_modes = {
            "bass": "cut",
            "drums": "cut",
            "vocals": vocals_blend_mode,
            "other": other_blend_mode,
        }
        
        for stem_name in ["vocals", "drums", "bass", "other"]:
            # Get segments for this stem
            t1_segment = stems1[stem_name][fade_start_sample:fade_start_sample + crossfade_len]
            t2_segment = stems2_trimmed[stem_name][:crossfade_len]
            
            if blend_modes[stem_name] == "cut":
                # Sharp swap at the specified point
                stem_fade = np.ones(crossfade_len)
                stem_fade[transition_samples[stem_name]:] = 0
                mixed_stems[stem_name] = t1_segment * stem_fade + t2_segment * (1 - stem_fade)
            
            elif blend_modes[stem_name] == "crossfade":
                # Gradual crossfade starting from the transition point
                transition_point = transition_samples[stem_name]
                
                # Before transition point: just track 1
                # After transition point: gradual crossfade
                stem_fade = np.ones(crossfade_len)
                
                # Create gradual fade starting at transition point
                fade_length = crossfade_len - transition_point
                if fade_length > 0:
                    gradual_fade = self.create_fade_curve(fade_length, fade_curve)
                    stem_fade[transition_point:] = 1 - gradual_fade
                
                mixed_stems[stem_name] = t1_segment * stem_fade + t2_segment * (1 - stem_fade)
        
        # Combine all stems
        mix_crossfade = sum(mixed_stems.values())
        
        # Part 3: After crossfade
        mix_after = sum(stem[crossfade_len:] for stem in stems2_trimmed.values())
        
        # Concatenate all parts
        mix_final = np.concatenate([mix_before, mix_crossfade, mix_after])
        
        # Normalize
        max_val = np.abs(mix_final).max()
        if max_val > 0:
            mix_final = mix_final * (0.95 / max_val)
        
        print(f"\n✅ Stem crossfade complete!")
        print(f"   Final duration: {len(mix_final) / self.sr:.2f}s")
        
        return mix_final, self.sr
    
    def save_mix(self, audio: np.ndarray, sr: int, output_path: str):
        """Save the mixed audio to a file."""
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        sf.write(output_path, audio, sr)
        print(f"💾 Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="DJ-style crossfade using Demucs stems with independent transition timing",
        epilog="Tip: Transition 'other' (backing) before vocals for smoother blends!"
    )
    parser.add_argument("track1", help="Path to first audio file")
    parser.add_argument("track2", help="Path to second audio file")
    parser.add_argument("--fade-start-bar", type=int, default=None)
    parser.add_argument("--fade-start-time", type=float, default=None)
    parser.add_argument("--fade-beats", type=int, default=16)
    parser.add_argument("--track2-start-bar", type=int, default=0)
    parser.add_argument("--fade-curve", choices=["linear", "equal_power", "smooth"], default="equal_power")
    
    # Individual stem transition controls
    parser.add_argument("--bass-transition", type=float, default=0.5, 
                       help="Bass transition point (0-1, default: 0.5)")
    parser.add_argument("--drums-transition", type=float, default=0.4,
                       help="Drums transition point (0-1, default: 0.4)")
    parser.add_argument("--vocals-transition", type=float, default=0.7,
                       help="Vocals transition point (0-1, default: 0.7)")
    parser.add_argument("--other-transition", type=float, default=0.6,
                       help="Other/backing transition point (0-1, default: 0.6)")
    
    # Blend mode controls
    parser.add_argument("--vocals-blend", choices=["cut", "crossfade"], default="crossfade",
                       help="Vocals blend mode: 'cut' for sharp swap, 'crossfade' for gradual (default: crossfade)")
    parser.add_argument("--other-blend", choices=["cut", "crossfade"], default="crossfade",
                       help="Other/backing blend mode: 'cut' for sharp swap, 'crossfade' for gradual (default: crossfade)")
    
    parser.add_argument("--metadata-dir", default="data/metadata")
    parser.add_argument("-o", "--out-path", default=None,
                       help="Output file path (default: auto-generated as Song1_Song2_mix.wav)")
    parser.add_argument("--sr", type=int, default=44100)
    
    args = parser.parse_args()
    
    # Auto-generate output filename if not provided
    if args.out_path is None:
        song1_name = os.path.splitext(os.path.basename(args.track1))[0]
        song2_name = os.path.splitext(os.path.basename(args.track2))[0]
        args.out_path = f"data/mixes/{song1_name}_{song2_name}_mix.wav"
        print(f"💾 Auto-generated output name: {args.out_path}")
    
    crossfader = DJStemCrossfader(sr=args.sr)
    
    try:
        mixed_audio, sr = crossfader.crossfade_stems(
            track1_path=args.track1,
            track2_path=args.track2,
            fade_start_bar=args.fade_start_bar,
            fade_start_time=args.fade_start_time,
            fade_beats=args.fade_beats,
            track2_start_bar=args.track2_start_bar,
            fade_curve=args.fade_curve,
            bass_transition_ratio=args.bass_transition,
            drums_transition_ratio=args.drums_transition,
            vocals_transition_ratio=args.vocals_transition,
            other_transition_ratio=args.other_transition,
            vocals_blend_mode=args.vocals_blend,
            other_blend_mode=args.other_blend,
            metadata_dir=args.metadata_dir,
        )
        
        crossfader.save_mix(mixed_audio, sr, args.out_path)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
