"""
Audio Mixer Module - FIXED VERSION
Takes mixing instructions and produces actual mixed audio output
Creates professional DJ mixes that you can listen to!
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Dict, Optional, Tuple
from pathlib import Path
from scipy import signal
from dataclasses import dataclass

from transition_generator import MixingInstructions, TransitionType


@dataclass
class MixedAudioResult:
    """Result of audio mixing"""
    audio: np.ndarray  # Mixed audio array
    sample_rate: int   # Sample rate (typically 44100)
    duration: float    # Duration in seconds
    transition_point: float  # Where the transition happens
    

class AudioMixer:
    """
    Renders actual audio from mixing instructions
    
    DJ Context: This is where the magic happens! We take all the analysis
    and planning, and turn it into actual sound you can hear.
    """
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize audio mixer
        
        Args:
            sample_rate: Output sample rate (44100 is CD quality)
        """
        self.sample_rate = sample_rate
        print(f"🎚️  Audio Mixer initialized (sample rate: {sample_rate} Hz)")
    
    def load_and_prepare_audio(self, file_path: str, target_bpm: Optional[float] = None,
                               target_key: Optional[str] = None) -> Tuple[np.ndarray, float]:
        """
        Load audio and optionally time-stretch/pitch-shift it
        
        Args:
            file_path: Path to audio file
            target_bpm: If provided, time-stretch to this BPM
            target_key: If provided, pitch-shift to this key (not implemented yet)
            
        Returns:
            (audio_array, actual_bpm)
        """
        print(f"   📂 Loading: {Path(file_path).name}")
        
        # Load audio
        y, sr = librosa.load(file_path, sr=self.sample_rate, mono=False)
        
        # Convert to mono if needed (for simplicity)
        if len(y.shape) > 1:
            y = librosa.to_mono(y)
        
        # Detect BPM if we need to time-stretch
        if target_bpm is not None:
            print(f"   🎵 Detecting BPM...")
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            current_bpm = float(tempo)
            
            if abs(current_bpm - target_bpm) > 0.5:  # Only stretch if significant difference
                stretch_factor = target_bpm / current_bpm
                print(f"   ⏩ Time-stretching: {current_bpm:.1f} → {target_bpm:.1f} BPM (factor: {stretch_factor:.3f})")
                y = librosa.effects.time_stretch(y, rate=stretch_factor)
            else:
                print(f"   ✓ BPM match: {current_bpm:.1f} BPM (no stretching needed)")
        else:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            current_bpm = float(tempo)
        
        return y, current_bpm
    
    def apply_eq(self, audio: np.ndarray, bass_gain: float = 1.0, 
                 mid_gain: float = 1.0, high_gain: float = 1.0) -> np.ndarray:
        """
        Apply 3-band EQ to audio
        
        DJ Context: This is how we prevent bass clashing and create smooth transitions.
        We can cut the bass of the outgoing track while bringing in the incoming track.
        
        Args:
            audio: Input audio
            bass_gain: Gain for bass frequencies (0-1+)
            mid_gain: Gain for mid frequencies (0-1+)
            high_gain: Gain for high frequencies (0-1+)
            
        Returns:
            EQ'd audio
        """
        # Define frequency bands (typical DJ mixer bands)
        # Bass: 20-250 Hz
        # Mid: 250-4000 Hz  
        # High: 4000-20000 Hz
        
        if bass_gain == 1.0 and mid_gain == 1.0 and high_gain == 1.0:
            return audio  # No EQ needed
        
        # Create three band-pass filters
        nyquist = self.sample_rate / 2
        
        # Bass filter (low-pass at 250 Hz)
        bass_sos = signal.butter(4, 250 / nyquist, btype='low', output='sos')
        bass = np.asarray(signal.sosfilt(bass_sos, audio))
        
        # High filter (high-pass at 4000 Hz)
        high_sos = signal.butter(4, 4000 / nyquist, btype='high', output='sos')
        high = np.asarray(signal.sosfilt(high_sos, audio))
        
        # Mid is what's left
        mid = audio - bass - high
        
        # Apply gains
        bass = bass * bass_gain
        mid = mid * mid_gain
        high = high * high_gain
        
        # Recombine
        return bass + mid + high
    
    def apply_eq_fade(self, audio: np.ndarray, bass_range: Tuple[float, float],
                      mid_range: Tuple[float, float], high_range: Tuple[float, float]) -> np.ndarray:
        """
        Apply EQ with fade from start to end values
        
        Args:
            audio: Input audio
            bass_range: (start_gain, end_gain) for bass
            mid_range: (start_gain, end_gain) for mid
            high_range: (start_gain, end_gain) for high
        """
        n_samples = len(audio)
        
        # Create filter banks
        nyquist = self.sample_rate / 2
        bass_sos = signal.butter(4, 250 / nyquist, btype='low', output='sos')
        high_sos = signal.butter(4, 4000 / nyquist, btype='high', output='sos')
        
        # Split into frequency bands
        bass = np.asarray(signal.sosfilt(bass_sos, audio))
        high = np.asarray(signal.sosfilt(high_sos, audio))
        mid = audio - bass - high
        
        # Create gain curves
        bass_curve = np.linspace(bass_range[0], bass_range[1], n_samples)
        mid_curve = np.linspace(mid_range[0], mid_range[1], n_samples)
        high_curve = np.linspace(high_range[0], high_range[1], n_samples)
        
        # Apply fades
        bass = bass * bass_curve
        mid = mid * mid_curve
        high = high * high_curve
        
        return bass + mid + high
    
    def mix_two_tracks(self, track_a_path: str, track_b_path: str,
                       instructions: MixingInstructions,
                       target_bpm: Optional[float] = None) -> MixedAudioResult:
        """
        Mix two tracks according to mixing instructions - FIXED VERSION
        
        This is the main function that creates a professional DJ mix!
        
        Args:
            track_a_path: Path to first track
            track_b_path: Path to second track
            instructions: Mixing instructions from DJ algorithm
            target_bpm: Optional BPM to match both tracks to
            
        Returns:
            MixedAudioResult with the mixed audio
        """
        print(f"\n🎧 Mixing Tracks")
        print(f"{'='*60}")
        
        # Extract key parameters
        transition = instructions.transition_point
        a_out_time = transition.track_a_out_time
        b_in_time = transition.track_b_in_time
        crossfade_duration = transition.crossfade_duration
        
        print(f"Transition Type: {transition.transition_type.value}")
        print(f"Track A out: {a_out_time:.1f}s")
        print(f"Track B in: {b_in_time:.1f}s")
        print(f"Crossfade: {crossfade_duration:.1f}s")
        
        # Load both tracks
        print(f"\n📂 Loading audio files...")
        track_a, bpm_a = self.load_and_prepare_audio(track_a_path, target_bpm)
        track_b, bpm_b = self.load_and_prepare_audio(track_b_path, target_bpm)
        
        # Calculate sample positions
        a_out_sample = int(a_out_time * self.sample_rate)
        a_crossfade_end_sample = int((a_out_time + crossfade_duration) * self.sample_rate)
        b_in_sample = int(b_in_time * self.sample_rate)
        crossfade_samples = int(crossfade_duration * self.sample_rate)
        
        print(f"\n✂️  Extracting sections...")
        print(f"   Track A: play until {a_out_time:.1f}s, then crossfade for {crossfade_duration:.1f}s")
        print(f"   Track B: skip first {b_in_time:.1f}s, start during crossfade")
        
        # Build the mix in three parts:
        # 1. Track A before crossfade (full volume)
        # 2. Crossfade region (both tracks blending)
        # 3. Track B after crossfade (full volume)
        
        # Part 1: Track A before crossfade
        track_a_before = track_a[:a_out_sample]
        
        # Part 2: Crossfade region
        track_a_crossfade = track_a[a_out_sample:a_crossfade_end_sample]
        track_b_crossfade = track_b[b_in_sample:b_in_sample + crossfade_samples]
        
        # Ensure both crossfade sections are the same length
        min_crossfade_len = min(len(track_a_crossfade), len(track_b_crossfade))
        track_a_crossfade = track_a_crossfade[:min_crossfade_len]
        track_b_crossfade = track_b_crossfade[:min_crossfade_len]
        
        # Part 3: Track B after crossfade
        track_b_after = track_b[b_in_sample + min_crossfade_len:]
        
        # Apply EQ to each section
        print(f"\n🎛️  Applying EQ curves...")
        
        # Track A: full EQ before, fade out bass during crossfade
        track_a_before = self.apply_eq(track_a_before, 1.0, 1.0, 1.0)
        if len(instructions.track_a_eq.bass) > 0:
            avg_bass_a = max(float(np.mean(instructions.track_a_eq.bass)), 0.2)
            track_a_crossfade = self.apply_eq_fade(
                track_a_crossfade, 
                (1.0, avg_bass_a),  # Bass fades down
                (1.0, 1.0),          # Mids stay full
                (1.0, 1.0)           # Highs stay full
            )
        
        # Track B: fade in bass during crossfade, full EQ after
        if len(instructions.track_b_eq.bass) > 0:
            avg_bass_b = max(float(np.mean(instructions.track_b_eq.bass)), 0.2)
            track_b_crossfade = self.apply_eq_fade(
                track_b_crossfade,
                (avg_bass_b, 1.0),  # Bass fades up
                (1.0, 1.0),          # Mids stay full
                (1.0, 1.0)           # Highs stay full
            )
        track_b_after = self.apply_eq(track_b_after, 1.0, 1.0, 1.0)
        
        # Apply volume crossfade (equal power)
        print(f"\n🔊 Applying volume crossfade...")
        fade_out = np.sqrt(np.linspace(1, 0, min_crossfade_len))
        fade_in = np.sqrt(np.linspace(0, 1, min_crossfade_len))
        
        track_a_crossfade = track_a_crossfade * fade_out
        track_b_crossfade = track_b_crossfade * fade_in
        
        # Mix the crossfade region
        crossfade_mixed = track_a_crossfade + track_b_crossfade
        
        # Concatenate all parts
        print(f"\n🎼 Assembling final mix...")
        mixed = np.concatenate([
            track_a_before,      # Track A at full volume
            crossfade_mixed,     # Crossfaded region
            track_b_after        # Track B at full volume
        ])
        
        # Normalize to prevent clipping
        print(f"\n🎚️  Normalizing audio...")
        max_val = np.abs(mixed).max()
        if max_val > 0.95:
            mixed = mixed * (0.95 / max_val)
        
        duration = len(mixed) / self.sample_rate
        
        print(f"\n✅ Mix complete!")
        print(f"   Total duration: {duration:.1f}s")
        print(f"   Transition point: {a_out_time:.1f}s")
        print(f"   Track A plays for: {a_out_time:.1f}s")
        print(f"   Crossfade lasts: {min_crossfade_len / self.sample_rate:.1f}s")
        print(f"   Track B continues for: {len(track_b_after) / self.sample_rate:.1f}s")
        
        return MixedAudioResult(
            audio=mixed,
            sample_rate=self.sample_rate,
            duration=duration,
            transition_point=a_out_time
        )
    
    def save_mix(self, result: MixedAudioResult, output_path: str):
        """
        Save mixed audio to file
        
        Args:
            result: MixedAudioResult from mix_two_tracks
            output_path: Where to save (e.g., "my_mix.wav")
        """
        print(f"\n💾 Saving mix to: {output_path}")
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save as WAV file (high quality)
        sf.write(output_path, result.audio, result.sample_rate)
        
        file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"   ✓ Saved! ({file_size_mb:.1f} MB)")
        print(f"   🎧 You can now listen to your mix!")


def quick_mix(track_a_path: str, track_b_path: str, output_path: str,
              crossfade_duration: float = 16.0):
    """
    Quick and simple mix between two tracks with default settings
    
    Args:
        track_a_path: Path to first track
        track_b_path: Path to second track  
        output_path: Where to save the mix
        crossfade_duration: Duration of crossfade in seconds
    """
    from transition_generator import TransitionPoint, TransitionType, MixingInstructions, EQCurve
    
    print("🎵 Quick Mix Mode")
    print("="*60)
    
    # Create simple default instructions
    mixer = AudioMixer()
    
    # Load track A to get duration
    y_a, _ = librosa.load(track_a_path, sr=mixer.sample_rate)
    a_duration = len(y_a) / mixer.sample_rate
    a_out_time = max(a_duration - crossfade_duration - 15, a_duration * 0.7)  # Start 15s before end or at 70%
    
    # Load track B
    y_b, _ = librosa.load(track_b_path, sr=mixer.sample_rate)
    b_duration = len(y_b) / mixer.sample_rate
    b_in_time = min(8.0, b_duration * 0.1)  # Start at 8s or 10% into track
    
    # Create simple instructions
    transition = TransitionPoint(
        track_a_out_time=a_out_time,
        track_b_in_time=b_in_time,
        crossfade_duration=crossfade_duration,
        transition_type=TransitionType.BEAT_MATCHED,
        confidence=1.0
    )
    
    # Simple EQ: reduce bass on outgoing track during crossfade
    eq_a = EQCurve(
        timestamps=[0, crossfade_duration],
        bass=[1.0, 0.3],
        mid=[1.0, 1.0],
        high=[1.0, 1.0]
    )
    
    eq_b = EQCurve(
        timestamps=[0, crossfade_duration],
        bass=[0.3, 1.0],
        mid=[1.0, 1.0],
        high=[1.0, 1.0]
    )
    
    # Simple volume automation (handled in mixer now)
    volume_a = []
    volume_b = []
    
    instructions = MixingInstructions(
        transition_point=transition,
        track_a_eq=eq_a,
        track_b_eq=eq_b,
        track_a_volume=volume_a,
        track_b_volume=volume_b,
        effects=[],
        notes="Quick mix with default settings"
    )
    
    # Mix it!
    result = mixer.mix_two_tracks(track_a_path, track_b_path, instructions)
    mixer.save_mix(result, output_path)
    
    print(f"\n{'='*60}")
    print(f"✅ Done! Your mix is ready at: {output_path}")
    print(f"{'='*60}")