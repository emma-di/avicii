"""
Beat-Synced Transition System
Aligns beats perfectly during crossfades for professional DJ mixing
"""

import numpy as np
from typing import Dict, Tuple, Any
from scipy import signal
import librosa


class BeatSyncedTransition:
    """
    Creates beat-matched transitions between tracks
    Uses time-stretching and phase alignment for perfect sync
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
    
    def create_beat_matched_transition(
        self,
        track_a: np.ndarray,
        track_b: np.ndarray,
        track_a_info: Dict,
        track_b_info: Dict,
        crossfade_beats: int = 16  # Number of beats to crossfade over
    ) -> Tuple[Any, Any, Dict]:
        """
        Create a beat-matched transition between two tracks
        
        Returns:
            track_a_processed: Track A with effects and fade (numpy array)
            track_b_processed: Track B time-stretched and beat-aligned (numpy array)
            transition_info: Details about the transition (dict)
        """
        bpm_a = track_a_info['bpm']
        bpm_b = track_b_info['bpm']
        
        print(f"\n      🎵 Beat-matching transition")
        print(f"         Track A: {bpm_a:.1f} BPM")
        print(f"         Track B: {bpm_b:.1f} BPM")
        
        # Time-stretch track B to match track A's tempo
        stretch_ratio = bpm_a / bpm_b
        
        if abs(stretch_ratio - 1.0) > 0.02:  # More than 2% difference
            print(f"         Time-stretching Track B: {stretch_ratio:.3f}x")
            track_b = self._time_stretch(track_b, stretch_ratio)
        else:
            print(f"         BPMs close enough, no stretch needed")
        
        # Calculate crossfade duration in samples
        beat_duration = 60.0 / bpm_a  # seconds per beat
        crossfade_duration = crossfade_beats * beat_duration
        crossfade_samples = int(crossfade_duration * self.sr)
        
        # Ensure crossfade isn't too long
        crossfade_samples = min(
            crossfade_samples,
            len(track_a),
            len(track_b)
        )
        
        print(f"         Crossfade: {crossfade_beats} beats ({crossfade_duration:.1f}s)")
        
        # Apply EQ-based transition (bass swap technique)
        track_a_faded, track_b_faded = self._apply_eq_transition(
            track_a, track_b, crossfade_samples, bpm_a
        )
        
        transition_info = {
            'crossfade_samples': crossfade_samples,
            'crossfade_duration': crossfade_duration,
            'stretch_ratio': stretch_ratio,
            'transition_type': 'bass_swap'
        }
        
        return track_a_faded, track_b_faded, transition_info
    
    def _time_stretch(self, audio: np.ndarray, ratio: float) -> Any:
        """
        Time-stretch audio to match BPM without changing pitch
        Returns: time-stretched audio as numpy array
        """
        if audio.ndim == 1:
            return librosa.effects.time_stretch(audio, rate=ratio)
        else:
            # Stretch each channel
            stretched = np.zeros((int(len(audio) / ratio), audio.shape[1]))
            for ch in range(audio.shape[1]):
                stretched[:, ch] = librosa.effects.time_stretch(audio[:, ch], rate=ratio)
            return stretched
    
    def _apply_eq_transition(
        self,
        track_a: np.ndarray,
        track_b: np.ndarray,
        crossfade_samples: int,
        bpm: float
    ) -> Tuple[Any, Any]:
        """
        Apply EQ-based transition (bass swap)
        Track A's bass fades out while Track B's bass fades in
        Creates smooth energy transfer
        Returns: (track_a_processed, track_b_processed) as numpy arrays
        """
        track_a = track_a.copy()
        track_b = track_b.copy()
        
        # Work on the crossfade regions
        crossfade_samples = min(crossfade_samples, len(track_a), len(track_b))
        
        # Track A: fade out last N samples
        a_fade_region = track_a[-crossfade_samples:]
        
        # Track B: fade in first N samples
        b_fade_region = track_b[:crossfade_samples]
        
        # Split into bass and highs
        a_bass, a_highs = self._split_frequency(a_fade_region, 250)
        b_bass, b_highs = self._split_frequency(b_fade_region, 250)
        
        # Create crossfade curves
        fade_out = np.linspace(1, 0, crossfade_samples)
        fade_in = np.linspace(0, 1, crossfade_samples)
        
        # Add beat-synced curve (makes it more rhythmic)
        beat_samples = int((60.0 / bpm) * self.sr)
        num_beats = crossfade_samples // beat_samples
        
        if num_beats > 0:
            # Create stepped fade for more dramatic effect
            step_fade_out = np.repeat(np.linspace(1, 0, num_beats), beat_samples)[:crossfade_samples]
            step_fade_in = np.repeat(np.linspace(0, 1, num_beats), beat_samples)[:crossfade_samples]
            
            # Blend stepped and smooth fades
            fade_out = 0.7 * fade_out + 0.3 * step_fade_out
            fade_in = 0.7 * fade_in + 0.3 * step_fade_in
        
        # Apply crossfade curves
        if a_bass.ndim == 1:
            # Bass swap: A's bass out, B's bass in
            a_bass *= fade_out
            b_bass *= fade_in
            
            # Highs: slower crossfade for smoothness
            a_highs *= (fade_out ** 0.5)
            b_highs *= (fade_in ** 0.5)
        else:
            fade_out_2d = fade_out[:, np.newaxis]
            fade_in_2d = fade_in[:, np.newaxis]
            
            a_bass *= fade_out_2d
            b_bass *= fade_in_2d
            
            a_highs *= (fade_out_2d ** 0.5)
            b_highs *= (fade_in_2d ** 0.5)
        
        # Recombine
        track_a[-crossfade_samples:] = a_bass + a_highs
        track_b[:crossfade_samples] = b_bass + b_highs
        
        return track_a, track_b
    
    def _split_frequency(self, audio: np.ndarray, cutoff: float = 250) -> Tuple[Any, Any]:
        """
        Split audio into bass (below cutoff) and highs (above cutoff)
        Returns: (bass, highs) as numpy arrays
        """
        # Low-pass for bass
        sos_low = signal.butter(4, cutoff, 'lowpass', fs=self.sr, output='sos')
        # High-pass for highs
        sos_high = signal.butter(4, cutoff, 'highpass', fs=self.sr, output='sos')
        
        if audio.ndim == 1:
            bass = signal.sosfilt(sos_low, audio)
            highs = signal.sosfilt(sos_high, audio)
        else:
            bass = np.zeros_like(audio)
            highs = np.zeros_like(audio)
            for ch in range(audio.shape[1]):
                bass[:, ch] = signal.sosfilt(sos_low, audio[:, ch])
                highs[:, ch] = signal.sosfilt(sos_high, audio[:, ch])
        
        return bass, highs
    
    def align_beats(
        self,
        track_a: np.ndarray,
        track_b: np.ndarray,
        beat_grid_a: Any,  # Can be BeatGrid object, list, or dict
        beat_grid_b: Any   # Can be BeatGrid object, list, or dict
    ) -> Tuple[Any, int]:
        """
        Align the first beat of track B with the last beat of track A
        Returns: (track_b_aligned, offset_samples) - aligned audio and offset in samples
        """
        # Get beat times - handle multiple types
        beats_a = []
        if beat_grid_a:
            if hasattr(beat_grid_a, 'beat_times'):
                beats_a = list(beat_grid_a.beat_times)
            elif hasattr(beat_grid_a, 'beats'):
                beats_a = list(beat_grid_a.beats)
            elif isinstance(beat_grid_a, (list, np.ndarray)):
                beats_a = list(beat_grid_a)
        
        beats_b = []
        if beat_grid_b:
            if hasattr(beat_grid_b, 'beat_times'):
                beats_b = list(beat_grid_b.beat_times)
            elif hasattr(beat_grid_b, 'beats'):
                beats_b = list(beat_grid_b.beats)
            elif isinstance(beat_grid_b, (list, np.ndarray)):
                beats_b = list(beat_grid_b)
        
        if not beats_a or not beats_b:
            return track_b, 0
        
        # Find where track A's exit point would be
        # and align track B's first beat there
        last_beat_a = beats_a[-1]
        first_beat_b = beats_b[0]
        
        # Calculate offset
        offset_samples = int((last_beat_a - first_beat_b) * self.sr)
        
        # Shift track B
        if offset_samples > 0:
            # Delay track B
            track_b_aligned = np.pad(track_b, ((offset_samples, 0), (0, 0)) if track_b.ndim > 1 else (offset_samples, 0), mode='constant')
        elif offset_samples < 0:
            # Start track B earlier
            track_b_aligned = track_b[abs(offset_samples):]
        else:
            track_b_aligned = track_b
        
        return track_b_aligned, offset_samples


class CreativeMixTechniques:
    """
    Advanced DJ techniques like muting, filtering, and creative transitions
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
    
    def apply_channel_mute_effect(
        self,
        audio: np.ndarray,
        mute_start: float,
        mute_duration: float,
        channel: str = 'bass'  # 'bass', 'highs', 'vocals'
    ) -> np.ndarray:
        """
        Mute a specific frequency range then bring it back
        Creates dramatic build-up effect
        """
        audio = audio.copy()
        
        start_sample = int(mute_start * self.sr)
        duration_samples = int(mute_duration * self.sr)
        end_sample = min(start_sample + duration_samples, len(audio))
        
        if start_sample >= len(audio):
            return audio
        
        segment = audio[start_sample:end_sample].copy()
        
        # Define frequency ranges
        if channel == 'bass':
            # Remove bass (below 250Hz)
            sos = signal.butter(4, 250, 'highpass', fs=self.sr, output='sos')
        elif channel == 'highs':
            # Remove highs (above 4kHz)
            sos = signal.butter(4, 4000, 'lowpass', fs=self.sr, output='sos')
        elif channel == 'vocals':
            # Remove vocal range (250Hz - 4kHz) 
            sos_low = signal.butter(4, 250, 'lowpass', fs=self.sr, output='sos')
            sos_high = signal.butter(4, 4000, 'highpass', fs=self.sr, output='sos')
        
        # Apply filter
        if audio.ndim == 1:
            if channel == 'vocals':
                filtered = signal.sosfilt(sos_low, segment) + signal.sosfilt(sos_high, segment)
            else:
                filtered = signal.sosfilt(sos, segment)
            audio[start_sample:end_sample] = filtered
        else:
            for ch in range(audio.shape[1]):
                if channel == 'vocals':
                    filtered = signal.sosfilt(sos_low, segment[:, ch]) + signal.sosfilt(sos_high, segment[:, ch])
                else:
                    filtered = signal.sosfilt(sos, segment[:, ch])
                audio[start_sample:end_sample, ch] = filtered
        
        # Fade back in at the end
        fade_samples = int(0.1 * self.sr)  # 100ms fade
        if end_sample - fade_samples > start_sample:
            fade = np.linspace(0, 1, fade_samples)
            original_section = audio[end_sample - fade_samples:end_sample].copy()
            
            # Get original unfiltered section
            original_unfiltered = audio[start_sample:start_sample + len(audio[start_sample:end_sample])].copy()
            
            if audio.ndim == 1:
                # Crossfade between filtered and original
                audio[end_sample - fade_samples:end_sample] = (
                    audio[end_sample - fade_samples:end_sample] * (1 - fade) +
                    original_section * fade
                )
            else:
                audio[end_sample - fade_samples:end_sample] = (
                    audio[end_sample - fade_samples:end_sample] * (1 - fade)[:, np.newaxis] +
                    original_section * fade[:, np.newaxis]
                )
        
        return audio
    
    def apply_breakdown_buildup(
        self,
        audio: np.ndarray,
        breakdown_start: float,
        bpm: float,
        breakdown_bars: int = 4
    ) -> np.ndarray:
        """
        Create a breakdown-buildup section
        Removes elements progressively then builds back up
        """
        audio = audio.copy()
        
        beat_duration = 60.0 / bpm
        bar_duration = beat_duration * 4  # 4 beats per bar
        breakdown_duration = breakdown_bars * bar_duration
        
        start_sample = int(breakdown_start * self.sr)
        duration_samples = int(breakdown_duration * self.sr)
        end_sample = min(start_sample + duration_samples, len(audio))
        
        if start_sample >= len(audio):
            return audio
        
        segment = audio[start_sample:end_sample].copy()
        segment_length = len(segment)
        
        # Progressive filter sweep (remove bass, then mids, then everything)
        for i in range(0, segment_length, int(self.sr * 0.1)):  # Every 100ms
            chunk_end = min(i + int(self.sr * 0.1), segment_length)
            progress = i / segment_length
            
            # Increase cutoff frequency over time
            cutoff = 100 + progress * 8000  # 100Hz to 8kHz
            
            sos = signal.butter(4, cutoff, 'highpass', fs=self.sr, output='sos')
            
            if audio.ndim == 1:
                segment[i:chunk_end] = signal.sosfilt(sos, segment[i:chunk_end])
                # Also reduce volume
                segment[i:chunk_end] *= (1 - progress * 0.5)
            else:
                for ch in range(segment.shape[1]):
                    segment[i:chunk_end, ch] = signal.sosfilt(sos, segment[i:chunk_end, ch])
                    segment[i:chunk_end, ch] *= (1 - progress * 0.5)
        
        audio[start_sample:end_sample] = segment
        
        return audio


if __name__ == "__main__":
    print("Beat-Synced Transition System")
    print("=" * 60)
    print("Features:")
    print("  ✓ BPM matching with time-stretching")
    print("  ✓ Beat alignment")
    print("  ✓ Bass swap transitions")
    print("  ✓ EQ-based crossfades")
    print("  ✓ Channel muting effects")
    print("  ✓ Breakdown/buildup sections")
    print("\nCreates professional, beat-matched transitions!")