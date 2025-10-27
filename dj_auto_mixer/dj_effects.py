"""
PARTY DJ Effects System
BALANCED effects with selective bass boost (not constant!)
Makes it OBVIOUS a DJ mixed this, but not jarring!
"""

import numpy as np
from scipy import signal
from typing import Dict, Tuple, Optional, List
import librosa


class DJEffects:
    """
    Professional DJ effects processor with BALANCED effects
    
    Effects include:
    - Selective bass boost (only where appropriate)
    - Filter buildups/sweeps
    - White noise risers
    - Echo outs with feedback
    - Reverb throws
    - Pitch shifts/drops
    - Beat drops
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
    
    def smart_effect_placement(self, audio: np.ndarray, track_analysis: Dict,
                              energy_level: str = 'medium') -> np.ndarray:
        """
        Apply BALANCED DJ effects that enhance the mix naturally
        
        Args:
            audio: Audio data (samples x channels)
            track_analysis: Track structure info
            energy_level: 'low', 'medium', or 'high'
            
        Returns:
            Audio with professional DJ effects
        """
        structure = track_analysis['structure']
        duration = len(audio) / self.sr
        beat_grid = track_analysis.get('beat_grid', None)
        bpm = track_analysis.get('bpm', 120)
        
        print(f"      🎧 Applying {energy_level} energy effects")
        
        # Get beat times for synchronization
        beat_times = self._extract_beat_times(beat_grid)
        
        # EFFECT 1: SELECTIVE BASS BOOST (only in high energy sections, not constant!)
        # Only boost bass for high energy tracks in specific sections
        if energy_level == 'high' and duration > 30:
            # Boost bass only in the middle section (the peak)
            boost_start = duration * 0.3  # Start at 30%
            boost_end = duration * 0.7    # End at 70%
            audio = self.apply_selective_bass_boost(
                audio, boost_start, boost_end, boost_db=4.0
            )
            print(f"        ✓ Selective bass boost ({boost_start:.1f}s - {boost_end:.1f}s)")
        
        # EFFECT 2: Beat drop (only for medium/high energy, in the middle)
        if energy_level in ['medium', 'high'] and duration > 30 and len(beat_times) > 8:
            drop_time = duration * 0.5  # Middle of track
            drop_beat = self._find_nearest_beat(drop_time, beat_times)
            audio = self.apply_beat_drop(audio, drop_beat, duration_beats=4, bpm=bpm)
            print(f"        ✓ Beat drop at {drop_beat:.1f}s")
        
        # EFFECT 3: Filter buildup before exit (essential for transitions)
        if duration > 15:
            buildup_start = duration - 14
            audio = self.apply_filter_buildup(
                audio, int(buildup_start * self.sr), 
                duration_sec=10.0, max_freq=2500
            )
            print(f"        ✓ Filter buildup")
        
        # EFFECT 4: Riser before climax (high energy only)
        if energy_level == 'high' and duration > 20:
            riser_start = duration - 10
            audio = self.add_riser(audio, riser_start, duration_sec=6.0)
            print(f"        ✓ Build riser")
        
        # EFFECT 5: Echo out at the end (classic DJ exit)
        if duration > 10:
            echo_start = duration - 7
            audio = self.apply_echo_out(audio, int(echo_start * self.sr), repeats=6, decay=0.7)
            print(f"        ✓ Echo out")
        
        return audio
    
    def _extract_beat_times(self, beat_grid) -> List[float]:
        """Extract beat times from BeatGrid object or list"""
        beat_times = []
        if beat_grid:
            if hasattr(beat_grid, 'beat_times'):
                beat_times = list(beat_grid.beat_times)
            elif hasattr(beat_grid, 'beats'):
                beat_times = list(beat_grid.beats)
            elif isinstance(beat_grid, (list, np.ndarray)):
                beat_times = list(beat_grid)
        return beat_times
    
    def _find_nearest_beat(self, time: float, beat_times: List[float]) -> float:
        """Find the nearest beat to a given time"""
        if not beat_times:
            return time
        return min(beat_times, key=lambda x: abs(x - time))
    
    def apply_selective_bass_boost(self, audio: np.ndarray, start_time: float,
                                   end_time: float, boost_db: float = 4.0) -> np.ndarray:
        """
        Boost bass only in a specific section - not the whole track!
        This makes it feel more natural and less jarring
        """
        audio = audio.copy()
        
        start_sample = int(start_time * self.sr)
        end_sample = int(end_time * self.sr)
        
        if start_sample >= len(audio) or end_sample <= start_sample:
            return audio
        
        end_sample = min(end_sample, len(audio))
        
        # Extract the section to boost
        section = audio[start_sample:end_sample].copy()
        
        # Create low shelf filter to boost bass (20-250Hz)
        gain = 10 ** (boost_db / 20)
        freq = 200  # Hz - boost below this
        Q = 0.707
        w0 = 2 * np.pi * freq / self.sr
        alpha = np.sin(w0) / (2 * Q)
        A = np.sqrt(gain)
        
        # Low shelf coefficients
        b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
        a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
        
        # Normalize
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1 / a0, a2 / a0])
        
        # Apply filter to section
        if section.ndim == 1:
            section = signal.filtfilt(b, a, section)
        else:
            for ch in range(section.shape[1]):
                section[:, ch] = signal.filtfilt(b, a, section[:, ch])
        
        # Fade in/out the boost to avoid abrupt changes
        fade_length = int(2.0 * self.sr)  # 2 second fade
        if len(section) > fade_length * 2:
            fade_in = np.linspace(0, 1, fade_length)
            fade_out = np.linspace(1, 0, fade_length)
            
            if section.ndim == 1:
                # Crossfade at start
                audio[start_sample:start_sample + fade_length] = (
                    audio[start_sample:start_sample + fade_length] * (1 - fade_in) +
                    section[:fade_length] * fade_in
                )
                # Full boost in middle
                audio[start_sample + fade_length:end_sample - fade_length] = section[fade_length:-fade_length]
                # Crossfade at end
                audio[end_sample - fade_length:end_sample] = (
                    audio[end_sample - fade_length:end_sample] * (1 - fade_out) +
                    section[-fade_length:] * fade_out
                )
            else:
                fade_in_2d = fade_in[:, np.newaxis]
                fade_out_2d = fade_out[:, np.newaxis]
                audio[start_sample:start_sample + fade_length] = (
                    audio[start_sample:start_sample + fade_length] * (1 - fade_in_2d) +
                    section[:fade_length] * fade_in_2d
                )
                audio[start_sample + fade_length:end_sample - fade_length] = section[fade_length:-fade_length]
                audio[end_sample - fade_length:end_sample] = (
                    audio[end_sample - fade_length:end_sample] * (1 - fade_out_2d) +
                    section[-fade_length:] * fade_out_2d
                )
        else:
            # Section too short for fades, just apply directly
            audio[start_sample:end_sample] = section
        
        return audio
    
    def apply_bass_boost(self, audio: np.ndarray, boost_db: float = 6.0) -> np.ndarray:
        """
        Boost the bass frequencies across entire track
        NOTE: This is the OLD method - we now use selective_bass_boost instead!
        """
        audio = audio.copy()
        
        # Low shelf filter to boost bass (20-250Hz)
        gain = 10 ** (boost_db / 20)
        freq = 200  # Hz - boost below this
        Q = 0.707
        w0 = 2 * np.pi * freq / self.sr
        alpha = np.sin(w0) / (2 * Q)
        A = np.sqrt(gain)
        
        # Low shelf coefficients
        b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
        a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
        
        # Normalize
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1 / a0, a2 / a0])
        
        # Apply filter
        if audio.ndim == 1:
            audio = signal.filtfilt(b, a, audio)
        else:
            for ch in range(audio.shape[1]):
                audio[:, ch] = signal.filtfilt(b, a, audio[:, ch])
        
        return audio
    
    def apply_beat_drop(self, audio: np.ndarray, drop_time: float, 
                       duration_beats: int = 4, bpm: float = 120) -> np.ndarray:
        """
        BEAT DROP - remove bass/elements then DROP them back!
        Classic DJ technique for building tension
        """
        audio = audio.copy()
        
        # Calculate duration in seconds
        beat_duration = 60.0 / bpm
        drop_duration = duration_beats * beat_duration
        
        drop_sample = int(drop_time * self.sr)
        drop_length = int(drop_duration * self.sr)
        end_sample = min(drop_sample + drop_length, len(audio))
        
        if drop_sample >= len(audio):
            return audio
        
        # Get segment
        segment = audio[drop_sample:end_sample].copy()
        segment_length = len(segment)
        
        # Remove bass for first half (tension)
        half_point = segment_length // 2
        
        # High-pass filter (remove bass)
        sos = signal.butter(6, 300, 'highpass', fs=self.sr, output='sos')
        
        if segment.ndim == 1:
            segment[:half_point] = signal.sosfilt(sos, segment[:half_point])
            # Add slight volume reduction
            segment[:half_point] *= 0.7
        else:
            for ch in range(segment.shape[1]):
                segment[:half_point, ch] = signal.sosfilt(sos, segment[:half_point, ch])
                segment[:half_point, ch] *= 0.7
        
        # Smooth crossfade back to full bass
        fade_length = int(0.1 * self.sr)  # 100ms fade
        if half_point + fade_length < segment_length:
            fade = np.linspace(0, 1, fade_length)
            
            if segment.ndim == 1:
                # Crossfade between filtered and original
                filtered_section = segment[half_point:half_point + fade_length].copy()
                original_section = audio[drop_sample + half_point:drop_sample + half_point + fade_length].copy()
                segment[half_point:half_point + fade_length] = (
                    filtered_section * (1 - fade) + original_section * fade
                )
            else:
                for ch in range(segment.shape[1]):
                    filtered_section = segment[half_point:half_point + fade_length, ch].copy()
                    original_section = audio[drop_sample + half_point:drop_sample + half_point + fade_length, ch].copy()
                    segment[half_point:half_point + fade_length, ch] = (
                        filtered_section * (1 - fade) + original_section * fade
                    )
        
        audio[drop_sample:end_sample] = segment
        return audio
    
    def apply_filter_buildup(self, audio: np.ndarray, start_sample: int,
                            duration_sec: float = 10.0, 
                            max_freq: int = 2500) -> np.ndarray:
        """
        Filter buildup - gradually removes bass (not too aggressive)
        Perfect for transitions
        """
        audio = audio.copy()
        duration_samples = int(duration_sec * self.sr)
        end_sample = min(start_sample + duration_samples, len(audio))
        
        if start_sample >= len(audio) or end_sample <= start_sample:
            return audio
        
        segment = audio[start_sample:end_sample]
        segment_length = len(segment)
        
        # Moderate frequency sweep (not extreme)
        freq_start = 100  # Hz - start low
        freq_end = max_freq  # Hz - end at max_freq (2500Hz default)
        
        # Apply filter gradually
        chunk_size = int(self.sr * 0.1)  # 100ms chunks
        for i in range(0, segment_length, chunk_size):
            chunk_end = min(i + chunk_size, segment_length)
            progress = i / segment_length
            
            # Linear curve (smooth, not dramatic)
            current_freq = freq_start + (freq_end - freq_start) * progress
            
            # Moderate filter (order 4, not 8)
            sos = signal.butter(4, current_freq, 'highpass', fs=self.sr, output='sos')
            
            if segment.ndim == 1:
                segment[i:chunk_end] = signal.sosfilt(sos, segment[i:chunk_end])
            else:
                for ch in range(segment.shape[1]):
                    segment[i:chunk_end, ch] = signal.sosfilt(sos, segment[i:chunk_end, ch])
        
        audio[start_sample:end_sample] = segment
        return audio
    
    def apply_echo_out(self, audio: np.ndarray, start_sample: int,
                      repeats: int = 6, decay: float = 0.65) -> np.ndarray:
        """
        Echo-out with repeats and tail
        """
        audio = audio.copy()
        
        if start_sample >= len(audio):
            return audio
        
        # Rhythmic delay (1/4 note at 120 BPM)
        delay_samples = int(self.sr * 0.5)  # 500ms
        
        for i in range(1, repeats + 1):
            echo_pos = start_sample + (i * delay_samples)
            
            if echo_pos >= len(audio):
                break
            
            echo_amp = decay ** i
            
            source_end = min(start_sample + delay_samples, len(audio))
            echo_end = min(echo_pos + (source_end - start_sample), len(audio))
            source_length = source_end - start_sample
            echo_length = echo_end - echo_pos
            
            actual_length = min(source_length, echo_length)
            
            if actual_length > 0:
                if audio.ndim == 1:
                    audio[echo_pos:echo_pos + actual_length] += (
                        audio[start_sample:start_sample + actual_length] * echo_amp
                    )
                else:
                    audio[echo_pos:echo_pos + actual_length] += (
                        audio[start_sample:start_sample + actual_length] * echo_amp
                    )
        
        return audio
    
    def add_riser(self, audio: np.ndarray, start_time: float,
                 duration_sec: float = 6.0) -> np.ndarray:
        """
        White noise riser - builds tension before a drop
        """
        audio = audio.copy()
        start_sample = int(start_time * self.sr)
        duration_samples = int(duration_sec * self.sr)
        end_sample = min(start_sample + duration_samples, len(audio))
        
        if start_sample >= len(audio) or end_sample <= start_sample:
            return audio
        
        riser_length = end_sample - start_sample
        
        # Moderate white noise
        noise = np.random.normal(0, 0.1, riser_length)  # Not too loud
        
        # High-pass filter
        sos = signal.butter(4, 1500, 'highpass', fs=self.sr, output='sos')
        noise_filtered = signal.sosfilt(sos, noise)
        
        # Smooth amplitude envelope
        envelope = np.linspace(0, 1, riser_length) * 0.3  # Moderate volume
        noise_shaped = noise_filtered * envelope
        
        # Mix with original
        if audio.ndim == 1:
            audio[start_sample:end_sample] += noise_shaped
        else:
            for ch in range(audio.shape[1]):
                audio[start_sample:end_sample, ch] += noise_shaped
        
        return audio


if __name__ == "__main__":
    print("Enhanced DJ Effects System - BALANCED VERSION")
    print("=" * 60)
    print("Professional effects that enhance without overwhelming:")
    print("  ✓ Selective bass boost (only in appropriate sections)")
    print("  ✓ Beat drops (tension and release)")
    print("  ✓ Filter buildups (smooth transitions)")
    print("  ✓ White noise risers (build energy)")
    print("  ✓ Echo outs (classic DJ exit)")
    print("\nBalanced and natural sounding!")