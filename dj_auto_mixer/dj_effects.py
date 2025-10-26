"""
PARTY DJ Effects System
MASSIVE effects with beat drops, bass boosts, stutters, and creative manipulation
Makes it OBVIOUS a DJ mixed this!
"""

import numpy as np
from scipy import signal
from typing import Dict, Tuple, Optional, List
import librosa


class DJEffects:
    """
    Professional DJ effects processor with STRONG, noticeable effects
    
    Effects include:
    - Aggressive filter buildups/sweeps
    - White noise risers
    - Echo outs with feedback
    - Reverb throws
    - Pitch shifts/drops
    - Bitcrusher
    - Synthetic air horns, sirens
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
        
        # EFFECT 1: BASS BOOST throughout (moderate - not too much!)
        audio = self.apply_bass_boost(audio, boost_db=4.0)  # +4dB instead of +6dB
        print(f"        ✓ Bass boost (+4dB)")
        
        # EFFECT 2: Beat drop (only for medium/high energy, in the middle)
        if energy_level in ['medium', 'high'] and duration > 30 and len(beat_times) > 8:
            drop_time = duration * 0.5  # Middle of track
            drop_beat = self._find_nearest_beat(drop_time, beat_times)
            audio = self.apply_beat_drop(audio, drop_beat, duration_beats=4, bpm=bpm)
            print(f"        ✓ Beat drop at {drop_beat:.1f}s")
        
        # EFFECT 3: Filter buildup before exit (essential for transitions)
        if duration > 15:
            buildup_start = duration - 14
            # Less aggressive than before
            audio = self.apply_filter_buildup(
                audio, int(buildup_start * self.sr), 
                duration_sec=10.0, max_freq=2500  # Was 4000Hz, now 2500Hz
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
    
    def apply_bass_boost(self, audio: np.ndarray, boost_db: float = 6.0) -> np.ndarray:
        """
        Boost the bass frequencies - makes it THUMP!
        """
        audio = audio.copy()
        
        # Low shelf filter to boost bass (20-250Hz)
        # Convert dB to linear gain
        gain = 10 ** (boost_db / 20)
        
        # Create low shelf filter
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
    
    def apply_stutter(self, audio: np.ndarray, stutter_time: float,
                     repeats: int = 4, bpm: float = 120) -> np.ndarray:
        """
        Stutter effect - repeat a small section multiple times
        Creates rhythmic tension
        """
        audio = audio.copy()
        
        # Stutter duration: 1/16th note
        beat_duration = 60.0 / bpm
        stutter_duration = beat_duration / 4  # 16th note
        stutter_samples = int(stutter_duration * self.sr)
        
        stutter_sample = int(stutter_time * self.sr)
        
        if stutter_sample >= len(audio) or stutter_sample + stutter_samples >= len(audio):
            return audio
        
        # Get the segment to repeat
        stutter_segment = audio[stutter_sample:stutter_sample + stutter_samples].copy()
        
        # Repeat it multiple times
        for i in range(repeats):
            pos = stutter_sample + (i * stutter_samples)
            end_pos = min(pos + stutter_samples, len(audio))
            actual_length = end_pos - pos
            
            if actual_length > 0:
                audio[pos:end_pos] = stutter_segment[:actual_length]
        
        return audio
    
    def apply_bass_cut_restore(self, audio: np.ndarray, cut_time: float,
                               cut_beats: int = 2, bpm: float = 120) -> np.ndarray:
        """
        Cut the bass out then restore it - creates drama!
        """
        audio = audio.copy()
        
        beat_duration = 60.0 / bpm
        cut_duration = cut_beats * beat_duration
        
        cut_sample = int(cut_time * self.sr)
        cut_length = int(cut_duration * self.sr)
        end_sample = min(cut_sample + cut_length, len(audio))
        
        if cut_sample >= len(audio):
            return audio
        
        # Remove bass from section
        sos = signal.butter(6, 250, 'highpass', fs=self.sr, output='sos')
        
        if audio.ndim == 1:
            audio[cut_sample:end_sample] = signal.sosfilt(sos, audio[cut_sample:end_sample])
            audio[cut_sample:end_sample] *= 0.6  # Reduce volume too
        else:
            for ch in range(audio.shape[1]):
                audio[cut_sample:end_sample, ch] = signal.sosfilt(sos, audio[cut_sample:end_sample, ch])
                audio[cut_sample:end_sample, ch] *= 0.6
        
        # Quick fade back in at the end
        fade_length = int(0.05 * self.sr)  # 50ms
        if end_sample - fade_length > cut_sample:
            fade = np.linspace(0.6, 1.0, fade_length)
            if audio.ndim == 1:
                audio[end_sample - fade_length:end_sample] *= fade
            else:
                audio[end_sample - fade_length:end_sample] *= fade[:, np.newaxis]
        
        return audio
    
    def add_reverb_throw_on_beat(self, audio: np.ndarray, throw_time: float,
                                 intensity: float = 2.0) -> np.ndarray:
        """
        Throw reverb on a specific beat - sounds like throwing into space
        """
        return self.add_reverb_throw(audio, throw_time, intensity)
    
    def add_vinyl_scratch(self, audio: np.ndarray, scratch_time: float) -> np.ndarray:
        """
        Vinyl scratch effect - classic DJ move!
        """
        audio = audio.copy()
        
        scratch_sample = int(scratch_time * self.sr)
        scratch_duration = int(0.2 * self.sr)  # 200ms scratch
        
        if scratch_sample >= len(audio) or scratch_sample + scratch_duration >= len(audio):
            return audio
        
        # Get segment
        segment = audio[scratch_sample:scratch_sample + scratch_duration].copy()
        
        # Create scratch pattern (back and forth)
        # Split into 4 parts: forward, back, forward, back
        quarter = len(segment) // 4
        
        # Reverse middle sections
        if audio.ndim == 1:
            segment[quarter:2*quarter] = segment[quarter:2*quarter][::-1]
            segment[3*quarter:] = segment[3*quarter:][::-1]
        else:
            segment[quarter:2*quarter] = segment[quarter:2*quarter][::-1, :]
            segment[3*quarter:] = segment[3*quarter:][::-1, :]
        
        # Apply pitch shifts for more scratch-like sound
        segment *= 0.8  # Reduce volume slightly
        
        audio[scratch_sample:scratch_sample + scratch_duration] = segment
        
        return audio
    
    def apply_flanger(self, audio: np.ndarray, start_sample: int,
                     duration_sec: float = 4.0) -> np.ndarray:
        """
        Flanger effect - sweeping comb filter for movement
        """
        audio = audio.copy()
        
        duration_samples = int(duration_sec * self.sr)
        end_sample = min(start_sample + duration_samples, len(audio))
        
        if start_sample >= len(audio):
            return audio
        
        segment = audio[start_sample:end_sample].copy()
        segment_length = len(segment)
        
        # Flanger parameters
        min_delay = 0.001  # 1ms
        max_delay = 0.005  # 5ms
        lfo_freq = 0.5  # Hz
        
        # Create LFO (Low Frequency Oscillator)
        t = np.arange(segment_length) / self.sr
        lfo = (np.sin(2 * np.pi * lfo_freq * t) + 1) / 2  # 0 to 1
        delay_samples = (min_delay + (max_delay - min_delay) * lfo) * self.sr
        
        # Apply varying delay
        output = np.zeros_like(segment)
        
        for i in range(segment_length):
            delay = int(delay_samples[i])
            if i >= delay:
                if audio.ndim == 1:
                    output[i] = segment[i] + segment[i - delay] * 0.7
                else:
                    output[i] = segment[i] + segment[i - delay] * 0.7
        
        audio[start_sample:end_sample] = output * 0.8  # Reduce overall volume to prevent clipping
        
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
        Strong echo-out with MORE repeats and LONGER tail
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
    
    def add_reverb_throw(self, audio: np.ndarray, throw_time: float, 
                        intensity: float = 1.5) -> np.ndarray:
        """
        DRAMATIC reverb throw - sounds like throwing into a huge space
        """
        audio = audio.copy()
        throw_sample = int(throw_time * self.sr)
        
        if throw_sample >= len(audio):
            return audio
        
        # Create BIG reverb
        delay_times = [0.029, 0.037, 0.041, 0.043, 0.053, 0.059]  # More delays
        reverb_length = int(self.sr * 3.0)  # 3 second tail!
        
        end_sample = min(throw_sample + reverb_length, len(audio))
        segment = audio[throw_sample:end_sample].copy()
        
        reverb = np.zeros_like(segment)
        for delay_time in delay_times:
            delay_samples = int(delay_time * self.sr)
            if delay_samples < len(segment):
                delayed = np.roll(segment, delay_samples, axis=0)
                reverb += delayed * 0.7  # Strong reverb
        
        # Mix with strong intensity
        audio[throw_sample:end_sample] = (
            segment * (1 - intensity * 0.3) + reverb * intensity * 0.5
        )
        
        return audio
    
    def add_pitch_drop(self, audio: np.ndarray, drop_time: float) -> np.ndarray:
        """
        Pitch drop effect - pitch down then back up
        """
        audio = audio.copy()
        drop_sample = int(drop_time * self.sr)
        drop_duration = int(self.sr * 2.0)  # 2 second effect
        
        end_sample = min(drop_sample + drop_duration, len(audio))
        
        if drop_sample >= len(audio) or end_sample <= drop_sample:
            return audio
        
        segment_length = end_sample - drop_sample
        
        # Create pitch shift curve (down then up)
        pitch_curve = np.zeros(segment_length)
        mid_point = segment_length // 2
        
        # Down in first half
        pitch_curve[:mid_point] = np.linspace(0, -4, mid_point)  # Down 4 semitones
        # Up in second half
        pitch_curve[mid_point:] = np.linspace(-4, 0, segment_length - mid_point)
        
        # Apply pitch shift
        segment = audio[drop_sample:end_sample].copy()
        
        if segment.ndim == 1:
            segment_mono = segment
        else:
            segment_mono = segment.mean(axis=1)
        
        # Simple pitch shift using resampling
        shifted = librosa.effects.pitch_shift(
            segment_mono, sr=self.sr, n_steps=-2
        )
        
        if audio.ndim == 1:
            audio[drop_sample:drop_sample + len(shifted)] = shifted[:len(segment)]
        else:
            for ch in range(audio.shape[1]):
                audio[drop_sample:drop_sample + len(shifted), ch] = shifted[:len(segment)]
        
        return audio
    
    def add_air_horn(self, audio: np.ndarray, horn_time: float) -> np.ndarray:
        """
        Synthetic air horn sound (classic DJ effect)
        """
        audio = audio.copy()
        horn_sample = int(horn_time * self.sr)
        horn_duration = int(self.sr * 0.8)  # 800ms
        
        if horn_sample >= len(audio):
            return audio
        
        # Generate synthetic air horn (descending tone)
        t = np.linspace(0, 0.8, horn_duration)
        
        # Frequency sweep (descending)
        freq_start = 400  # Hz
        freq_end = 200    # Hz
        freq = np.linspace(freq_start, freq_end, horn_duration)
        
        # Generate tone with harmonics
        horn = np.zeros(horn_duration)
        for harmonic in [1, 2, 3]:
            horn += np.sin(2 * np.pi * freq * harmonic * t) / harmonic
        
        # Envelope
        envelope = np.concatenate([
            np.linspace(0, 1, horn_duration // 4),  # Attack
            np.ones(horn_duration // 2),             # Sustain
            np.linspace(1, 0, horn_duration // 4)    # Release
        ])
        
        horn *= envelope * 0.3  # Amplitude
        
        # Add to audio
        end_sample = min(horn_sample + horn_duration, len(audio))
        actual_length = end_sample - horn_sample
        
        if audio.ndim == 1:
            audio[horn_sample:end_sample] += horn[:actual_length]
        else:
            for ch in range(audio.shape[1]):
                audio[horn_sample:end_sample, ch] += horn[:actual_length]
        
        return audio


if __name__ == "__main__":
    print("Enhanced DJ Effects System")
    print("=" * 60)
    print("STRONG, noticeable effects:")
    print("  ✓ Aggressive filter buildups")
    print("  ✓ Intense white noise risers")
    print("  ✓ Echo outs with long tail")
    print("  ✓ Dramatic reverb throws")
    print("  ✓ Pitch drops")
    print("  ✓ Synthetic air horns")
    print("\nThese effects are MUCH more noticeable!")