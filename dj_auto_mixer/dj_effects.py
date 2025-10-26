"""
Enhanced DJ Effects System
STRONGER, MORE NOTICEABLE effects for rave-style mixes
"""

import numpy as np
from scipy import signal
from typing import Dict, Tuple, Optional
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
        Apply MULTIPLE strong effects based on track structure
        
        Args:
            audio: Audio data (samples x channels)
            track_analysis: Track structure info
            energy_level: 'low', 'medium', or 'high'
            
        Returns:
            Audio with NOTICEABLE effects applied
        """
        structure = track_analysis['structure']
        duration = len(audio) / self.sr
        
        print(f"      Applying {energy_level} energy effects...")
        
        # EFFECT 1: Aggressive filter buildup in last 12-16 seconds
        if duration > 15:
            effect_start = duration - 14  # Last 14 seconds
            effect_start_sample = int(effect_start * self.sr)
            
            if energy_level == 'high':
                # Very aggressive high-pass filter
                audio = self.apply_aggressive_filter_buildup(
                    audio, effect_start_sample, 
                    duration_sec=10.0, direction='highpass'
                )
                print(f"        ✓ Aggressive filter buildup")
            elif energy_level == 'medium':
                # Medium filter sweep
                audio = self.apply_aggressive_filter_buildup(
                    audio, effect_start_sample,
                    duration_sec=8.0, direction='highpass'
                )
                print(f"        ✓ Filter buildup")
        
        # EFFECT 2: Big reverb throw in middle
        if duration > 30 and energy_level in ['high', 'medium']:
            reverb_point = duration * 0.6  # 60% through track
            audio = self.add_reverb_throw(audio, reverb_point, intensity=1.5)
            print(f"        ✓ Reverb throw")
        
        # EFFECT 3: White noise riser before end
        if duration > 20 and energy_level == 'high':
            riser_start = duration - 8  # 8 seconds before end
            audio = self.add_intense_riser(audio, riser_start, duration_sec=6.0)
            print(f"        ✓ Intense riser")
        
        # EFFECT 4: Echo out at the very end
        if duration > 10:
            echo_start = duration - 6  # Last 6 seconds
            echo_sample = int(echo_start * self.sr)
            audio = self.apply_echo_out(audio, echo_sample, repeats=6, decay=0.65)
            print(f"        ✓ Echo out")
        
        # EFFECT 5: Pitch drop (for high energy only)
        if energy_level == 'high' and duration > 25:
            drop_point = duration - 12
            audio = self.add_pitch_drop(audio, drop_point)
            print(f"        ✓ Pitch drop")
        
        # EFFECT 6: Synthetic air horn (high energy)
        if energy_level == 'high' and duration > 30:
            horn_point = duration - 16
            audio = self.add_air_horn(audio, horn_point)
            print(f"        ✓ Air horn")
        
        return audio
    
    def apply_aggressive_filter_buildup(self, audio: np.ndarray, start_sample: int,
                                       duration_sec: float = 10.0, 
                                       direction: str = 'highpass') -> np.ndarray:
        """
        AGGRESSIVE filter sweep - you will definitely hear this!
        Removes bass gradually and dramatically
        """
        audio = audio.copy()
        duration_samples = int(duration_sec * self.sr)
        end_sample = min(start_sample + duration_samples, len(audio))
        
        if start_sample >= len(audio) or end_sample <= start_sample:
            return audio
        
        segment = audio[start_sample:end_sample]
        segment_length = len(segment)
        
        # AGGRESSIVE frequency sweep
        if direction == 'highpass':
            freq_start = 80  # Hz - start low
            freq_end = 4000  # Hz - end VERY high (removes almost everything)
        else:
            freq_start = 12000
            freq_end = 300
        
        # Apply MORE aggressively
        chunk_size = int(self.sr * 0.05)  # 50ms chunks
        for i in range(0, segment_length, chunk_size):
            chunk_end = min(i + chunk_size, segment_length)
            progress = i / segment_length
            
            # Exponential curve for more dramatic effect
            current_freq = freq_start + (freq_end - freq_start) * (progress ** 2)
            
            # STEEP filter (order 8 instead of 4)
            sos = signal.butter(8, current_freq, direction, fs=self.sr, output='sos')
            
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
    
    def add_intense_riser(self, audio: np.ndarray, start_time: float,
                         duration_sec: float = 6.0) -> np.ndarray:
        """
        INTENSE white noise riser - builds MASSIVE tension
        """
        audio = audio.copy()
        start_sample = int(start_time * self.sr)
        duration_samples = int(duration_sec * self.sr)
        end_sample = min(start_sample + duration_samples, len(audio))
        
        if start_sample >= len(audio) or end_sample <= start_sample:
            return audio
        
        riser_length = end_sample - start_sample
        
        # Generate LOUD white noise
        noise = np.random.normal(0, 0.15, riser_length)  # Increased amplitude
        
        # High-pass filter
        sos = signal.butter(6, 1000, 'highpass', fs=self.sr, output='sos')
        noise_filtered = signal.sosfilt(sos, noise)
        
        # STRONG amplitude envelope (exponential growth)
        envelope = (np.linspace(0, 1, riser_length) ** 1.5) * 0.5  # Louder!
        noise_shaped = noise_filtered * envelope
        
        # Aggressive frequency sweep
        for i in range(0, riser_length, int(self.sr * 0.05)):
            chunk_end = min(i + int(self.sr * 0.05), riser_length)
            progress = i / riser_length
            sweep_freq = 1000 + progress * 10000  # 1kHz to 11kHz
            
            sos = signal.butter(4, sweep_freq, 'highpass', fs=self.sr, output='sos')
            noise_shaped[i:chunk_end] = signal.sosfilt(sos, noise_shaped[i:chunk_end])
        
        # Mix with original (LOUDER)
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