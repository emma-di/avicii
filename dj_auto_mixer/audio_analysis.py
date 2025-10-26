"""
Audio Analysis Module using Librosa
Handles all technical audio feature extraction: BPM, energy, bass, structure detection
"""

import librosa
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class BeatGrid:
    """Structured beat grid information"""
    beat_times: np.ndarray  # Timestamp of each beat in seconds
    downbeat_times: np.ndarray  # Timestamp of downbeats (measure starts)
    bpm: float
    beat_frames: np.ndarray


@dataclass
class TrackStructure:
    """Song structure segments"""
    intro: Tuple[float, float]  # (start, end) in seconds
    outro: Tuple[float, float]
    breakdown_sections: List[Tuple[float, float]]  # Low energy sections good for transitions
    high_energy_sections: List[Tuple[float, float]]


class AudioAnalyzer:
    """Main audio analysis class using librosa"""
    
    def __init__(self, sr: int = 22050, hop_length: int = 512):
        """
        Initialize analyzer with standard DJ-friendly parameters
        
        Args:
            sr: Sample rate (22050 is sufficient for most DJ analysis)
            hop_length: FFT hop length for spectral analysis
        """
        self.sr = sr
        self.hop_length = hop_length
        
    def load_audio(self, file_path: str) -> np.ndarray:
        """Load audio file and convert to mono"""
        y, _ = librosa.load(file_path, sr=self.sr, mono=True)
        return y
    
    def analyze_bpm_and_beats(self, y: np.ndarray) -> BeatGrid:
        """
        Analyze tempo and extract beat grid
        
        DJ Context: Beat matching is the foundation of DJ mixing. We need precise
        beat locations to synchronize tracks. Downbeats (first beat of a measure)
        are crucial for phrase-aligned transitions.
        
        Returns:
            BeatGrid with beat times, downbeats, and BPM
        """
        # Extract tempo and beat frames
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=self.sr, hop_length=self.hop_length)
        
        # Convert frames to time
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sr, hop_length=self.hop_length)
        
        # Estimate downbeats (every 4th beat in 4/4 time, most common in DJ music)
        # More sophisticated: use librosa.beat.beat_track with aggregation
        if len(beat_times) >= 4:
            # Assume 4/4 time signature, align to strongest beats
            downbeat_indices = np.arange(0, len(beat_times), 4)
            downbeat_times = beat_times[downbeat_indices]
        else:
            downbeat_times = np.array([beat_times[0]]) if len(beat_times) > 0 else np.array([])
        
        return BeatGrid(
            beat_times=beat_times,
            downbeat_times=downbeat_times,
            bpm=float(tempo),
            beat_frames=beat_frames
        )
    
    def calculate_energy_profile(self, y: np.ndarray, smoothing_window: int = 43) -> np.ndarray:
        """
        Calculate time-varying energy profile
        
        DJ Context: Energy management is key to set flow. We track RMS energy
        to identify high/low energy sections and match energy levels during transitions.
        
        Args:
            y: Audio time series
            smoothing_window: Frames to smooth over (43 frames ≈ 1 second at default hop)
            
        Returns:
            Normalized energy curve over time
        """
        # RMS energy per frame
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        
        # Smooth to remove transient spikes
        rms_smooth = np.convolve(rms, np.ones(smoothing_window)/smoothing_window, mode='same')
        
        # Normalize to 0-1 range
        rms_normalized = (rms_smooth - np.min(rms_smooth)) / (np.max(rms_smooth) - np.min(rms_smooth) + 1e-8)
        
        return rms_normalized
    
    def analyze_bass_profile(self, y: np.ndarray, bass_cutoff: float = 250.0) -> Dict[str, Any]:
        """
        Analyze bass frequency content over time
        
        DJ Context: CRITICAL RULE - Never have two bass-heavy tracks at full volume
        simultaneously. This creates a muddy, unprofessional mix. We isolate bass
        frequencies (20-250 Hz) to manage the low-end during crossfades.
        
        Args:
            y: Audio time series
            bass_cutoff: Upper frequency for bass range in Hz
            
        Returns:
            Dictionary with bass energy profile and bass-heavy sections
        """
        # Compute Short-Time Fourier Transform
        D = librosa.stft(y, hop_length=self.hop_length)
        S = np.abs(D)
        
        # Frequency bins
        freqs = librosa.fft_frequencies(sr=self.sr)
        
        # Isolate bass frequencies (20-250 Hz)
        bass_bins = np.where((freqs >= 20) & (freqs <= bass_cutoff))[0]
        bass_energy = np.sum(S[bass_bins, :], axis=0)
        
        # Normalize
        bass_normalized = bass_energy / (np.max(bass_energy) + 1e-8)
        
        # Identify bass-heavy sections (above 70% threshold)
        bass_heavy_threshold = 0.7
        bass_heavy_frames = bass_normalized > bass_heavy_threshold
        
        return {
            'bass_profile': bass_normalized,
            'bass_heavy_frames': bass_heavy_frames,
            'bass_energy_mean': float(np.mean(bass_normalized))
        }
    
    def analyze_loudness(self, y: np.ndarray) -> Dict[str, float]:
        """
        Analyze loudness characteristics
        
        DJ Context: Consistent loudness prevents jarring volume changes between tracks.
        We measure peak and RMS levels to normalize tracks and prevent clipping.
        
        Returns:
            Dictionary with loudness metrics
        """
        # RMS (average) loudness
        rms_db = librosa.amplitude_to_db(np.array([np.sqrt(np.mean(y**2))]))
        
        # Peak level
        peak_db = librosa.amplitude_to_db(np.array([np.max(np.abs(y))]))
        
        # Crest factor (peak to RMS ratio) - indicates dynamic range
        crest_factor = peak_db[0] - rms_db[0]
        
        return {
            'rms_db': float(rms_db[0]),
            'peak_db': float(peak_db[0]),
            'crest_factor': float(crest_factor)
        }
    
    def detect_structure(self, y: np.ndarray, beat_times: np.ndarray) -> TrackStructure:
        """
        Detect song structure (intro, outro, breakdowns)
        
        DJ Context: Transitions work best during specific song sections:
        - Intros/Outros: Minimal elements, ideal for mixing
        - Breakdowns: Low energy sections, good for energy shifts
        - Avoid transitioning during vocal-heavy choruses
        
        Args:
            y: Audio time series
            beat_times: Beat timestamps from beat grid
            
        Returns:
            TrackStructure with identified sections
        """
        duration = librosa.get_duration(y=y, sr=self.sr)
        
        # Calculate energy profile for structure detection
        energy = self.calculate_energy_profile(y)
        energy_times = librosa.frames_to_time(np.arange(len(energy)), sr=self.sr, hop_length=self.hop_length)
        
        # Intro detection: First 10-20% of track, typically lower energy
        intro_duration = min(duration * 0.15, 32.0)  # Max 32 seconds
        intro = (0.0, intro_duration)
        
        # Outro detection: Last 10-20% of track
        outro_start = max(duration * 0.85, duration - 32.0)  # Max 32 seconds
        outro = (outro_start, duration)
        
        # Breakdown detection: Low energy sections in the middle
        breakdown_sections = []
        if len(energy) > 0:
            # Find sections where energy drops below 40% of max
            low_energy_threshold = 0.4
            low_energy_mask = energy < low_energy_threshold
            
            # Find contiguous low-energy regions
            in_breakdown = False
            breakdown_start = 0
            
            for i, is_low in enumerate(low_energy_mask):
                time = energy_times[i] if i < len(energy_times) else duration
                
                # Skip intro/outro regions
                if time < intro[1] or time > outro[0]:
                    continue
                    
                if is_low and not in_breakdown:
                    breakdown_start = time
                    in_breakdown = True
                elif not is_low and in_breakdown:
                    # Only keep breakdowns longer than 8 seconds
                    if time - breakdown_start > 8.0:
                        breakdown_sections.append((breakdown_start, time))
                    in_breakdown = False
        
        # High energy sections (for reference, inverse of breakdowns)
        high_energy_sections = []
        if len(energy) > 0:
            high_energy_threshold = 0.7
            high_energy_mask = energy > high_energy_threshold
            
            in_high_energy = False
            high_start = 0
            
            for i, is_high in enumerate(high_energy_mask):
                time = energy_times[i] if i < len(energy_times) else duration
                
                if time < intro[1] or time > outro[0]:
                    continue
                    
                if is_high and not in_high_energy:
                    high_start = time
                    in_high_energy = True
                elif not is_high and in_high_energy:
                    if time - high_start > 8.0:
                        high_energy_sections.append((high_start, time))
                    in_high_energy = False
        
        return TrackStructure(
            intro=intro,
            outro=outro,
            breakdown_sections=breakdown_sections,
            high_energy_sections=high_energy_sections
        )
    
    def find_optimal_transition_beats(self, beat_grid: BeatGrid, 
                                     section_start: float, 
                                     section_end: float,
                                     phrase_length: int = 16) -> List[float]:
        """
        Find optimal beat positions for transitions within a section
        
        DJ Context: Professional transitions happen on phrase boundaries (typically
        every 8 or 16 beats in electronic music). This maintains musical coherence.
        
        Args:
            beat_grid: Beat grid from analyze_bpm_and_beats
            section_start: Start time of section to search
            section_end: End time of section
            phrase_length: Beats per phrase (8 or 16 typical)
            
        Returns:
            List of optimal transition timestamps
        """
        # Find beats within the section
        section_beats = beat_grid.beat_times[
            (beat_grid.beat_times >= section_start) & 
            (beat_grid.beat_times <= section_end)
        ]
        
        if len(section_beats) == 0:
            return []
        
        # Find beats that align with phrase boundaries
        # Count from the first downbeat
        first_downbeat_idx = np.searchsorted(beat_grid.beat_times, section_beats[0])
        
        optimal_beats = []
        for i, beat_time in enumerate(section_beats):
            # Check if this beat is on a phrase boundary
            beats_from_start = (first_downbeat_idx + i) % phrase_length
            if beats_from_start == 0:
                optimal_beats.append(float(beat_time))
        
        return optimal_beats
    
    def calculate_tempo_compatibility(self, bpm1: float, bpm2: float) -> Dict[str, Any]:
        """
        Calculate BPM compatibility between two tracks
        
        DJ Context: Tracks need compatible tempos for beat matching. Options:
        1. Direct match: Within ±6% (typical pitch adjustment range)
        2. Harmonic match: 2:1 or 1:2 ratio (double-time/half-time)
        3. Nudge match: Within ±10 BPM can be manually adjusted
        
        Returns:
            Compatibility score and recommended adjustment
        """
        bpm_ratio = bpm2 / bpm1
        
        # Direct match score (within ±6%)
        direct_diff = abs(bpm1 - bpm2) / bpm1
        direct_compatible = direct_diff < 0.06
        
        # Harmonic match (2:1 or 1:2 ratio)
        harmonic_2x = abs(bpm_ratio - 2.0) < 0.06
        harmonic_half = abs(bpm_ratio - 0.5) < 0.06
        harmonic_compatible = harmonic_2x or harmonic_half
        
        # Calculate overall compatibility
        if direct_compatible:
            score = 100.0 * (1.0 - direct_diff / 0.06)
            method = "direct"
            adjustment = bpm2 / bpm1  # Pitch adjustment ratio
        elif harmonic_compatible:
            score = 90.0
            method = "harmonic_2x" if harmonic_2x else "harmonic_half"
            adjustment = 2.0 if harmonic_2x else 0.5
        elif abs(bpm1 - bpm2) < 10:
            score = 70.0
            method = "manual_nudge"
            adjustment = 1.0
        else:
            score = max(0, 50.0 - abs(bpm1 - bpm2))
            method = "incompatible"
            adjustment = 1.0
        
        return {
            'score': score,
            'method': method,
            'adjustment': adjustment,
            'bpm_diff': abs(bpm1 - bpm2)
        }


def analyze_key_compatibility(key1: str, key2: str) -> float:
    """
    Calculate harmonic key compatibility
    
    DJ Context: Mixing in compatible keys sounds more harmonious. The Camelot
    wheel is a DJ tool for key matching:
    - Same key: 100% compatible
    - Adjacent keys (+1/-1): 80% compatible  
    - Relative major/minor: 90% compatible
    - Incompatible keys can clash
    
    Args:
        key1, key2: Musical keys in standard notation (e.g., "A minor", "C major")
        
    Returns:
        Compatibility score 0-100
    """
    # Camelot wheel mapping (simplified)
    camelot_wheel = {
        'C major': 1, 'A minor': 1,
        'G major': 2, 'E minor': 2,
        'D major': 3, 'B minor': 3,
        'A major': 4, 'F# minor': 4,
        'E major': 5, 'C# minor': 5,
        'B major': 6, 'G# minor': 6,
        'F# major': 7, 'D# minor': 7,
        'Db major': 8, 'Bb minor': 8,
        'Ab major': 9, 'F minor': 9,
        'Eb major': 10, 'C minor': 10,
        'Bb major': 11, 'G minor': 11,
        'F major': 12, 'D minor': 12,
    }
    
    # Normalize key names
    key1 = key1.strip()
    key2 = key2.strip()
    
    if key1 not in camelot_wheel or key2 not in camelot_wheel:
        return 50.0  # Unknown key, neutral score
    
    pos1 = camelot_wheel[key1]
    pos2 = camelot_wheel[key2]
    
    # Same key
    if key1 == key2:
        return 100.0
    
    # Same position (relative major/minor)
    if pos1 == pos2:
        return 90.0
    
    # Adjacent keys (±1 on wheel)
    diff = min(abs(pos1 - pos2), 12 - abs(pos1 - pos2))
    if diff == 1:
        return 80.0
    elif diff == 2:
        return 60.0
    elif diff <= 3:
        return 40.0
    else:
        return 20.0


# Example usage and testing
if __name__ == "__main__":
    analyzer = AudioAnalyzer()
    
    # Example: Analyze a track
    print("Audio Analysis Module - Test")
    print("="*50)
    print("This module provides all librosa-based audio analysis")
    print("including BPM, energy, bass, and structure detection.")
    print("\nKey DJ Concepts Implemented:")
    print("✓ Beat matching and phrase alignment")
    print("✓ Energy profile tracking")
    print("✓ Bass frequency isolation (20-250 Hz)")
    print("✓ Song structure detection (intro/outro/breakdowns)")
    print("✓ Tempo and key compatibility scoring")