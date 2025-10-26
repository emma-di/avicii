"""
Transition Generator Module
Generates detailed mixing instructions for crossfades between tracks
Handles EQ automation, volume curves, and transition timing
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class TransitionType(Enum):
    """Types of DJ transitions"""
    BEAT_MATCHED = "beat_matched"  # Standard EQ crossfade with beat sync
    ENERGY_BUILD = "energy_build"  # Build energy during transition
    DROP_SWAP = "drop_swap"  # Swap at a drop/peak moment
    BREAKDOWN_MIX = "breakdown_mix"  # Mix during low energy breakdown
    HARD_CUT = "hard_cut"  # Instant cut (for dramatic effect)
    ECHO_OUT = "echo_out"  # Echo/reverb fade out
    SPINBACK = "spinback"  # Spinback effect (fast tempo down)


@dataclass
class TransitionPoint:
    """Optimal transition point between two tracks"""
    track_a_out_time: float  # When to start fading out track A (seconds)
    track_b_in_time: float   # When to start fading in track B (seconds)
    crossfade_duration: float  # Duration of overlap (seconds)
    transition_type: TransitionType
    confidence: float  # How good this transition point is (0-100)


@dataclass
class EQCurve:
    """EQ automation curve"""
    timestamps: List[float]  # Time points
    bass: List[float]  # Bass level 0-1
    mid: List[float]   # Mid level 0-1
    high: List[float]  # High level 0-1


@dataclass
class MixingInstructions:
    """Complete mixing instructions for a transition"""
    transition_point: TransitionPoint
    track_a_eq: EQCurve
    track_b_eq: EQCurve
    track_a_volume: List[Tuple[float, float]]  # [(time, volume), ...]
    track_b_volume: List[Tuple[float, float]]
    effects: List[str]
    notes: str  # Human-readable mixing tips


class TransitionGenerator:
    """
    Generates mixing instructions based on track analysis
    
    DJ Context: A transition is more than just crossfading volume.
    Professional DJs manipulate EQ, timing, and effects to create
    seamless blends that maintain energy and clarity.
    """
    
    def __init__(self):
        """Initialize with standard DJ mixing parameters"""
        self.min_crossfade = 4  # Minimum crossfade duration in beats
        self.max_crossfade = 32  # Maximum crossfade duration in beats
        self.standard_crossfade = 16  # Standard crossfade for matched tracks
    
    def find_transition_point(self,
                             track_a_analysis: Dict,
                             track_b_analysis: Dict,
                             compatibility_score: float) -> TransitionPoint:
        """
        Find optimal transition point between tracks
        
        DJ Context: Transition points matter. Mixing out of a vocal section
        into another vocal section = bad. Mixing during instrumental breaks,
        intros/outros, or breakdowns = good.
        
        Args:
            track_a_analysis: Complete analysis of outgoing track
            track_b_analysis: Complete analysis of incoming track
            compatibility_score: Overall compatibility (affects transition length)
            
        Returns:
            Optimal transition point with timing and type
        """
        structure_a = track_a_analysis['structure']
        structure_b = track_b_analysis['structure']
        vocals_a = track_a_analysis.get('vocals', {})
        vocals_b = track_b_analysis.get('vocals', {})
        energy_a = track_a_analysis['energy_profile']
        energy_b = track_b_analysis['energy_profile']
        bpm_a = track_a_analysis['bpm']
        beat_grid_a = track_a_analysis['beat_grid']
        
        # Calculate track duration from beat grid or structure
        if hasattr(beat_grid_a, 'beat_times') and len(beat_grid_a.beat_times) > 0:
            track_a_duration = float(beat_grid_a.beat_times[-1])
        else:
            track_a_duration = structure_a.outro[1]
        
        # Better transition out point strategy:
        # Don't start mixing out too early - aim for 70-80% through the track
        outro_start = structure_a.outro[0]
        ideal_out_point = track_a_duration * 0.75  # 75% through the track
        
        # If outro detection put us before 65% of track, use better timing
        if outro_start < track_a_duration * 0.65:
            track_a_out = ideal_out_point
        elif outro_start < track_a_duration * 0.8:
            # Use midpoint between ideal and detected outro
            track_a_out = (ideal_out_point + outro_start) / 2
        else:
            # Outro is late in track, use it
            track_a_out = outro_start
        
        # Prefer mixing in during intro, but skip the very beginning
        # Start at least 4-8 seconds into the track to avoid intro silence
        track_b_in = max(structure_b.intro[0], 4.0)
        if track_b_in < 4.0:
            track_b_in = 6.0  # Default to 6 seconds in
        
        # Determine transition type and duration based on characteristics
        transition_type = self._determine_transition_type(
            track_a_analysis, track_b_analysis, compatibility_score
        )
        
        # Calculate crossfade duration based on energy and compatibility
        crossfade_beats = self._calculate_crossfade_duration(
            track_a_analysis, track_b_analysis, compatibility_score
        )
        
        # Convert beats to seconds using track A's BPM
        bpm_a = track_a_analysis['bpm']
        beat_duration = 60.0 / bpm_a
        crossfade_duration = crossfade_beats * beat_duration
        
        # Adjust out point if needed to accommodate crossfade
        min_outro_duration = crossfade_duration + 4 * beat_duration  # Need some buffer
        if structure_a.outro[1] - track_a_out < min_outro_duration:
            # Start transition earlier
            track_a_out = max(structure_a.outro[1] - min_outro_duration, 
                            structure_a.outro[0])
        
        # Check for vocal clashing and adjust if needed
        if self._would_vocals_clash(vocals_a, vocals_b):
            # Use longer crossfade and ensure we're in instrumental sections
            crossfade_duration *= 1.5
            # Try to find breakdown in track A
            if structure_a.breakdown_sections:
                track_a_out = structure_a.breakdown_sections[-1][0]
        
        confidence = self._calculate_transition_confidence(
            track_a_out, track_b_in, crossfade_duration,
            track_a_analysis, track_b_analysis
        )
        
        return TransitionPoint(
            track_a_out_time=track_a_out,
            track_b_in_time=track_b_in,
            crossfade_duration=crossfade_duration,
            transition_type=transition_type,
            confidence=confidence
        )
    
    def _determine_transition_type(self, track_a: Dict, track_b: Dict, 
                                   compatibility: float) -> TransitionType:
        """
        Determine appropriate transition type
        
        DJ Context: Different situations call for different techniques:
        - Similar tracks: Standard beat-matched crossfade
        - Energy increase: Build during transition
        - Energy decrease: Use breakdown or echo out
        - Poor compatibility: Hard cut or dramatic effect
        """
        energy_a = np.mean(track_a['energy_profile'])
        energy_b = np.mean(track_b['energy_profile'])
        energy_diff = energy_b - energy_a
        
        if compatibility > 80:
            if abs(energy_diff) < 0.1:
                return TransitionType.BEAT_MATCHED
            elif energy_diff > 0.2:
                return TransitionType.ENERGY_BUILD
            else:
                return TransitionType.BREAKDOWN_MIX
        elif compatibility > 60:
            return TransitionType.BEAT_MATCHED
        else:
            # Low compatibility, use more dramatic transition
            if energy_diff > 0.3:
                return TransitionType.DROP_SWAP
            else:
                return TransitionType.ECHO_OUT
    
    def _calculate_crossfade_duration(self, track_a: Dict, track_b: Dict,
                                     compatibility: float) -> int:
        """
        Calculate optimal crossfade duration in beats
        
        DJ Context:
        - High energy, similar tracks: Short crossfade (4-8 beats)
        - Different energy: Longer crossfade (16-32 beats)
        - Poor compatibility: Either very short (cut) or very long (gradual blend)
        """
        energy_a = np.mean(track_a['energy_profile'])
        energy_b = np.mean(track_b['energy_profile'])
        energy_diff = abs(energy_b - energy_a)
        
        # Base duration on compatibility
        if compatibility > 85:
            base_beats = 8  # Quick mix for very compatible tracks
        elif compatibility > 70:
            base_beats = 16  # Standard mix
        else:
            base_beats = 24  # Longer mix for less compatible tracks
        
        # Adjust for energy difference
        if energy_diff > 0.3:
            base_beats += 8  # More time for large energy shifts
        
        # Clamp to valid range
        return max(self.min_crossfade, min(base_beats, self.max_crossfade))
    
    def _would_vocals_clash(self, vocals_a: Dict, vocals_b: Dict) -> bool:
        """Check if vocals would clash during transition"""
        if not vocals_a or not vocals_b:
            return False
        
        # Clash if both have heavy or moderate vocals
        a_has_vocals = vocals_a.get('classification') in ['heavy_vocals', 'moderate_vocals']
        b_has_vocals = vocals_b.get('classification') in ['heavy_vocals', 'moderate_vocals']
        
        return a_has_vocals and b_has_vocals
    
    def _calculate_transition_confidence(self, out_time: float, in_time: float,
                                        duration: float, track_a: Dict, 
                                        track_b: Dict) -> float:
        """
        Calculate confidence in this transition point
        
        Returns score 0-100 based on:
        - Whether we're in good sections (intro/outro)
        - Beat alignment
        - Energy matching at transition point
        """
        confidence = 50.0  # Base confidence
        
        # Bonus for being in outro/intro
        if out_time >= track_a['structure'].outro[0]:
            confidence += 20
        if in_time <= track_b['structure'].intro[1]:
            confidence += 20
        
        # Bonus for phrase alignment (on downbeat)
        beat_grid_a = track_a['beat_grid']
        if len(beat_grid_a.downbeat_times) > 0:
            nearest_downbeat = beat_grid_a.downbeat_times[
                np.argmin(np.abs(beat_grid_a.downbeat_times - out_time))
            ]
            if abs(nearest_downbeat - out_time) < 0.1:  # Within 100ms
                confidence += 10
        
        return min(confidence, 100.0)
    
    def generate_eq_automation(self, transition_point: TransitionPoint,
                               track_a: Dict, track_b: Dict) -> Tuple[EQCurve, EQCurve]:
        """
        Generate EQ automation curves for both tracks
        
        DJ Context: THE GOLDEN RULE - Never have two bass-heavy tracks
        at full bass simultaneously. This is what separates amateur from
        professional mixing.
        
        Strategy:
        1. First half: Track A at full bass, Track B bass cut (high-pass filter)
        2. Midpoint: Crossover - both at reduced bass
        3. Second half: Track A bass out, Track B bass in
        """
        duration = transition_point.crossfade_duration
        num_points = 20  # Number of automation points
        
        # Time points throughout transition
        times = np.linspace(0, duration, num_points)
        
        # Bass content analysis
        bass_a = track_a['bass_profile']['bass_energy_mean']
        bass_b = track_b['bass_profile']['bass_energy_mean']
        
        # Both tracks have significant bass - need careful EQ management
        if bass_a > 0.5 and bass_b > 0.5:
            # Track A EQ: Bass fades out, mids/highs stay
            track_a_bass = self._create_fadeout_curve(times, duration, steepness=2.0)
            track_a_mid = np.ones_like(times)  # Keep mids constant
            track_a_high = np.ones_like(times)  # Keep highs constant
            
            # Track B EQ: Bass fades in (starting from high-pass), mids/highs normal
            track_b_bass = self._create_fadein_curve(times, duration, start_value=0.1, steepness=2.0)
            track_b_mid = np.ones_like(times)
            track_b_high = np.ones_like(times)
        
        # Only Track A has bass - can keep it longer
        elif bass_a > 0.5:
            track_a_bass = self._create_fadeout_curve(times, duration, steepness=1.5)
            track_a_mid = self._create_fadeout_curve(times, duration, steepness=1.0)
            track_a_high = self._create_fadeout_curve(times, duration, steepness=1.0)
            
            track_b_bass = self._create_fadein_curve(times, duration, start_value=0.3)
            track_b_mid = self._create_fadein_curve(times, duration)
            track_b_high = self._create_fadein_curve(times, duration)
        
        # Only Track B has bass - bring it in cleanly
        elif bass_b > 0.5:
            track_a_bass = self._create_fadeout_curve(times, duration)
            track_a_mid = self._create_fadeout_curve(times, duration)
            track_a_high = self._create_fadeout_curve(times, duration)
            
            track_b_bass = self._create_fadein_curve(times, duration, steepness=1.5)
            track_b_mid = self._create_fadein_curve(times, duration, steepness=1.0)
            track_b_high = self._create_fadein_curve(times, duration, steepness=1.0)
        
        # Neither has much bass - standard crossfade
        else:
            track_a_bass = self._create_fadeout_curve(times, duration)
            track_a_mid = self._create_fadeout_curve(times, duration)
            track_a_high = self._create_fadeout_curve(times, duration)
            
            track_b_bass = self._create_fadein_curve(times, duration)
            track_b_mid = self._create_fadein_curve(times, duration)
            track_b_high = self._create_fadein_curve(times, duration)
        
        eq_a = EQCurve(
            timestamps=times.tolist(),
            bass=track_a_bass.tolist(),
            mid=track_a_mid.tolist(),
            high=track_a_high.tolist()
        )
        
        eq_b = EQCurve(
            timestamps=times.tolist(),
            bass=track_b_bass.tolist(),
            mid=track_b_mid.tolist(),
            high=track_b_high.tolist()
        )
        
        return eq_a, eq_b
    
    def generate_volume_automation(self, transition_point: TransitionPoint,
                                   crossfade_type: str = 'equal_power') -> Tuple[List, List]:
        """
        Generate volume automation curves
        
        DJ Context: Equal power crossfade maintains consistent perceived loudness.
        Mathematical formula: sqrt(x) for one track, sqrt(1-x) for the other.
        This prevents a "dip" in the middle of the transition.
        
        Args:
            transition_point: Transition timing information
            crossfade_type: 'equal_power', 'linear', or 'logarithmic'
            
        Returns:
            (track_a_volume, track_b_volume) as lists of (time, volume) tuples
        """
        duration = transition_point.crossfade_duration
        num_points = 50  # Smooth curve
        times = np.linspace(0, duration, num_points)
        
        # Progress from 0 to 1
        progress = times / duration
        
        if crossfade_type == 'equal_power':
            # Equal power crossfade (most common in professional DJ mixing)
            volume_a = np.sqrt(1 - progress)
            volume_b = np.sqrt(progress)
        elif crossfade_type == 'linear':
            volume_a = 1 - progress
            volume_b = progress
        elif crossfade_type == 'logarithmic':
            # Logarithmic for more gradual start/end
            volume_a = np.power(1 - progress, 2)
            volume_b = np.power(progress, 2)
        else:
            volume_a = 1 - progress
            volume_b = progress
        
        # Convert to list of tuples
        track_a_vol = [(float(t), float(v)) for t, v in zip(times, volume_a)]
        track_b_vol = [(float(t), float(v)) for t, v in zip(times, volume_b)]
        
        return track_a_vol, track_b_vol
    
    def _create_fadeout_curve(self, times: np.ndarray, duration: float,
                             steepness: float = 1.0, end_value: float = 0.0) -> np.ndarray:
        """Create smooth fade-out curve"""
        progress = times / duration
        curve = np.power(1 - progress, steepness)
        return curve * (1 - end_value) + end_value
    
    def _create_fadein_curve(self, times: np.ndarray, duration: float,
                            steepness: float = 1.0, start_value: float = 0.0) -> np.ndarray:
        """Create smooth fade-in curve"""
        progress = times / duration
        curve = np.power(progress, steepness)
        return curve * (1 - start_value) + start_value
    
    def generate_complete_instructions(self, track_a: Dict, track_b: Dict,
                                      compatibility_score: float) -> MixingInstructions:
        """
        Generate complete mixing instructions
        
        Returns:
            Full mixing instructions including timing, EQ, volume, and notes
        """
        # Find transition point
        transition = self.find_transition_point(track_a, track_b, compatibility_score)
        
        # Generate EQ automation
        eq_a, eq_b = self.generate_eq_automation(transition, track_a, track_b)
        
        # Generate volume automation
        vol_a, vol_b = self.generate_volume_automation(transition)
        
        # Determine effects
        effects = self._suggest_effects(transition, track_a, track_b)
        
        # Generate human-readable notes
        notes = self._generate_mixing_notes(transition, track_a, track_b, compatibility_score)
        
        return MixingInstructions(
            transition_point=transition,
            track_a_eq=eq_a,
            track_b_eq=eq_b,
            track_a_volume=vol_a,
            track_b_volume=vol_b,
            effects=effects,
            notes=notes
        )
    
    def _suggest_effects(self, transition: TransitionPoint, 
                        track_a: Dict, track_b: Dict) -> List[str]:
        """Suggest DJ effects for transition"""
        effects = []
        
        if transition.transition_type == TransitionType.ECHO_OUT:
            effects.append("echo_fade_out")
            effects.append("reverb")
        elif transition.transition_type == TransitionType.DROP_SWAP:
            effects.append("high_pass_filter_sweep")
        elif transition.transition_type == TransitionType.ENERGY_BUILD:
            effects.append("low_pass_filter_open")
            effects.append("reverb")
        
        # Always use HPF on incoming track if both have bass
        bass_a = track_a['bass_profile']['bass_energy_mean']
        bass_b = track_b['bass_profile']['bass_energy_mean']
        if bass_a > 0.5 and bass_b > 0.5:
            effects.append("high_pass_filter_incoming")
        
        return effects
    
    def _generate_mixing_notes(self, transition: TransitionPoint,
                              track_a: Dict, track_b: Dict,
                              compatibility: float) -> str:
        """Generate human-readable mixing tips"""
        notes = []
        
        notes.append(f"Transition Type: {transition.transition_type.value}")
        notes.append(f"Crossfade Duration: {transition.crossfade_duration:.1f}s")
        notes.append(f"Compatibility Score: {compatibility:.0f}/100")
        
        # Vocal warnings
        if track_a.get('vocals', {}).get('classification') == 'heavy_vocals':
            notes.append("⚠️ Track A has heavy vocals - ensure transition starts during instrumental section")
        if track_b.get('vocals', {}).get('classification') == 'heavy_vocals':
            notes.append("⚠️ Track B has heavy vocals - bring in during intro/breakdown")
        
        # Bass warnings
        bass_a = track_a['bass_profile']['bass_energy_mean']
        bass_b = track_b['bass_profile']['bass_energy_mean']
        if bass_a > 0.5 and bass_b > 0.5:
            notes.append("🔊 Both tracks are bass-heavy - carefully manage low-end EQ")
            notes.append("   → Cut Track B bass first 50% of transition")
        
        # Tempo notes
        bpm_diff = abs(track_a['bpm'] - track_b['bpm'])
        if bpm_diff > 5:
            notes.append(f"⚡ BPM difference: {bpm_diff:.1f} - may need tempo adjustment")
        
        return "\n".join(notes)


# Example usage
if __name__ == "__main__":
    print("Transition Generator Module - Test")
    print("="*50)
    print("This module generates detailed mixing instructions")
    print("\nKey Features:")
    print("✓ Optimal transition point detection")
    print("✓ EQ automation (bass management)")
    print("✓ Volume curves (equal power crossfade)")
    print("✓ Effect suggestions")
    print("✓ Human-readable mixing notes")
    print("\nDJ Techniques Implemented:")
    print("• Never mix two bass-heavy tracks at full volume")
    print("• Avoid vocal clashing")
    print("• Match energy levels")
    print("• Use phrase boundaries (8/16 bars)")
    print("• Apply effects appropriately")