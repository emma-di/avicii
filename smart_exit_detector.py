"""
Vibe-Based Exit Detector
Determines optimal exit time based on song's ACTUAL vibe, not fixed duration
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ExitPoint:
    time: float
    reason: str
    score: float
    section: str


class VibeBasedExitDetector:
    """
    Determines exit time based on song's actual vibe and energy
    
    Philosophy:
    - High energy bangers = SHORT (30-60s) - quick hit, keep energy
    - Medium energy = MEDIUM (60-90s) - normal DJ length
    - Chill/low energy = LONGER (90-150s) - let it breathe
    - Always respect song structure (don't cut mid-chorus)
    """
    
    def __init__(self):
        pass
    
    def get_best_exit_point(self, track: Dict, preference: str = 'vibe_based') -> ExitPoint:
        """
        Find optimal exit point based on VIBE, not fixed time
        
        Args:
            track: Track analysis dict
            preference: 'vibe_based' (dynamic) or 'early'/'balanced'/'late' (fixed)
            
        Returns:
            ExitPoint with time and reason
        """
        energy_profile = track.get('energy_profile', [])
        structure = track.get('structure', None)
        bpm = track.get('bpm', 120)
        
        # Calculate average energy to determine vibe
        avg_energy = np.mean(energy_profile) if len(energy_profile) > 0 else 0.5
        energy_variance = np.std(energy_profile) if len(energy_profile) > 1 else 0.1
        
        # DETERMINE SONG VIBE
        if avg_energy > 0.7:
            vibe = 'high_energy'  # Banger - keep it short!
            target_min = 30.0
            target_max = 60.0
            print(f"      Detected: HIGH ENERGY banger")
        elif avg_energy > 0.45:
            vibe = 'medium_energy'  # Normal energy
            target_min = 60.0
            target_max = 90.0
            print(f"      Detected: MEDIUM ENERGY track")
        else:
            vibe = 'low_energy'  # Chill - let it play longer
            target_min = 90.0
            target_max = 150.0
            print(f"      Detected: CHILL/LOW ENERGY track")
        
        # Override with user preference if specified
        if preference == 'early':
            target_min = 30.0
            target_max = 60.0
        elif preference == 'balanced':
            target_min = 60.0
            target_max = 90.0
        elif preference == 'late':
            target_min = 90.0
            target_max = 150.0
        
        # Find exit points within target range
        exit_candidates = self._find_exit_candidates(
            track, target_min, target_max
        )
        
        if not exit_candidates:
            # Fallback: use middle of target range
            fallback_time = (target_min + target_max) / 2
            return ExitPoint(
                time=fallback_time,
                reason=f"Fallback exit ({vibe})",
                score=50.0,
                section='unknown'
            )
        
        # Score candidates and pick best
        best_exit = max(exit_candidates, key=lambda x: x.score)
        
        print(f"      Vibe: {vibe}, Exit: {best_exit.time:.1f}s")
        
        return best_exit
    
    def _find_exit_candidates(self, track: Dict, min_time: float, 
                             max_time: float) -> List[ExitPoint]:
        """
        Find all possible exit points within time range
        """
        candidates = []
        
        energy_profile = track.get('energy_profile', [])
        structure = track.get('structure', None)
        beat_grid = track.get('beat_grid', [])
        
        if len(energy_profile) == 0:
            return candidates
        
        # Convert time range to indices
        total_duration = len(energy_profile) / 10  # Assuming 10 Hz energy profile
        min_idx = int(min_time * 10)
        max_idx = int(max_time * 10)
        max_idx = min(max_idx, len(energy_profile) - 1)
        
        if min_idx >= len(energy_profile):
            return candidates
        
        # Look for good exit points
        
        # 1. ENERGY DIPS (best exit points)
        for i in range(min_idx, max_idx):
            if i > 10 and i < len(energy_profile) - 10:
                # Check if this is a local energy minimum
                window_before = energy_profile[max(0, i-10):i]
                window_after = energy_profile[i:min(len(energy_profile), i+10)]
                current = energy_profile[i]
                
                if len(window_before) > 0 and len(window_after) > 0:
                    avg_before = np.mean(window_before)
                    avg_after = np.mean(window_after)
                    
                    # Good exit if energy is lower than surroundings
                    if current < avg_before * 0.8 and current < avg_after * 0.8:
                        time_sec = float(i / 10)
                        score = 90.0 - abs(current - 0.3) * 50  # Prefer mid-low energy
                        
                        candidates.append(ExitPoint(
                            time=time_sec,
                            reason="Energy dip (perfect exit)",
                            score=score,
                            section='low_energy'
                        ))
        
        # 2. REPETITIVE SECTIONS (good to exit)
        if len(energy_profile) > 50:
            for i in range(min_idx, max_idx, 10):
                if i + 50 < len(energy_profile):
                    segment = energy_profile[i:i+50]
                    
                    # Check for repetition (low variance = repetitive)
                    variance = np.var(segment)
                    
                    if variance < 0.01:  # Very repetitive
                        time_sec = float(i / 10)
                        score = 75.0
                        
                        candidates.append(ExitPoint(
                            time=time_sec,
                            reason="Repetitive section detected",
                            score=score,
                            section='repetitive'
                        ))
        
        # 3. BEFORE ENERGY PEAKS (exit before it gets too hype)
        for i in range(min_idx, max_idx):
            if i < len(energy_profile) - 20:
                future_window = energy_profile[i:min(len(energy_profile), i+20)]
                current = energy_profile[i]
                
                if len(future_window) > 0:
                    future_max = np.max(future_window)
                    
                    # Exit before big energy peak
                    if future_max > current + 0.2 and current < 0.6:
                        time_sec = float(i / 10)
                        score = 70.0
                        
                        candidates.append(ExitPoint(
                            time=time_sec,
                            reason="Before energy peak",
                            score=score,
                            section='pre_peak'
                        ))
        
        # 4. STRUCTURAL BOUNDARIES (chorus endings, verse endings)
        if structure:
            # Try to use outro if it's in range
            outro = structure.outro if hasattr(structure, 'outro') else None
            if outro and len(outro) >= 2:
                outro_start, outro_end = outro[0], outro[1]
                if min_time <= outro_start <= max_time:
                    candidates.append(ExitPoint(
                        time=float(outro_start),
                        reason="Start of outro",
                        score=95.0,
                        section='outro'
                    ))
            
            # Check for chorus endings
            if hasattr(structure, 'chorus') and structure.chorus:
                for chorus_start, chorus_end in structure.chorus:
                    if min_time <= chorus_end <= max_time:
                        candidates.append(ExitPoint(
                            time=float(chorus_end),
                            reason="End of chorus",
                            score=85.0,
                            section='chorus_end'
                        ))
        
        # 5. BEAT-ALIGNED EXITS (always exit on a beat)
        # Handle BeatGrid object or list of beats
        beat_times = []
        if beat_grid:
            if hasattr(beat_grid, 'beat_times'):
                # BeatGrid object
                beat_times = beat_grid.beat_times
            elif hasattr(beat_grid, 'beats'):
                # Could be .beats attribute
                beat_times = beat_grid.beats
            elif isinstance(beat_grid, (list, np.ndarray)):
                # Already a list/array of times
                beat_times = beat_grid
        
        if len(beat_times) > 0:
            for beat_time in beat_times:
                if min_time <= beat_time <= max_time:
                    # Find nearby candidate and align to beat
                    for candidate in candidates:
                        if abs(candidate.time - beat_time) < 2.0:
                            # Adjust to beat
                            candidate.time = float(beat_time)
                            candidate.score += 5.0  # Bonus for beat alignment
        
        return candidates


if __name__ == "__main__":
    print("Vibe-Based Exit Detector")
    print("=" * 60)
    print("Determines exit time based on song's ACTUAL vibe:")
    print("  • High energy → SHORT (30-60s)")
    print("  • Medium energy → NORMAL (60-90s)")
    print("  • Low energy/chill → LONGER (90-150s)")
    print("\nRespects song structure and energy flow!")