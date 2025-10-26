"""
Main DJ Mixing Algorithm
Integrates audio analysis, CLAP classification, and transition generation
to create intelligent, professional-sounding DJ mixes
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import json
from pathlib import Path

from audio_analysis import AudioAnalyzer, BeatGrid, TrackStructure, analyze_key_compatibility
from clap_classifier import CLAPClassifier, calculate_vocal_clash_penalty, calculate_genre_compatibility
from transition_generator import TransitionGenerator, MixingInstructions, TransitionPoint


@dataclass
class TrackInfo:
    """Input track information"""
    file_path: str
    bpm: float
    key: str
    stems: Optional[Dict[str, str]] = None


@dataclass
class AnalyzedTrack:
    """Complete track analysis"""
    track_info: TrackInfo
    bpm: float
    key: str
    beat_grid: BeatGrid
    energy_profile: np.ndarray
    bass_profile: Dict
    loudness: Dict
    structure: TrackStructure
    vocals: Dict
    genres: List[Tuple[str, float]]
    energy_semantic: Dict
    mood: List[Tuple[str, float]]
    embedding: np.ndarray


@dataclass
class PlaylistTransition:
    """Transition between two tracks in playlist"""
    track_a_index: int
    track_b_index: int
    mixing_instructions: MixingInstructions
    compatibility_score: float


class DJMixingAlgorithm:
    """
    Main DJ Mixing Algorithm
    
    Combines multiple analysis techniques to create intelligent, professional
    transitions between tracks while avoiding common DJ mistakes.
    """
    
    def __init__(self, clap_model_name: str = "laion/larger_clap_music_and_speech"):
        """
        Initialize DJ mixing algorithm
        
        Args:
            clap_model_name: CLAP model to use for semantic analysis
        """
        print("Initializing DJ Mixing Algorithm...")
        self.audio_analyzer = AudioAnalyzer()
        self.clap_classifier = CLAPClassifier(model_name=clap_model_name)
        self.transition_generator = TransitionGenerator()
        print("✓ Initialization complete")
    
    def analyze_track(self, track_info: TrackInfo) -> AnalyzedTrack:
        """
        Perform complete analysis on a single track
        
        Combines:
        - Librosa: BPM, energy, bass, structure (technical analysis)
        - CLAP: Vocals, genre, mood (semantic analysis)
        
        Args:
            track_info: Track metadata and file path
            
        Returns:
            Complete analyzed track data
        """
        print(f"\n📀 Analyzing: {Path(track_info.file_path).name}")
        
        # Load audio
        y = self.audio_analyzer.load_audio(track_info.file_path)
        
        # Technical analysis (librosa)
        print("  🔧 Technical analysis...")
        beat_grid = self.audio_analyzer.analyze_bpm_and_beats(y)
        energy_profile = self.audio_analyzer.calculate_energy_profile(y)
        bass_profile = self.audio_analyzer.analyze_bass_profile(y)
        loudness = self.audio_analyzer.analyze_loudness(y)
        structure = self.audio_analyzer.detect_structure(y, beat_grid.beat_times)
        
        # Semantic analysis (CLAP)
        print("  🎵 Semantic analysis...")
        clap_results = self.clap_classifier.analyze_track_comprehensive(track_info.file_path)
        
        # Use provided BPM/key or detected ones
        bpm = track_info.bpm if track_info.bpm > 0 else beat_grid.bpm
        key = track_info.key if track_info.key else "Unknown"
        
        print(f"  ✓ BPM: {bpm:.1f} | Key: {key} | Vocals: {clap_results['vocals']['classification']}")
        
        return AnalyzedTrack(
            track_info=track_info,
            bpm=bpm,
            key=key,
            beat_grid=beat_grid,
            energy_profile=energy_profile,
            bass_profile=bass_profile,
            loudness=loudness,
            structure=structure,
            vocals=clap_results['vocals'],
            genres=clap_results['genres'],
            energy_semantic=clap_results['energy'],
            mood=clap_results['mood'],
            embedding=clap_results['embedding']
        )
    
    def calculate_transition_score(self, track_a: AnalyzedTrack, 
                                   track_b: AnalyzedTrack) -> Dict[str, float]:
        """
        Calculate comprehensive compatibility score between two tracks
        
        Scoring factors:
        1. Beat matching (BPM compatibility)
        2. Key compatibility (harmonic mixing)
        3. Energy matching (flow management)
        4. Vocal clash penalty (avoid vocal collisions)
        5. Genre compatibility (maintain coherence)
        6. Semantic similarity (CLAP embeddings)
        
        Returns:
            Dictionary with individual scores and overall score (0-100)
        """
        scores = {}
        
        # 1. Beat Match Score (20% weight)
        bpm_compat = self.audio_analyzer.calculate_tempo_compatibility(
            track_a.bpm, track_b.bpm
        )
        scores['beat_match'] = bpm_compat['score']
        
        # 2. Key Compatibility (15% weight)
        scores['key_compatibility'] = analyze_key_compatibility(track_a.key, track_b.key)
        
        # 3. Energy Match Score (20% weight)
        # Compare average energy levels
        energy_a = np.mean(track_a.energy_profile)
        energy_b = np.mean(track_b.energy_profile)
        energy_diff = abs(energy_a - energy_b)
        scores['energy_match'] = max(0, 100 - energy_diff * 200)  # Penalize large differences
        
        # 4. Vocal Clash Penalty (25% weight - most important!)
        vocal_penalty = calculate_vocal_clash_penalty(track_a.vocals, track_b.vocals)
        scores['vocal_compatibility'] = 100 - vocal_penalty
        
        # 5. Genre Compatibility (15% weight)
        scores['genre_compatibility'] = calculate_genre_compatibility(
            track_a.genres, track_b.genres
        )
        
        # 6. Semantic Similarity (5% weight)
        similarity = self.clap_classifier.calculate_similarity(
            track_a.embedding, track_b.embedding
        )
        scores['semantic_similarity'] = similarity
        
        # Calculate weighted overall score
        weights = {
            'beat_match': 0.20,
            'key_compatibility': 0.15,
            'energy_match': 0.20,
            'vocal_compatibility': 0.25,  # Most important!
            'genre_compatibility': 0.15,
            'semantic_similarity': 0.05
        }
        
        overall = sum(scores[key] * weights[key] for key in scores.keys())
        scores['overall_score'] = overall
        
        return scores
    
    def find_optimal_transition_point(self, track_a: AnalyzedTrack,
                                     track_b: AnalyzedTrack) -> TransitionPoint:
        """
        Find best transition point between tracks
        
        Delegates to TransitionGenerator with full track analysis
        """
        # Convert AnalyzedTrack to dict format for transition generator
        track_a_dict = {
            'bpm': track_a.bpm,
            'key': track_a.key,
            'beat_grid': track_a.beat_grid,
            'energy_profile': track_a.energy_profile,
            'bass_profile': track_a.bass_profile,
            'structure': track_a.structure,
            'vocals': track_a.vocals,
        }
        
        track_b_dict = {
            'bpm': track_b.bpm,
            'key': track_b.key,
            'beat_grid': track_b.beat_grid,
            'energy_profile': track_b.energy_profile,
            'bass_profile': track_b.bass_profile,
            'structure': track_b.structure,
            'vocals': track_b.vocals,
        }
        
        compatibility = self.calculate_transition_score(track_a, track_b)['overall_score']
        
        return self.transition_generator.find_transition_point(
            track_a_dict, track_b_dict, compatibility
        )
    
    def generate_mixing_instructions(self, track_a: AnalyzedTrack,
                                    track_b: AnalyzedTrack) -> MixingInstructions:
        """
        Generate complete mixing instructions for transition
        
        Returns:
            Detailed mixing instructions including EQ, volume, effects
        """
        # Convert to dict format
        track_a_dict = {
            'bpm': track_a.bpm,
            'key': track_a.key,
            'beat_grid': track_a.beat_grid,
            'energy_profile': track_a.energy_profile,
            'bass_profile': track_a.bass_profile,
            'structure': track_a.structure,
            'vocals': track_a.vocals,
        }
        
        track_b_dict = {
            'bpm': track_b.bpm,
            'key': track_b.key,
            'beat_grid': track_b.beat_grid,
            'energy_profile': track_b.energy_profile,
            'bass_profile': track_b.bass_profile,
            'structure': track_b.structure,
            'vocals': track_b.vocals,
        }
        
        compatibility = self.calculate_transition_score(track_a, track_b)['overall_score']
        
        return self.transition_generator.generate_complete_instructions(
            track_a_dict, track_b_dict, compatibility
        )
    
    def sequence_playlist(self, tracks: List[AnalyzedTrack],
                         optimization: str = 'greedy') -> Dict:
        """
        Optimize track order for entire playlist
        
        DJ Context: Track order matters! Build energy gradually, create peaks
        and valleys, maintain genre coherence, and tell a story with music.
        
        Args:
            tracks: List of analyzed tracks
            optimization: Algorithm to use ('greedy' or 'genetic')
            
        Returns:
            Optimized playlist with transitions
        """
        print(f"\n🎧 Sequencing {len(tracks)} tracks...")
        
        if len(tracks) < 2:
            return {
                'optimized_order': list(range(len(tracks))),
                'transitions': [],
                'total_score': 0,
                'energy_curve': []
            }
        
        if optimization == 'greedy':
            return self._sequence_greedy(tracks)
        else:
            # Future: implement genetic algorithm for better optimization
            return self._sequence_greedy(tracks)
    
    def _sequence_greedy(self, tracks: List[AnalyzedTrack]) -> Dict:
        """
        Greedy sequencing algorithm
        
        Strategy:
        1. Start with a good opener (medium-high energy, broad appeal)
        2. For each position, pick the best-matching remaining track
        3. Occasionally allow energy dips for variety
        
        This creates locally optimal transitions but may not be globally optimal.
        """
        n = len(tracks)
        
        # Calculate all pairwise compatibility scores
        print("  📊 Calculating compatibility matrix...")
        compat_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    score = self.calculate_transition_score(tracks[i], tracks[j])
                    compat_matrix[i, j] = score['overall_score']
        
        # Find good starting track (medium-high energy, not too niche)
        energy_scores = [np.mean(t.energy_profile) for t in tracks]
        genre_diversity = [len(t.genres) for t in tracks]  # More genres = more versatile
        
        start_scores = []
        for i in range(n):
            score = energy_scores[i] * 50 + genre_diversity[i] * 10
            start_scores.append(score)
        
        current_idx: int = int(np.argmax(start_scores))
        sequence = [current_idx]
        remaining = set(range(n)) - {current_idx}
        
        print(f"  🎬 Starting with track {current_idx}: {Path(tracks[current_idx].track_info.file_path).name}")
        
        # Greedy selection
        while remaining:
            # Find best next track
            best_score = -1
            best_idx: Optional[int] = None
            
            for idx in remaining:
                score = compat_matrix[current_idx, idx]
                
                # Slight bonus for gradual energy progression
                if len(sequence) > 1:
                    current_energy = energy_scores[current_idx]
                    next_energy = energy_scores[idx]
                    # Prefer gradual changes
                    energy_bonus = 10 if abs(next_energy - current_energy) < 0.2 else 0
                    score += energy_bonus
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            
            # Safety check (should never happen if remaining is not empty)
            if best_idx is None:
                # Fallback: pick any remaining track
                best_idx = next(iter(remaining))
            
            sequence.append(best_idx)
            remaining.remove(best_idx)
            assert best_idx is not None  # Type narrowing for checker
            current_idx = best_idx
            
            print(f"  ➡️  Track {best_idx} (compatibility: {best_score:.1f})")
        
        # Generate transitions for optimized sequence
        print("\n  🎚️  Generating mixing instructions...")
        transitions = []
        for i in range(len(sequence) - 1):
            track_a = tracks[sequence[i]]
            track_b = tracks[sequence[i + 1]]
            
            compatibility = compat_matrix[sequence[i], sequence[i + 1]]
            instructions = self.generate_mixing_instructions(track_a, track_b)
            
            transitions.append(PlaylistTransition(
                track_a_index=sequence[i],
                track_b_index=sequence[i + 1],
                mixing_instructions=instructions,
                compatibility_score=compatibility
            ))
        
        # Calculate energy curve
        energy_curve = [energy_scores[idx] for idx in sequence]
        
        # Calculate total score
        total_score = np.mean([t.compatibility_score for t in transitions]) if transitions else 0
        
        print(f"\n  ✓ Sequencing complete! Average compatibility: {total_score:.1f}/100")
        
        return {
            'optimized_order': sequence,
            'transitions': transitions,
            'total_score': float(total_score),
            'energy_curve': energy_curve
        }


def save_mixing_instructions(result: Dict, output_path: str):
    """
    Save mixing instructions to JSON file
    
    Args:
        result: Output from sequence_playlist
        output_path: Where to save JSON
    """
    # Convert to serializable format
    output = {
        'optimized_order': result['optimized_order'],
        'total_score': result['total_score'],
        'energy_curve': result['energy_curve'],
        'transitions': []
    }
    
    for transition in result['transitions']:
        trans_dict = {
            'track_a_index': transition.track_a_index,
            'track_b_index': transition.track_b_index,
            'compatibility_score': transition.compatibility_score,
            'transition_point': {
                'track_a_out_time': transition.mixing_instructions.transition_point.track_a_out_time,
                'track_b_in_time': transition.mixing_instructions.transition_point.track_b_in_time,
                'crossfade_duration': transition.mixing_instructions.transition_point.crossfade_duration,
                'transition_type': transition.mixing_instructions.transition_point.transition_type.value,
                'confidence': transition.mixing_instructions.transition_point.confidence
            },
            'eq_automation': {
                'track_a': {
                    'timestamps': transition.mixing_instructions.track_a_eq.timestamps,
                    'bass': transition.mixing_instructions.track_a_eq.bass,
                    'mid': transition.mixing_instructions.track_a_eq.mid,
                    'high': transition.mixing_instructions.track_a_eq.high
                },
                'track_b': {
                    'timestamps': transition.mixing_instructions.track_b_eq.timestamps,
                    'bass': transition.mixing_instructions.track_b_eq.bass,
                    'mid': transition.mixing_instructions.track_b_eq.mid,
                    'high': transition.mixing_instructions.track_b_eq.high
                }
            },
            'volume_automation': {
                'track_a': transition.mixing_instructions.track_a_volume,
                'track_b': transition.mixing_instructions.track_b_volume
            },
            'effects': transition.mixing_instructions.effects,
            'notes': transition.mixing_instructions.notes
        }
        output['transitions'].append(trans_dict)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Mixing instructions saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("DJ AUTO-MIXING ALGORITHM")
    print("="*60)
    print("\nThis algorithm creates professional DJ mixes by analyzing:")
    print("  🎵 Technical features (BPM, key, energy, bass)")
    print("  🎤 Semantic features (vocals, genre, mood)")
    print("  🎚️  Optimal transition points and mixing instructions")
    print("\nKey DJ Rules Implemented:")
    print("  ✓ Beat matching and phrase alignment")
    print("  ✓ Never mix two bass-heavy tracks at full volume")
    print("  ✓ Avoid vocal clashing")
    print("  ✓ Maintain energy flow")
    print("  ✓ Ensure harmonic compatibility")
    print("="*60)