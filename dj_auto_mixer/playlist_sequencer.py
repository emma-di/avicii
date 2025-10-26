"""
Enhanced Playlist Sequencer
Automatically orders tracks for optimal DJ set flow using multiple strategies:
- Energy progression analysis
- Compatibility scoring
- Simulated annealing optimization
- Genre/mood coherence
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import random
from pathlib import Path


@dataclass
class TrackOrder:
    """Represents an ordered track sequence"""
    indices: List[int]  # Order of track indices
    score: float  # Total quality score
    energy_curve: List[float]  # Energy progression
    transition_scores: List[float]  # Individual transition scores


class PlaylistSequencer:
    """
    Advanced playlist sequencing with multiple optimization strategies
    
    Strategies:
    1. Greedy (fast, decent results)
    2. Simulated Annealing (good results, moderate time)
    3. Energy-Curve Optimization (focus on flow)
    4. Genre-Aware (maintain coherence)
    """
    
    def __init__(self, compatibility_matrix: np.ndarray,
                 energy_scores: List[float],
                 track_metadata: List[Dict]):
        """
        Initialize sequencer
        
        Args:
            compatibility_matrix: NxN matrix of pairwise compatibility scores
            energy_scores: Energy level for each track (0-1)
            track_metadata: Metadata for each track (genre, mood, etc.)
        """
        self.compat_matrix = compatibility_matrix
        self.energy_scores = np.array(energy_scores)
        self.track_metadata = track_metadata
        self.n_tracks = len(energy_scores)
    
    def sequence(self, strategy: str = 'auto', 
                target_energy_curve: str = 'wave') -> TrackOrder:
        """
        Sequence tracks using specified strategy
        
        Args:
            strategy: 'greedy', 'simulated_annealing', 'energy_first', 'auto'
            target_energy_curve: 'wave', 'build', 'descend', 'plateau'
            
        Returns:
            TrackOrder with optimal sequence
        """
        if strategy == 'auto':
            # Choose strategy based on playlist size
            if self.n_tracks <= 4:
                strategy = 'greedy'
            elif self.n_tracks <= 10:
                strategy = 'simulated_annealing'
            else:
                strategy = 'energy_first'
        
        print(f"\n🎵 Sequencing {self.n_tracks} tracks using '{strategy}' strategy")
        print(f"   Target energy curve: {target_energy_curve}")
        
        if strategy == 'greedy':
            return self._sequence_greedy(target_energy_curve)
        elif strategy == 'simulated_annealing':
            return self._sequence_simulated_annealing(target_energy_curve)
        elif strategy == 'energy_first':
            return self._sequence_energy_first(target_energy_curve)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _sequence_greedy(self, target_energy: str) -> TrackOrder:
        """
        Greedy sequencing with energy awareness
        
        Fast algorithm: pick best next track at each step
        """
        # Find good starting track
        start_idx = self._find_good_starter(target_energy)
        
        sequence = [start_idx]
        remaining = set(range(self.n_tracks)) - {start_idx}
        
        print(f"   🎬 Starting with track {start_idx}")
        
        while remaining:
            current_idx = sequence[-1]
            current_energy = self.energy_scores[current_idx]
            
            # Calculate target energy for next track
            progress = len(sequence) / self.n_tracks
            target_next_energy = self._get_target_energy(progress, target_energy)
            
            # Score each remaining track
            best_score = -1
            best_idx = None
            
            for idx in remaining:
                # Base score: compatibility
                score = self.compat_matrix[current_idx, idx]
                
                # Energy bonus: prefer tracks matching target energy progression
                next_energy = self.energy_scores[idx]
                energy_diff = abs(next_energy - target_next_energy)
                energy_bonus = 20 * (1 - energy_diff)  # 0-20 points
                score += energy_bonus
                
                # Genre continuity bonus
                if self._same_genre(current_idx, idx):
                    score += 10
                
                # Avoid extreme BPM jumps
                bpm_penalty = self._calculate_bpm_penalty(current_idx, idx)
                score -= bpm_penalty
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx is not None:
                sequence.append(best_idx)
                remaining.remove(best_idx)
                print(f"   ➡️  Track {best_idx} (score: {best_score:.1f})")
        
        return self._create_track_order(sequence)
    
    def _sequence_simulated_annealing(self, target_energy: str,
                                     iterations: int = 1000) -> TrackOrder:
        """
        Simulated annealing optimization
        
        Better global optimization by allowing temporary worse moves
        """
        # Start with greedy solution
        current_order = self._sequence_greedy(target_energy)
        current_sequence = current_order.indices.copy()
        current_score = current_order.score
        
        best_sequence = current_sequence.copy()
        best_score = current_score
        
        # Simulated annealing parameters
        initial_temp = 100.0
        cooling_rate = 0.995
        temperature = initial_temp
        
        print(f"   🔥 Optimizing with simulated annealing ({iterations} iterations)")
        print(f"      Initial score: {current_score:.1f}")
        
        for iteration in range(iterations):
            # Generate neighbor solution (swap two random positions)
            new_sequence = current_sequence.copy()
            i, j = random.sample(range(len(new_sequence)), 2)
            new_sequence[i], new_sequence[j] = new_sequence[j], new_sequence[i]
            
            # Evaluate new solution
            new_order = self._create_track_order(new_sequence)
            new_score = new_order.score
            
            # Calculate acceptance probability
            score_diff = new_score - current_score
            if score_diff > 0:
                # Better solution, always accept
                current_sequence = new_sequence
                current_score = new_score
                
                if new_score > best_score:
                    best_sequence = new_sequence.copy()
                    best_score = new_score
            else:
                # Worse solution, accept with probability based on temperature
                acceptance_prob = np.exp(score_diff / temperature)
                if random.random() < acceptance_prob:
                    current_sequence = new_sequence
                    current_score = new_score
            
            # Cool down
            temperature *= cooling_rate
            
            # Progress update
            if iteration % 200 == 0:
                print(f"      Iteration {iteration}: best={best_score:.1f}, current={current_score:.1f}, temp={temperature:.1f}")
        
        print(f"   ✓ Optimization complete! Final score: {best_score:.1f}")
        return self._create_track_order(best_sequence)
    
    def _sequence_energy_first(self, target_energy: str) -> TrackOrder:
        """
        Sort primarily by energy curve, then optimize transitions
        
        Good for creating specific energy progressions
        """
        print(f"   ⚡ Sorting by energy curve first")
        
        # Sort by energy based on target curve
        if target_energy == 'build':
            # Low to high energy
            sorted_indices = np.argsort(self.energy_scores)
        elif target_energy == 'descend':
            # High to low energy
            sorted_indices = np.argsort(self.energy_scores)[::-1]
        elif target_energy == 'plateau':
            # Similar energy throughout
            sorted_indices = np.argsort(self.energy_scores)
        else:  # wave
            # Create wave pattern: medium, high, medium, low, medium, high...
            sorted_by_energy = np.argsort(self.energy_scores)
            n = len(sorted_by_energy)
            
            # Split into energy tiers
            low = sorted_by_energy[:n//3]
            mid = sorted_by_energy[n//3:2*n//3]
            high = sorted_by_energy[2*n//3:]
            
            # Interleave: mid, high, low, mid, high, low...
            sorted_indices = []
            for i in range(max(len(low), len(mid), len(high))):
                if i < len(mid):
                    sorted_indices.append(mid[i])
                if i < len(high):
                    sorted_indices.append(high[i])
                if i < len(low):
                    sorted_indices.append(low[i])
            
            sorted_indices = np.array(sorted_indices)
        
        sequence = sorted_indices.tolist()
        
        # Local optimization: swap adjacent tracks if it improves compatibility
        improved = True
        iterations = 0
        while improved and iterations < 10:
            improved = False
            iterations += 1
            
            for i in range(len(sequence) - 1):
                idx_a = sequence[i]
                idx_b = sequence[i + 1]
                
                current_score = self.compat_matrix[idx_a, idx_b]
                
                # Try swapping
                sequence[i], sequence[i + 1] = sequence[i + 1], sequence[i]
                new_score = self.compat_matrix[sequence[i], sequence[i + 1]]
                
                if new_score > current_score + 5:  # Only swap if significant improvement
                    improved = True
                    print(f"      Swapped positions {i} and {i+1} (+{new_score - current_score:.1f})")
                else:
                    # Revert swap
                    sequence[i], sequence[i + 1] = sequence[i + 1], sequence[i]
        
        return self._create_track_order(sequence)
    
    def _find_good_starter(self, target_energy: str) -> int:
        """Find a good track to start the set"""
        
        if target_energy == 'build':
            # Start with lower energy
            return int(np.argmin(self.energy_scores))
        elif target_energy == 'descend':
            # Start with high energy
            return int(np.argmax(self.energy_scores))
        else:  # wave or plateau
            # Start with medium energy
            median_energy = np.median(self.energy_scores)
            distances = np.abs(self.energy_scores - median_energy)
            return int(np.argmin(distances))
    
    def _get_target_energy(self, progress: float, curve_type: str) -> float:
        """
        Get target energy level at given progress through set
        
        Args:
            progress: 0-1, how far through the set
            curve_type: Type of energy curve
            
        Returns:
            Target energy level (0-1)
        """
        if curve_type == 'build':
            # Linear increase
            return progress
        elif curve_type == 'descend':
            # Linear decrease
            return 1 - progress
        elif curve_type == 'plateau':
            # Stay around 0.7
            return 0.7
        else:  # wave
            # Sine wave with slight upward trend
            wave = 0.5 + 0.3 * np.sin(progress * 2 * np.pi * 1.5)
            trend = progress * 0.2
            return wave + trend
    
    def _same_genre(self, idx1: int, idx2: int) -> bool:
        """Check if two tracks have same primary genre"""
        if not self.track_metadata:
            return False
        
        genres1 = self.track_metadata[idx1].get('genres', [])
        genres2 = self.track_metadata[idx2].get('genres', [])
        
        if not genres1 or not genres2:
            return False
        
        # Check if top genres match
        top_genre1 = genres1[0][0] if genres1 else None
        top_genre2 = genres2[0][0] if genres2 else None
        
        return top_genre1 == top_genre2
    
    def _calculate_bpm_penalty(self, idx1: int, idx2: int) -> float:
        """Penalize large BPM jumps"""
        if not self.track_metadata:
            return 0
        
        bpm1 = self.track_metadata[idx1].get('bpm', 120)
        bpm2 = self.track_metadata[idx2].get('bpm', 120)
        
        bpm_diff = abs(bpm2 - bpm1)
        
        if bpm_diff < 5:
            return 0
        elif bpm_diff < 10:
            return 5
        elif bpm_diff < 20:
            return 15
        else:
            return 30
    
    def _create_track_order(self, sequence: List[int]) -> TrackOrder:
        """
        Create TrackOrder object from sequence
        
        Calculates score, energy curve, and transition scores
        """
        # Calculate transition scores
        transition_scores = []
        for i in range(len(sequence) - 1):
            score = self.compat_matrix[sequence[i], sequence[i + 1]]
            transition_scores.append(score)
        
        # Extract energy curve
        energy_curve = [self.energy_scores[idx] for idx in sequence]
        
        # Calculate total score
        # Base score: average transition quality
        avg_transition = np.mean(transition_scores) if transition_scores else 0
        
        # Energy flow score: penalize jerky changes
        energy_flow_score = 0
        if len(energy_curve) > 1:
            energy_changes = np.diff(energy_curve)
            smoothness = 1 - np.std(energy_changes) * 2  # Less std = smoother
            energy_flow_score = max(0, smoothness * 20)  # 0-20 points
        
        # Genre coherence score
        genre_score = 0
        for i in range(len(sequence) - 1):
            if self._same_genre(sequence[i], sequence[i + 1]):
                genre_score += 5
        genre_score = min(genre_score, 20)  # Cap at 20
        
        # Total score
        total_score = avg_transition + energy_flow_score + genre_score
        
        return TrackOrder(
            indices=sequence,
            score=int(total_score),
            energy_curve=energy_curve,
            transition_scores=transition_scores
        )


def auto_sequence_tracks(analyzed_tracks: List, 
                        strategy: str = 'auto',
                        energy_curve: str = 'wave') -> TrackOrder:
    """
    Helper function to automatically sequence a list of analyzed tracks
    
    Args:
        analyzed_tracks: List of AnalyzedTrack objects from dj_mixer
        strategy: 'auto', 'greedy', 'simulated_annealing', 'energy_first'
        energy_curve: 'wave', 'build', 'descend', 'plateau'
        
    Returns:
        TrackOrder with optimal sequence
    """
    from dj_mixer import DJMixingAlgorithm
    
    n = len(analyzed_tracks)
    
    # Build compatibility matrix
    print(f"📊 Calculating {n}x{n} compatibility matrix...")
    dj_algo = DJMixingAlgorithm()
    compat_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                scores = dj_algo.calculate_transition_score(
                    analyzed_tracks[i], 
                    analyzed_tracks[j]
                )
                compat_matrix[i, j] = scores['overall_score']
    
    # Extract energy scores and metadata
    energy_scores = [np.mean(track.energy_profile) for track in analyzed_tracks]
    
    track_metadata = []
    for track in analyzed_tracks:
        track_metadata.append({
            'bpm': track.bpm,
            'genres': track.genres,
            'energy': np.mean(track.energy_profile)
        })
    
    # Create sequencer and run
    sequencer = PlaylistSequencer(compat_matrix, energy_scores, track_metadata)
    order = sequencer.sequence(strategy, energy_curve)
    
    print(f"\n✅ Sequencing complete!")
    print(f"   Final score: {order.score:.1f}/100")
    print(f"   Energy flow: {order.energy_curve[0]:.2f} → {order.energy_curve[-1]:.2f}")
    
    return order


if __name__ == "__main__":
    print("Enhanced Playlist Sequencer")
    print("=" * 60)
    print("Automatically orders tracks for optimal DJ set flow")
    print("\nStrategies:")
    print("  • Greedy - Fast, decent results")
    print("  • Simulated Annealing - Best quality, moderate time")
    print("  • Energy First - Optimize energy curve")
    print("  • Auto - Choose based on playlist size")
    print("\nEnergy Curves:")
    print("  • Wave - Peaks and valleys with upward trend")
    print("  • Build - Gradual energy increase")
    print("  • Descend - Start high, wind down")
    print("  • Plateau - Maintain consistent energy")
    print("\nUsage:")
    print("  order = auto_sequence_tracks(analyzed_tracks)")
    print("  print(f'Best order: {order.indices}')")