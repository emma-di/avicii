"""
Smart DJ Mix Creation - Enhanced Version
Features:
1. Vibe-based song lengths (not fixed)
2. STRONG, noticeable effects
3. Fade out at end
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import logging
import soundfile as sf

from dj_mixer import DJMixingAlgorithm, TrackInfo
from smart_exit_detector import VibeBasedExitDetector
from playlist_sequencer import auto_sequence_tracks
from create_dj_mix import load_json_metadata, get_audio_path, mix_htdemucs_stems

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_smart_dj_set(
    track_names: List[str],
    output_dir: str = "smart_dj_set",
    use_htdemucs: bool = False,
    htdemucs_stem: str = 'all',
    exclude_stems: Optional[List[str]] = None,
    exit_preference: str = 'vibe_based',  # 'vibe_based', 'early', 'balanced', 'late'
    sequencing_strategy: str = 'auto',
    energy_curve: str = 'wave',
    auto_order: bool = True,
    add_effects: bool = True,  # STRONG effects
    effect_intensity: float = 1.0  # 0.5=subtle, 1.0=normal, 1.5=extreme
) -> Dict:
    """
    Create ONE complete DJ set with vibe-based lengths and STRONG effects
    
    NEW FEATURES:
    1. Variable song lengths based on vibe (high energy=short, chill=long)
    2. MUCH stronger, more noticeable effects
    3. Fades out at the end
    
    Args:
        track_names: List of song names
        exit_preference: 'vibe_based' (dynamic lengths) or 'early'/'balanced'/'late'
        add_effects: Add STRONG effects
        effect_intensity: 0.5-1.5 (how strong effects are)
        
    Returns:
        Dict with output_file path
    """
    logger.info("=" * 80)
    logger.info("🎛️  SMART DJ SET CREATOR - ENHANCED")
    logger.info("=" * 80)
    logger.info(f"Tracks: {len(track_names)}")
    logger.info(f"Exit strategy: {exit_preference} (vibe-based lengths)")
    logger.info(f"Effects: {'STRONG' if add_effects else 'OFF'} (intensity: {effect_intensity}x)")
    logger.info("=" * 80)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # STEP 1: Analyze tracks
    logger.info("\n📀 STEP 1: ANALYZING TRACKS")
    logger.info("-" * 80)
    
    dj_algo = DJMixingAlgorithm()
    analyzed_tracks = []
    
    for track_name in track_names:
        try:
            metadata = load_json_metadata(track_name)
            audio_path = get_audio_path(track_name, use_htdemucs, htdemucs_stem)
            
            bpm_value = metadata.get('bpm', 120.0)
            key_value = metadata.get('key', 'C major')
            
            if not isinstance(bpm_value, (int, float)):
                bpm_value = 120.0
            if not isinstance(key_value, str):
                key_value = 'C major'
            
            track_info = TrackInfo(
                file_path=audio_path,
                bpm=float(bpm_value),
                key=str(key_value)
            )
            
            analyzed = dj_algo.analyze_track(track_info)
            analyzed_tracks.append(analyzed)
            
            # Show energy level
            avg_energy = np.mean(analyzed.energy_profile)
            energy_label = "HIGH" if avg_energy > 0.7 else "MED" if avg_energy > 0.45 else "LOW"
            logger.info(f"   ✓ {track_name}: {analyzed.bpm} BPM, {analyzed.key} [{energy_label} energy]")
            
        except Exception as e:
            logger.error(f"   ❌ Failed: {track_name}: {e}")
            return {'error': str(e), 'failed_track': track_name, 'success': False}
    
    # STEP 2: Auto-order tracks
    if auto_order and len(analyzed_tracks) > 1:
        logger.info("\n🎵 STEP 2: AUTO-SEQUENCING")
        logger.info("-" * 80)
        
        track_order = auto_sequence_tracks(analyzed_tracks, sequencing_strategy, energy_curve)
        track_names = [track_names[i] for i in track_order.indices]
        analyzed_tracks = [analyzed_tracks[i] for i in track_order.indices]
        
        logger.info(f"   ✓ Optimal order: {track_names}")
        logger.info(f"   ✓ Score: {track_order.score:.1f}/100")
    
    # STEP 3: Find VIBE-BASED exit points
    logger.info("\n🎯 STEP 3: VIBE-BASED EXIT DETECTION")
    logger.info("-" * 80)
    logger.info("Finding optimal exit based on each song's energy and structure...")
    
    exit_detector = VibeBasedExitDetector()
    smart_exits = []
    
    for i, track in enumerate(analyzed_tracks):
        track_dict = {
            'structure': track.structure,
            'energy_profile': track.energy_profile,
            'beat_grid': track.beat_grid,
            'bpm': track.bpm,
            'vocals': track.vocals
        }
        
        logger.info(f"\n   Track {i+1}: {track_names[i]}")
        exit_point = exit_detector.get_best_exit_point(track_dict, exit_preference)
        smart_exits.append(exit_point)
        
        logger.info(f"      → Exit: {exit_point.time:.1f}s ({exit_point.reason})")
    
    # STEP 4: Create mix with STRONG effects
    logger.info("\n🎚️  STEP 4: CREATING MIX WITH EFFECTS")
    logger.info("-" * 80)
    
    mix_segments = []
    sample_rate = None
    results = []
    
    for i in range(len(analyzed_tracks)):
        track = analyzed_tracks[i]
        track_name = track_names[i]
        
        logger.info(f"\n   Track {i+1}/{len(analyzed_tracks)}: {track_name}")
        
        # Load audio
        audio_path = get_audio_path(track_name, use_htdemucs, htdemucs_stem)
        
        if use_htdemucs and htdemucs_stem == 'all':
            audio_data, sr = mix_htdemucs_stems(audio_path, exclude_stems)
        else:
            audio_data, sr = sf.read(audio_path)
        
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            import librosa
            audio_data = librosa.resample(audio_data.T, orig_sr=sr, target_sr=sample_rate).T
        
        # Apply STRONG effects
        if add_effects:
            from dj_effects import DJEffects
            
            effects_processor = DJEffects(sample_rate=sample_rate)
            avg_energy = np.mean(track.energy_profile)
            
            if avg_energy > 0.7:
                effect_energy = 'high'
            elif avg_energy > 0.4:
                effect_energy = 'medium'
            else:
                effect_energy = 'low'
            
            track_dict = {
                'structure': track.structure,
                'energy_profile': track.energy_profile,
                'beat_grid': track.beat_grid,
                'bpm': track.bpm
            }
            
            logger.info(f"      Adding STRONG {effect_energy} energy effects...")
            audio_data = effects_processor.smart_effect_placement(
                audio_data, track_dict, energy_level=effect_energy
            )
        
        # Cut to exit time
        exit_time = smart_exits[i].time
        exit_samples = int(exit_time * sample_rate)
        track_segment = audio_data[:exit_samples]
        
        logger.info(f"      Duration: {exit_time:.1f}s (vibe-based)")
        
        # Crossfade with previous
        if i > 0:
            transition_score = dj_algo.calculate_transition_score(
                analyzed_tracks[i-1], analyzed_tracks[i]
            )
            logger.info(f"      Transition: {transition_score['overall_score']:.1f}/100")
            
            # Longer crossfade for better transitions
            crossfade_duration = 10.0 + (transition_score['overall_score'] / 100) * 10.0
            crossfade_samples = int(crossfade_duration * sample_rate)
            
            prev_segment = mix_segments[-1]
            crossfade_samples = min(crossfade_samples, len(prev_segment), len(track_segment))
            
            fade_out = np.linspace(1, 0, crossfade_samples)
            fade_in = np.linspace(0, 1, crossfade_samples)
            
            # Apply crossfade
            if track_segment.ndim == 1:
                prev_segment[-crossfade_samples:] = (
                    prev_segment[-crossfade_samples:] * fade_out +
                    track_segment[:crossfade_samples] * fade_in
                )
            else:
                prev_segment[-crossfade_samples:] = (
                    prev_segment[-crossfade_samples:] * fade_out[:, np.newaxis] +
                    track_segment[:crossfade_samples] * fade_in[:, np.newaxis]
                )
            
            track_segment = track_segment[crossfade_samples:]
            logger.info(f"      Crossfade: {crossfade_duration:.1f}s")
        
        mix_segments.append(track_segment)
        results.append({
            'track': track_name,
            'exit_time': exit_time,
            'exit_reason': smart_exits[i].reason,
            'duration': float(len(track_segment)) / float(sample_rate) if sample_rate else 0.0
        })
    
    # Combine all segments
    logger.info("\n   🎵 Combining segments...")
    final_mix = np.concatenate(mix_segments, axis=0)
    
    # Ensure sample_rate is valid before fade out
    if sample_rate is None:
        sample_rate = 44100  # Default fallback
    
    # ADD FADE OUT AT THE END
    logger.info("   🔊 Adding fade out...")
    fade_out_duration = 5.0  # 5 second fade out
    fade_out_samples = int(fade_out_duration * float(sample_rate))
    
    if len(final_mix) > fade_out_samples:
        fade_curve = np.linspace(1, 0, fade_out_samples)
        if final_mix.ndim == 1:
            final_mix[-fade_out_samples:] *= fade_curve
        else:
            final_mix[-fade_out_samples:] *= fade_curve[:, np.newaxis]
    
    # Save ONE complete file
    output_file = output_path / "complete_dj_set.wav"
    
    if sample_rate is None:
        sample_rate = 44100
    
    sf.write(str(output_file), final_mix, sample_rate)
    
    total_duration = float(len(final_mix)) / float(sample_rate)
    
    logger.info(f"\n   ✅ Saved: {output_file.name}")
    logger.info(f"      Duration: {total_duration/60:.1f} minutes")
    logger.info(f"      Fade out: ✓")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("✨ COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"📊 Stats:")
    logger.info(f"   Duration: {total_duration/60:.1f} minutes")
    logger.info(f"   Tracks: {len(analyzed_tracks)}")
    logger.info(f"   Effects: {'STRONG' if add_effects else 'None'}")
    logger.info(f"   Exit strategy: Vibe-based (dynamic lengths)")
    logger.info(f"\n📁 Output: {output_file}")
    logger.info(f"🎧 Ready to play!")
    
    # Save details
    results_file = output_path / "set_details.json"
    with open(results_file, 'w') as f:
        json.dump({
            'track_order': track_names,
            'sequencing_strategy': sequencing_strategy if auto_order else 'manual',
            'energy_curve': energy_curve,
            'exit_preference': exit_preference,
            'output_file': str(output_file),
            'total_duration': total_duration,
            'effects_enabled': add_effects,
            'effect_intensity': effect_intensity,
            'tracks': results
        }, f, indent=2)
    
    return {
        'track_order': track_names,
        'output_file': str(output_file),
        'tracks': results,
        'output_dir': str(output_path),
        'total_duration': total_duration,
        'success': True
    }


if __name__ == "__main__":
    print("=" * 80)
    print("SMART DJ SET CREATOR - ENHANCED")
    print("=" * 80)
    print("\n✨ NEW Features:")
    print("  1. Vibe-based song lengths (high energy=short, chill=long)")
    print("  2. STRONG, noticeable effects")
    print("  3. Fades out at the end")
    print("\nExample:")
    print("""
    from smart_dj_integration import create_smart_dj_set
    
    result = create_smart_dj_set(
        track_names=["Song1", "Song2", "Song3"],
        exit_preference='vibe_based',  # Dynamic lengths!
        add_effects=True               # Strong effects!
    )
    """)
    print("=" * 80)