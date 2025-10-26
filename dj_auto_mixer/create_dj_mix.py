"""
Complete DJ Mix Creation Pipeline - Updated for exact folder structure
Creates intelligent DJ mixes from JSON metadata and audio files

EXACT FOLDER STRUCTURE:
data/
├── htdemucs/
│   ├── SongName/              # Folder per song
│   │   ├── bass.mp3 (or .wav)
│   │   ├── drums.mp3 (or .wav)
│   │   ├── other.mp3 (or .wav)
│   │   └── vocals.mp3 (or .wav)
├── metadata/
│   └── SongName.json          # Direct JSON files
└── mp3s/
    └── SongName.mp3           # Direct audio files
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
import logging
import soundfile as sf

from dj_mixer import DJMixingAlgorithm
from audio_mixer import AudioMixer, quick_mix
from transition_generator import TransitionType

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DictToObject:
    """
    Converts a dictionary to an object with attributes
    This allows dict access like metadata['key'] and object access like metadata.key
    """
    def __init__(self, data: Dict[str, Any]):
        self._data = data
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObject(value))
            else:
                setattr(self, key, value)
    
    def __getitem__(self, key):
        return self._data[key]
    
    def get(self, key, default=None):
        return self._data.get(key, default)
    
    def __contains__(self, key):
        return key in self._data


def load_json_metadata(track_name: str) -> DictToObject:
    """
    Load JSON metadata file from data/metadata/{track_name}.json
    
    Args:
        track_name: Name of the track (without .json extension)
    
    Returns:
        DictToObject containing track metadata (supports both dict and attribute access)
    """
    json_path = Path('data/metadata') / f"{track_name}.json"
    
    if not json_path.exists():
        raise FileNotFoundError(
            f"Could not find JSON at: {json_path}\n"
            f"Make sure file exists in data/metadata/ folder"
        )
    
    logger.info(f"Loading metadata from: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Add file_path if it doesn't exist (required by DJ algorithm)
    if 'file_path' not in data:
        # Try to find the audio file
        mp3_folder = Path('data/mp3s')
        for ext in ['.mp3', '.wav', '.flac', '.m4a', '.ogg']:
            audio_file = mp3_folder / f"{track_name}{ext}"
            if audio_file.exists():
                data['file_path'] = str(audio_file)
                logger.info(f"  Auto-added file_path: {data['file_path']}")
                break
        
        # If still not found, set a placeholder
        if 'file_path' not in data:
            data['file_path'] = f"data/mp3s/{track_name}.mp3"
            logger.warning(f"  file_path not found in JSON, using placeholder: {data['file_path']}")
    
    # Wrap the dict to support both dict and object access
    return DictToObject(data)


def get_audio_path(track_name: str, use_htdemucs: bool = False, 
                   stem: str = 'all') -> str:
    """
    Get the audio file path based on folder structure
    
    Args:
        track_name: Name of the track
        use_htdemucs: If True, use htdemucs folder, else mp3s folder
        stem: Which stem to use from htdemucs ('bass', 'drums', 'other', 'vocals', 'all')
    
    Returns:
        Path to audio file
    """
    if use_htdemucs:
        # htdemucs structure: data/htdemucs/{track_name}/
        htdemucs_folder = Path('data/htdemucs') / track_name
        
        if not htdemucs_folder.exists():
            raise FileNotFoundError(
                f"Could not find htdemucs folder: {htdemucs_folder}\n"
                f"Expected structure: data/htdemucs/{track_name}/"
            )
        
        if stem == 'all':
            # Return the folder path - we'll mix all stems together
            return str(htdemucs_folder)
        else:
            # Return specific stem - try multiple extensions
            extensions = ['.mp3', '.wav', '.flac', '.m4a']
            for ext in extensions:
                stem_file = htdemucs_folder / f"{stem}{ext}"
                if stem_file.exists():
                    logger.info(f"Found stem: {stem_file}")
                    return str(stem_file)
            
            raise FileNotFoundError(
                f"Could not find stem: {stem} in {htdemucs_folder}\n"
                f"Tried extensions: {extensions}\n"
                f"Available stems should be: bass, drums, other, vocals"
            )
    else:
        # mp3s structure: data/mp3s/{track_name}.{ext}
        mp3_folder = Path('data/mp3s')
        
        # Try common extensions
        extensions = ['.mp3', '.wav', '.flac', '.m4a', '.ogg']
        
        for ext in extensions:
            audio_file = mp3_folder / f"{track_name}{ext}"
            if audio_file.exists():
                logger.info(f"Found audio: {audio_file}")
                return str(audio_file)
        
        raise FileNotFoundError(
            f"Could not find audio file for '{track_name}' in {mp3_folder}\n"
            f"Tried extensions: {extensions}"
        )


def mix_htdemucs_stems(htdemucs_folder: str, exclude_stems: Optional[List[str]] = None) -> Tuple[np.ndarray, int]:
    """
    Mix all stems from htdemucs folder into one audio track
    
    Args:
        htdemucs_folder: Path to folder containing stems
        exclude_stems: List of stems to exclude (e.g., ['vocals'] for instrumental)
    
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    folder = Path(htdemucs_folder)
    stem_names = ['bass', 'drums', 'other', 'vocals']
    
    if exclude_stems:
        stem_names = [s for s in stem_names if s not in exclude_stems]
    
    logger.info(f"Mixing stems from: {folder}")
    logger.info(f"Using stems: {', '.join(stem_names)}")
    
    mixed_audio: Optional[np.ndarray] = None
    sample_rate: Optional[int] = None
    
    for stem_name in stem_names:
        # Try both .mp3 and .wav extensions
        stem_path = None
        for ext in ['.mp3', '.wav', '.flac', '.m4a']:
            test_path = folder / f"{stem_name}{ext}"
            if test_path.exists():
                stem_path = test_path
                break
        
        if stem_path is None:
            logger.warning(f"Stem not found: {stem_name} (tried .mp3, .wav, .flac, .m4a), skipping")
            continue
        
        logger.info(f"  Loading: {stem_path.name}")
        audio, sr = sf.read(str(stem_path))
        
        if sample_rate is None:
            sample_rate = sr
            mixed_audio = audio
        else:
            # Mix in this stem
            if len(audio) != len(mixed_audio):  # type: ignore
                # Handle length mismatch by padding/truncating
                target_len = min(len(audio), len(mixed_audio))  # type: ignore
                audio = audio[:target_len]
                mixed_audio = mixed_audio[:target_len]  # type: ignore
            
            mixed_audio = mixed_audio + audio  # type: ignore
    
    if mixed_audio is None or sample_rate is None:
        raise ValueError(f"No stems found in {folder}")
    
    logger.info(f"✓ Mixed {len(stem_names)} stems together")
    return mixed_audio, sample_rate


def create_dj_mix_from_json(
    track1_name: str,
    track2_name: str,
    output_path: str = "dj_mix_output.wav",
    use_htdemucs: bool = False,
    htdemucs_stem: str = 'all',
    exclude_stems: Optional[List[str]] = None,
    target_transition_duration: float = 16.0
) -> Dict:
    """
    Create a DJ mix from two tracks using the exact folder structure
    
    FOLDER STRUCTURE USED:
    data/metadata/{track_name}.json  - Track metadata
    data/mp3s/{track_name}.mp3       - Full audio files
    data/htdemucs/{track_name}/      - Separated stems folder
        ├── bass.wav
        ├── drums.wav
        ├── other.wav
        └── vocals.wav
    
    Args:
        track1_name: Name of first track (e.g., "Dynamite")
        track2_name: Name of second track (e.g., "DJGotUsFallinInLove")
        output_path: Where to save the mixed audio
        use_htdemucs: If True, use htdemucs stems, else use mp3s
        htdemucs_stem: Which stem to use ('bass', 'drums', 'other', 'vocals', 'all')
                      'all' will mix all stems back together
        exclude_stems: List of stems to exclude when htdemucs_stem='all'
                      e.g., ['vocals'] for instrumental mix
        target_transition_duration: How long the crossfade should be (seconds)
    
    Returns:
        Dictionary with mix results and statistics
    """
    logger.info("=" * 60)
    logger.info("🎵 DJ MIX CREATION PIPELINE")
    logger.info("=" * 60)
    
    # Step 1: Load JSON metadata from data/metadata/
    logger.info("\n📂 Loading metadata from data/metadata/...")
    metadata1 = load_json_metadata(track1_name)
    metadata2 = load_json_metadata(track2_name)
    
    logger.info(f"   Track 1: {track1_name}")
    logger.info(f"   Track 2: {track2_name}")
    
    # Step 2: Get audio paths
    logger.info("\n🎧 Locating audio files...")
    if use_htdemucs:
        logger.info(f"   Using: data/htdemucs/ (stem: {htdemucs_stem})")
        if exclude_stems:
            logger.info(f"   Excluding stems: {', '.join(exclude_stems)}")
    else:
        logger.info(f"   Using: data/mp3s/")
    
    audio1_path = get_audio_path(track1_name, use_htdemucs, htdemucs_stem)
    audio2_path = get_audio_path(track2_name, use_htdemucs, htdemucs_stem)
    
    # Step 3: If using htdemucs with 'all' stems, mix them together first
    if use_htdemucs and htdemucs_stem == 'all':
        logger.info("\n🎼 Mixing htdemucs stems...")
        
        # Mix track 1 stems
        audio1_data, sr1 = mix_htdemucs_stems(audio1_path, exclude_stems)
        temp_audio1 = Path('/tmp') / f"{track1_name}_mixed.wav"
        sf.write(str(temp_audio1), audio1_data, sr1)
        audio1_path = str(temp_audio1)
        logger.info(f"   Track 1 mixed stems saved to: {audio1_path}")
        
        # Mix track 2 stems
        audio2_data, sr2 = mix_htdemucs_stems(audio2_path, exclude_stems)
        temp_audio2 = Path('/tmp') / f"{track2_name}_mixed.wav"
        sf.write(str(temp_audio2), audio2_data, sr2)
        audio2_path = str(temp_audio2)
        logger.info(f"   Track 2 mixed stems saved to: {audio2_path}")
    
    # Step 4: Initialize DJ algorithm
    logger.info("\n🤖 Initializing DJ algorithm...")
    dj = DJMixingAlgorithm()
    
    # Step 5: Analyze tracks
    logger.info("\n🔍 Analyzing tracks...")
    logger.info("   (This includes semantic analysis, energy, mood, etc.)")
    
    analysis1 = dj.analyze_track(metadata1)  # type: ignore[arg-type]
    logger.info(f"   ✓ Track 1: BPM={analysis1.bpm:.1f}, Key={analysis1.key}")
    
    analysis2 = dj.analyze_track(metadata2)  # type: ignore[arg-type]
    logger.info(f"   ✓ Track 2: BPM={analysis2.bpm:.1f}, Key={analysis2.key}")
    
    # Step 6: Calculate compatibility
    logger.info("\n🎯 Calculating compatibility...")
    try:
        transition_score = dj.calculate_transition_score(analysis1, analysis2)
        
        # The function returns keys: 'overall_score', 'beat_match', 'key_compatibility', etc.
        # NOT 'total_score', 'harmonic_score', etc.
        
        # Check if we got a valid result
        if transition_score is None or (isinstance(transition_score, dict) and not transition_score):
            logger.warning("   ⚠️  calculate_transition_score returned None/empty, using fallback calculation")
            transition_score = None
        
        # If the library function failed, calculate basic compatibility ourselves
        if transition_score is None:
            # Basic BPM compatibility (closer BPMs = better)
            bpm_diff = abs(analysis1.bpm - analysis2.bpm)
            bpm_score = max(0, 100 - (bpm_diff * 5))  # 0-100 scale
            
            # Basic key compatibility
            key_score = 100.0 if analysis1.key == analysis2.key else 50.0
            
            # Average for total
            total_score = (bpm_score + key_score) / 2
            
            transition_score = {
                'overall_score': total_score,
                'beat_match': bpm_score,
                'key_compatibility': key_score,
                'energy_match': 70.0,
                'vocal_compatibility': 60.0,
                'genre_compatibility': 60.0,
                'semantic_similarity': 60.0
            }
            logger.info("   ℹ️  Using basic compatibility calculation")
        
        # Extract scores (use actual keys from dj_mixer.py)
        if isinstance(transition_score, dict):
            overall = transition_score.get('overall_score', 0.0)
            beat = transition_score.get('beat_match', 0.0)
            key_compat = transition_score.get('key_compatibility', 0.0)
            energy = transition_score.get('energy_match', 0.0)
            vocal = transition_score.get('vocal_compatibility', 0.0)
            genre = transition_score.get('genre_compatibility', 0.0)
            semantic = transition_score.get('semantic_similarity', 0.0)
        else:
            # If it's an object, try to get attributes
            overall = getattr(transition_score, 'overall_score', 0.0)
            beat = getattr(transition_score, 'beat_match', 0.0)
            key_compat = getattr(transition_score, 'key_compatibility', 0.0)
            energy = getattr(transition_score, 'energy_match', 0.0)
            vocal = getattr(transition_score, 'vocal_compatibility', 0.0)
            genre = getattr(transition_score, 'genre_compatibility', 0.0)
            semantic = getattr(transition_score, 'semantic_similarity', 0.0)
        
        # Convert 0-100 scale to 0-10 scale for display
        logger.info(f"   Overall Score: {overall/10:.1f}/10")
        logger.info(f"   - Beat Match: {beat/10:.1f}/10")
        logger.info(f"   - Key: {key_compat/10:.1f}/10")
        logger.info(f"   - Energy: {energy/10:.1f}/10")
        logger.info(f"   - Vocals: {vocal/10:.1f}/10")
        logger.info(f"   - Genre: {genre/10:.1f}/10")
        logger.info(f"   - Semantic: {semantic/10:.1f}/10")
        
    except Exception as e:
        logger.error(f"   ⚠️  Compatibility calculation failed: {e}")
        import traceback
        traceback.print_exc()
        # Use basic fallback
        bpm_diff = abs(analysis1.bpm - analysis2.bpm)
        bpm_score = max(0, 100 - (bpm_diff * 5))
        key_score = 100.0 if analysis1.key == analysis2.key else 50.0
        total_score = (bpm_score + key_score) / 2
        
        transition_score = {
            'overall_score': total_score,
            'beat_match': bpm_score,
            'key_compatibility': key_score,
            'energy_match': 70.0,
            'vocal_compatibility': 60.0,
            'genre_compatibility': 60.0,
            'semantic_similarity': 60.0
        }
        logger.info(f"   Overall Score: {total_score/10:.1f}/10 (fallback calculation)")
        logger.info(f"   - Beat Match: {bpm_score/10:.1f}/10")
        logger.info(f"   - Key: {key_score/10:.1f}/10")
    
    # Step 7: Generate mixing instructions
    logger.info("\n🎚️  Generating mixing instructions...")
    try:
        instructions = dj.generate_mixing_instructions(analysis1, analysis2)
        
        # Check if instructions is valid
        if instructions is None:
            logger.warning("   ⚠️  generate_mixing_instructions returned None, using fallback")
            instructions = None
        elif not hasattr(instructions, 'transition_point'):
            logger.warning("   ⚠️  instructions missing transition_point, using fallback")
            instructions = None
            
    except Exception as e:
        logger.warning(f"   ⚠️  generate_mixing_instructions failed: {e}, using fallback")
        import traceback
        traceback.print_exc()
        instructions = None
    
    # If library function failed, create our own simple instructions
    if instructions is None:
        # Create a simple instruction object matching the expected structure
        from transition_generator import TransitionPoint, TransitionType, EQCurve, MixingInstructions
        
        transition_point = TransitionPoint(
            track_a_out_time=120.0,  # Default values
            track_b_in_time=0.0,
            crossfade_duration=target_transition_duration,
            transition_type=TransitionType.BEAT_MATCHED,
            confidence=70.0
        )
        
        # Simple EQ curves
        eq_curve = EQCurve(
            timestamps=[0.0, target_transition_duration],
            bass=[1.0, 0.0],
            mid=[1.0, 0.5],
            high=[1.0, 1.0]
        )
        
        instructions = MixingInstructions(
            transition_point=transition_point,
            track_a_eq=eq_curve,
            track_b_eq=eq_curve,
            track_a_volume=[(0.0, 1.0), (target_transition_duration, 0.0)],
            track_b_volume=[(0.0, 0.0), (target_transition_duration, 1.0)],
            effects=[],
            notes="Simple crossfade"
        )
        logger.info("   ℹ️  Using simple crossfade instructions")
    
    # Safely log instruction details
    # MixingInstructions has: transition_point.transition_type and transition_point.crossfade_duration
    if hasattr(instructions, 'transition_point'):
        tp = instructions.transition_point
        if hasattr(tp, 'transition_type'):
            trans_type = tp.transition_type
            if hasattr(trans_type, 'value'):
                trans_type = trans_type.value
            logger.info(f"   Transition Type: {trans_type}")
        
        if hasattr(tp, 'crossfade_duration'):
            logger.info(f"   Duration: {tp.crossfade_duration:.1f}s")
    
    # Step 8: Render audio
    logger.info("\n🎼 Rendering audio mix...")
    mixer = AudioMixer()
    
    result = mixer.mix_two_tracks(
        audio1_path,
        audio2_path,
        instructions
    )
    
    logger.info(f"   Duration: {result.duration:.1f}s")
    logger.info(f"   Transition at: {result.transition_point:.1f}s")
    
    # Step 9: Save output
    logger.info(f"\n💾 Saving mix to: {output_path}")
    mixer.save_mix(result, output_path)
    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    logger.info(f"   ✓ Saved! ({file_size:.1f} MB)")
    logger.info(f"   🎧 You can now listen to your mix!")
    
    logger.info("\n" + "=" * 60)
    logger.info("✨ MIX COMPLETE!")
    logger.info("=" * 60)
    
    # Return summary
    # Convert overall_score from 0-100 to 0-10 scale for user-friendly display
    overall_score = transition_score.get('overall_score', 0.0) if isinstance(transition_score, dict) else 0.0
    
    # Extract transition info from instructions.transition_point
    if hasattr(instructions, 'transition_point'):
        tp = instructions.transition_point
        trans_type = getattr(getattr(tp, 'transition_type', None), 'value', 'crossfade') if hasattr(tp, 'transition_type') else 'crossfade'
        duration = getattr(tp, 'crossfade_duration', target_transition_duration)
    else:
        trans_type = 'crossfade'
        duration = target_transition_duration
    
    return {
        'output_path': output_path,
        'duration': result.duration,
        'transition_point': result.transition_point,
        'compatibility_score': overall_score / 10.0,  # Convert 0-100 to 0-10
        'track1': {
            'name': track1_name,
            'bpm': analysis1.bpm,
            'key': analysis1.key
        },
        'track2': {
            'name': track2_name,
            'bpm': analysis2.bpm,
            'key': analysis2.key
        },
        'instructions': {
            'type': trans_type,
            'duration': duration
        }
    }


def quick_test_mix(
    track1_name: str,
    track2_name: str,
    output_path: str = "quick_test_mix.wav",
    use_htdemucs: bool = False
):
    """
    Quick test mix without full AI analysis
    Just loads audio and does a simple crossfade
    
    Args:
        track1_name: Name of first track (e.g., "Dynamite")
        track2_name: Name of second track (e.g., "DJGotUsFallinInLove")
        output_path: Where to save output
        use_htdemucs: Use htdemucs processed audio (will mix all stems)
    """
    logger.info("🚀 Quick Test Mix (No AI Analysis)")
    
    audio1 = get_audio_path(track1_name, use_htdemucs, 'all')
    audio2 = get_audio_path(track2_name, use_htdemucs, 'all')
    
    # If using htdemucs, mix stems first
    if use_htdemucs:
        logger.info("Mixing htdemucs stems...")
        audio1_data, sr1 = mix_htdemucs_stems(audio1)
        temp_audio1 = Path('/tmp') / f"{track1_name}_quick.wav"
        sf.write(str(temp_audio1), audio1_data, sr1)
        audio1 = str(temp_audio1)
        
        audio2_data, sr2 = mix_htdemucs_stems(audio2)
        temp_audio2 = Path('/tmp') / f"{track2_name}_quick.wav"
        sf.write(str(temp_audio2), audio2_data, sr2)
        audio2 = str(temp_audio2)
    
    logger.info(f"   Track 1: {audio1}")
    logger.info(f"   Track 2: {audio2}")
    logger.info(f"   Output: {output_path}")
    
    quick_mix(audio1, audio2, output_path)
    
    logger.info("✨ Quick mix complete!")


def batch_create_mixes(
    track_names: List[str],
    output_dir: str = "batch_mixes",
    use_htdemucs: bool = False,
    htdemucs_stem: str = 'all',
    exclude_stems: Optional[List[str]] = None
):
    """
    Create mixes for multiple track pairs
    
    Args:
        track_names: List of track names (e.g., ["Dynamite", "DJGotUsFallinInLove"])
        output_dir: Directory to save outputs
        use_htdemucs: Use htdemucs stems
        htdemucs_stem: Which stem to use
        exclude_stems: Stems to exclude
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    logger.info(f"🎵 Batch Mix Creation: {len(track_names)} tracks")
    logger.info(f"   Output directory: {output_dir}")
    
    results = []
    
    for i in range(len(track_names) - 1):
        track1 = track_names[i]
        track2 = track_names[i + 1]
        
        output_file = output_path / f"mix_{i+1:02d}_{track1}_to_{track2}.wav"
        
        logger.info(f"\n--- Mix {i+1}/{len(track_names)-1} ---")
        
        try:
            result = create_dj_mix_from_json(
                track1,
                track2,
                output_path=str(output_file),
                use_htdemucs=use_htdemucs,
                htdemucs_stem=htdemucs_stem,
                exclude_stems=exclude_stems
            )
            results.append(result)
            logger.info("✅ Success!")
            
        except Exception as e:
            logger.error(f"❌ Failed: {e}")
            results.append(None)
    
    # Summary
    successful = sum(1 for r in results if r is not None)
    logger.info(f"\n🎉 Batch Complete: {successful}/{len(results)} successful")
    
    return results


# Example usage
if __name__ == "__main__":
    """
    EXAMPLES FOR YOUR EXACT FOLDER STRUCTURE:
    
    data/
    ├── htdemucs/
    │   ├── Dynamite/
    │   │   ├── bass.wav
    │   │   ├── drums.wav
    │   │   ├── other.wav
    │   │   └── vocals.wav
    ├── metadata/
    │   └── Dynamite.json
    └── mp3s/
        └── Dynamite.mp3
    """
    
    # ===== EXAMPLE 1: Basic mix using mp3s =====
    create_dj_mix_from_json(
        track1_name="Dynamite",
        track2_name="DJGotUsFallinInLove",
        output_path="mix_from_mp3s.wav",
        use_htdemucs=False  # Use data/mp3s/
    )
    
    # ===== EXAMPLE 2: Mix using htdemucs (all stems) =====
    # create_dj_mix_from_json(
    #     track1_name="Dynamite",
    #     track2_name="DJGotUsFallinInLove",
    #     output_path="mix_from_htdemucs.wav",
    #     use_htdemucs=True,  # Use data/htdemucs/
    #     htdemucs_stem='all'  # Mix all stems together
    # )
    
    # ===== EXAMPLE 3: Instrumental mix (exclude vocals) =====
    # create_dj_mix_from_json(
    #     track1_name="Dynamite",
    #     track2_name="DJGotUsFallinInLove",
    #     output_path="instrumental_mix.wav",
    #     use_htdemucs=True,
    #     htdemucs_stem='all',
    #     exclude_stems=['vocals']  # Only use bass, drums, other
    # )
    
    # ===== EXAMPLE 4: Mix only specific stem (e.g., just bass) =====
    # create_dj_mix_from_json(
    #     track1_name="Dynamite",
    #     track2_name="DJGotUsFallinInLove",
    #     output_path="bass_only_mix.wav",
    #     use_htdemucs=True,
    #     htdemucs_stem='bass'  # Only bass.wav
    # )
    
    # ===== EXAMPLE 5: Quick test without AI analysis =====
    # quick_test_mix(
    #     "Dynamite",
    #     "DJGotUsFallinInLove",
    #     "quick_test.wav",
    #     use_htdemucs=False
    # )
    
    # ===== EXAMPLE 6: Batch process multiple tracks =====
    # track_names = ["Dynamite", "DJGotUsFallinInLove", "Track3", "Track4"]
    # batch_create_mixes(
    #     track_names,
    #     output_dir="my_dj_set",
    #     use_htdemucs=False
    # )