"""
CLAP Classification Module
Uses CLAP (Contrastive Language-Audio Pretraining) for semantic audio analysis
Detects vocals, genre, mood, and other musical characteristics
"""

import torch
import numpy as np
from transformers import ClapModel, ClapProcessor
from typing import Dict, List, Tuple, Optional, Any
import librosa


class CLAPClassifier:
    """
    Semantic audio classifier using CLAP
    
    DJ Context: Technical features (BPM, key) aren't enough. We need to understand:
    - Vocal presence (to avoid vocal clashing)
    - Genre/style (to maintain set coherence)
    - Mood/energy (for emotional flow)
    """
    
    def __init__(self, model_name: str = "laion/larger_clap_music_and_speech"):
        """
        Initialize CLAP model
        
        Args:
            model_name: Hugging Face model identifier
        """
        print(f"Loading CLAP model: {model_name}")
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)
        self.model = ClapModel.from_pretrained(model_name).to(self.device)
        self.processor = ClapProcessor.from_pretrained(model_name)
        print(f"Model loaded on {self.device}")
        
        # Prompt engineering: Carefully crafted labels for zero-shot classification
        self._setup_classification_labels()
    
    def _setup_classification_labels(self):
        """
        Setup classification labels for different aspects of music
        
        Prompt Engineering Strategy:
        - Use descriptive, multi-word phrases (not single words)
        - Include musical terminology
        - Cover spectrum of possibilities
        - Test and refine based on results
        """
        
        # Vocal Detection Labels
        # DJ Context: Critical for avoiding vocal clashing
        self.vocal_labels = [
            "song with prominent lead vocals and singing throughout the track",
            "music with occasional vocals and singing in some sections",
            "mostly instrumental music with brief vocal samples or chops",
            "purely instrumental music without any vocals or singing",
        ]
        
        # Genre/Style Labels  
        # DJ Context: Maintain genre coherence, avoid jarring style shifts
        self.genre_labels = [
            "electronic dance music house techno edm club music",
            "hip hop rap urban music with beats and rhymes",
            "rock music with guitars drums and live instruments",
            "pop music mainstream commercial radio friendly",
            "reggaeton latin music with dembow rhythm",
            "ambient chill music downtempo relaxing atmospheric",
            "drum and bass jungle fast breakbeats electronic",
            "dubstep bass music with wobbles and drops",
            "trance music euphoric uplifting melodic electronic",
            "techno music repetitive electronic four on the floor",
        ]
        
        # Energy/Mood Labels
        # DJ Context: Manage energy progression throughout set
        self.energy_labels = [
            "very high energy upbeat euphoric exciting intense music",
            "medium energy balanced steady moderate tempo music",
            "low energy calm mellow relaxing slow tempo music",
        ]
        
        # Mood/Atmosphere Labels
        self.mood_labels = [
            "dark moody atmospheric mysterious brooding music",
            "bright happy uplifting cheerful positive music",
            "aggressive intense powerful forceful heavy music",
            "emotional melodic beautiful touching sentimental music",
        ]
        
        # Instrumentation Density Labels
        # DJ Context: Helps identify good transition points (minimal instrumentation)
        self.density_labels = [
            "dense busy music with many instruments and layers",
            "moderate instrumentation with balanced arrangement",
            "minimal sparse music with few instruments",
        ]
    
    def load_audio_for_clap(self, file_path: str, duration: float = 30.0) -> np.ndarray:
        """
        Load audio optimized for CLAP
        
        Strategy: Analyze a representative segment (30 seconds from middle)
        to avoid intros/outros that may not represent the track's character
        
        Args:
            file_path: Path to audio file
            duration: Seconds to analyze (CLAP works well with 10-30 second clips)
            
        Returns:
            Audio array suitable for CLAP
        """
        # Load full track to find middle section
        y, sr = librosa.load(file_path, sr=48000, mono=True)  # CLAP expects 48kHz
        
        # Extract middle section
        total_duration = len(y) / sr
        if total_duration > duration:
            start_time = (total_duration - duration) / 2
            start_sample = int(start_time * sr)
            end_sample = start_sample + int(duration * sr)
            y = y[start_sample:end_sample]
        
        return y
    
    def classify_vocals(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Classify vocal presence and prominence
        
        DJ Context: Never mix two tracks with heavy vocals during vocal sections.
        This creates an amateur, cluttered sound. We need to detect vocal levels
        to plan transitions during instrumental breaks.
        
        Returns:
            Dictionary with vocal level classification and confidence
        """
        # Process audio
        inputs = self.processor(
            audios=audio,
            text=self.vocal_labels,
            return_tensors="pt",
            padding=True,
            sampling_rate=48000
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_audio = outputs.logits_per_audio
            probs = logits_per_audio.softmax(dim=-1).cpu().numpy()[0]
        
        # Map to descriptive categories
        categories = ["heavy_vocals", "moderate_vocals", "minimal_vocals", "instrumental"]
        results = {cat: float(prob) for cat, prob in zip(categories, probs)}
        
        # Determine primary classification
        primary = categories[np.argmax(probs)]
        confidence = float(np.max(probs))
        
        return {
            'classification': primary,
            'confidence': confidence,
            'scores': results
        }
    
    def classify_genre(self, audio: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Classify genre/style
        
        DJ Context: Maintain genre coherence. Sudden shifts from house to rock
        sound jarring unless intentional. We identify primary genres to score
        compatibility between tracks.
        
        Returns:
            List of (genre, confidence) tuples, sorted by confidence
        """
        inputs = self.processor(
            audios=audio,
            text=self.genre_labels,
            return_tensors="pt",
            padding=True,
            sampling_rate=48000
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_audio = outputs.logits_per_audio
            probs = logits_per_audio.softmax(dim=-1).cpu().numpy()[0]
        
        # Simplified genre names
        genre_names = [
            "house_techno",
            "hip_hop",
            "rock",
            "pop",
            "reggaeton",
            "ambient",
            "dnb",
            "dubstep",
            "trance",
            "techno"
        ]
        
        # Sort by confidence
        genre_scores = [(name, float(prob)) for name, prob in zip(genre_names, probs)]
        genre_scores.sort(key=lambda x: x[1], reverse=True)
        
        return genre_scores[:top_k]
    
    def classify_energy(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Classify energy level
        
        DJ Context: Energy management is the art of DJing. Build energy gradually,
        create peaks and valleys, give the crowd moments to breathe.
        
        Returns:
            Energy classification and score
        """
        inputs = self.processor(
            audios=audio,
            text=self.energy_labels,
            return_tensors="pt",
            padding=True,
            sampling_rate=48000
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_audio = outputs.logits_per_audio
            probs = logits_per_audio.softmax(dim=-1).cpu().numpy()[0]
        
        energy_levels = ["high", "medium", "low"]
        
        # Calculate weighted energy score (0-100)
        energy_score = probs[0] * 100 + probs[1] * 50 + probs[2] * 0
        
        return {
            'level': energy_levels[np.argmax(probs)],
            'score': float(energy_score),
            'confidence': float(np.max(probs)),
            'distribution': {level: float(prob) for level, prob in zip(energy_levels, probs)}
        }
    
    def classify_mood(self, audio: np.ndarray) -> List[Tuple[str, float]]:
        """
        Classify mood/atmosphere
        
        DJ Context: Mood consistency creates cohesive sets. Avoid jarring
        emotional shifts unless building to a dramatic moment.
        
        Returns:
            List of (mood, confidence) tuples
        """
        inputs = self.processor(
            audios=audio,
            text=self.mood_labels,
            return_tensors="pt",
            padding=True,
            sampling_rate=48000
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_audio = outputs.logits_per_audio
            probs = logits_per_audio.softmax(dim=-1).cpu().numpy()[0]
        
        moods = ["dark", "bright", "aggressive", "emotional"]
        mood_scores = [(mood, float(prob)) for mood, prob in zip(moods, probs)]
        mood_scores.sort(key=lambda x: x[1], reverse=True)
        
        return mood_scores
    
    def classify_density(self, audio: np.ndarray) -> str:
        """
        Classify instrumentation density
        
        DJ Context: Minimal sections (just drums, or no vocals) are ideal
        for transitions. Dense, busy sections can clash.
        
        Returns:
            Density classification
        """
        inputs = self.processor(
            audios=audio,
            text=self.density_labels,
            return_tensors="pt",
            padding=True,
            sampling_rate=48000
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_audio = outputs.logits_per_audio
            probs = logits_per_audio.softmax(dim=-1).cpu().numpy()[0]
        
        densities = ["dense", "moderate", "minimal"]
        return densities[np.argmax(probs)]
    
    def get_audio_embedding(self, audio: np.ndarray) -> np.ndarray:
        """
        Get CLAP audio embedding for similarity comparison
        
        DJ Context: Find similar-sounding tracks even if they're different genres.
        Useful for smart playlist generation.
        
        Returns:
            Audio embedding vector
        """
        inputs = self.processor(
            audios=audio,
            return_tensors="pt",
            sampling_rate=48000
        ).to(self.device)
        
        with torch.no_grad():
            audio_embed = self.model.get_audio_features(**inputs)
            # Normalize
            audio_embed = audio_embed / audio_embed.norm(dim=-1, keepdim=True)
        
        return audio_embed.cpu().numpy()[0]
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two audio embeddings
        
        Returns:
            Similarity score 0-100
        """
        similarity = np.dot(embedding1, embedding2)
        # Convert from [-1, 1] to [0, 100]
        return float((similarity + 1) * 50)
    
    def analyze_track_comprehensive(self, file_path: str) -> Dict[str, Any]:
        """
        Run all CLAP analyses on a track
        
        Returns:
            Complete semantic analysis
        """
        print(f"Analyzing: {file_path}")
        
        # Load audio
        audio = self.load_audio_for_clap(file_path)
        
        # Run all classifications
        results = {
            'vocals': self.classify_vocals(audio),
            'genres': self.classify_genre(audio),
            'energy': self.classify_energy(audio),
            'mood': self.classify_mood(audio),
            'density': self.classify_density(audio),
            'embedding': self.get_audio_embedding(audio)
        }
        
        return results


def calculate_vocal_clash_penalty(track_a_vocals: Dict, track_b_vocals: Dict) -> float:
    """
    Calculate penalty for vocal clashing
    
    DJ Context: THE CARDINAL SIN of DJing is mixing two vocal tracks.
    Heavy penalty when both tracks have prominent vocals.
    
    Returns:
        Penalty score (0 = no penalty, 100 = maximum penalty)
    """
    a_vocal_score = track_a_vocals['scores']['heavy_vocals'] + track_a_vocals['scores']['moderate_vocals'] * 0.5
    b_vocal_score = track_b_vocals['scores']['heavy_vocals'] + track_b_vocals['scores']['moderate_vocals'] * 0.5
    
    # Penalty is product of vocal scores
    penalty = a_vocal_score * b_vocal_score * 100
    
    return float(penalty)


def calculate_genre_compatibility(genres_a: List[Tuple[str, float]], 
                                 genres_b: List[Tuple[str, float]]) -> float:
    """
    Calculate genre compatibility score
    
    DJ Context: Genres should either match or be complementary. Electronic
    subgenres often mix well together. Rock and techno? Usually not.
    
    Returns:
        Compatibility score 0-100
    """
    # Extract top genres
    top_a = {genre: score for genre, score in genres_a[:2]}
    top_b = {genre: score for genre, score in genres_b[:2]}
    
    # Check for exact matches
    common_genres = set(top_a.keys()) & set(top_b.keys())
    if common_genres:
        return 100.0
    
    # Define genre compatibility groups
    electronic_genres = {"house_techno", "techno", "trance", "dubstep", "dnb"}
    urban_genres = {"hip_hop", "reggaeton"}
    chill_genres = {"ambient", "pop"}
    
    def get_genre_group(genres):
        for genre in genres:
            if genre in electronic_genres:
                return "electronic"
            if genre in urban_genres:
                return "urban"
            if genre in chill_genres:
                return "chill"
        return "other"
    
    group_a = get_genre_group(top_a.keys())
    group_b = get_genre_group(top_b.keys())
    
    # Same group = compatible
    if group_a == group_b:
        return 80.0
    
    # Some groups mix better than others
    compatible_pairs = {
        ("electronic", "chill"),
        ("chill", "electronic"),
        ("urban", "electronic")
    }
    
    if (group_a, group_b) in compatible_pairs:
        return 60.0
    
    return 30.0  # Low compatibility


# Example usage
if __name__ == "__main__":
    print("CLAP Classifier Module - Test")
    print("="*50)
    print("This module uses CLAP for semantic audio understanding")
    print("\nClassification Categories:")
    print("✓ Vocal presence (to avoid vocal clashing)")
    print("✓ Genre/style (for set coherence)")
    print("✓ Energy level (for flow management)")
    print("✓ Mood/atmosphere (for emotional continuity)")
    print("✓ Instrumentation density (for transition planning)")
    print("\nNote: Requires audio files to run actual classification")