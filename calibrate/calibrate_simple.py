# python -m calibrate.calibrate_simple

import librosa
import numpy as np
import json
import os
import subprocess

def estimate_key(y, sr):
    """Estimate musical key using Krumhansl-Schmuckler profiles"""
    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        chroma_avg = chroma.mean(axis=1)
        if chroma_avg.sum() > 0:
            chroma_avg = chroma_avg / chroma_avg.sum()
        
        best_correlation = -1
        best_key = 'C'
        best_mode = 'major'
        
        for shift in range(12):
            shifted_major = np.roll(major_profile, shift)
            correlation = np.corrcoef(chroma_avg, shifted_major)[0, 1]
            if not np.isnan(correlation) and correlation > best_correlation:
                best_correlation = correlation
                best_key = keys[shift]
                best_mode = 'major'
            
            shifted_minor = np.roll(minor_profile, shift)
            correlation = np.corrcoef(chroma_avg, shifted_minor)[0, 1]
            if not np.isnan(correlation) and correlation > best_correlation:
                best_correlation = correlation
                best_key = keys[shift]
                best_mode = 'minor'
        
        return f"{best_key} {best_mode}"
    except:
        return "C major"

def analyze_simple(file_path):
    """Simple analysis - BPM, key, duration only"""
    print(f"🎧 Analyzing: {os.path.basename(file_path)}")
    
    try:
        # Load audio
        y, sr = librosa.load(file_path, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        print(f"📊 Loaded: {duration:.1f}s at {sr}Hz")
        
        # Get BPM
        try:
            bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
            bpm = max(60, min(200, int(round(float(bpm)))))
        except Exception as e:
            print(f"⚠️ BPM detection failed: {e}, using 120")
            bpm = 120
        
        # Get key
        key = estimate_key(y, sr)
        
        print(f"🎵 Features: {bpm} BPM, {key}")
        
        return {
            "bpm": bpm,
            "key": key,
            "duration": round(float(duration), 2)
        }
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None

def split_stems_simple(file_path):
    """Split audio into stems using Demucs - MP3 output"""
    song_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Clean up song name
    clean_song_name = "".join(c for c in song_name if c.isalnum() or c in (' ', '-', '_')).strip()
    clean_song_name = clean_song_name.replace(' ', '_')
    
    print(f"🎵 Processing: {song_name}")
    print(f"🔧 Clean name: {clean_song_name}")
    
    # Expected stem folder patterns (handle both layouts)
    possible_folders = [
        os.path.join("data", "htdemucs", song_name),
        os.path.join("data", "htdemucs", clean_song_name),
        os.path.join("data", "separated", "htdemucs", song_name),
        os.path.join("data", "separated", "htdemucs", clean_song_name),
    ]
    
    # Check if stems already exist
    for folder in possible_folders:
        if os.path.exists(folder) and has_all_stems(folder):
            print(f"✅ Found existing stems: {folder}")
            return folder, song_name
    
    # Need to split
    print("🎛️ Running Demucs to split stems...")
    os.makedirs("data", exist_ok=True)
    
    try:
        # Run Demucs with MP3 output
        result = subprocess.run([
            "demucs",
            "--mp3",  # Use MP3 format (smaller files)
            "--mp3-bitrate", "320",  # High quality
            "-n", "htdemucs",
            "--out", "data",
            file_path
        ], check=True, capture_output=True, text=True)
        
        print("✅ Demucs separation completed")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Demucs failed: {e}")
        print(f"stderr: {e.stderr}")
        
        # Try alternative approach (fallback to WAV)
        try:
            print("🔄 Trying alternative Demucs command (WAV format)...")
            subprocess.run([
                "demucs",
                "-n", "htdemucs",
                "--out", "data",
                file_path
            ], check=True, capture_output=True, text=True)
            print("✅ Alternative Demucs succeeded")
        except:
            print("❌ Both Demucs methods failed")
            return None, song_name
    
    # Find the created folder
    stem_folder = find_stem_folder("data", song_name, clean_song_name)
    
    if stem_folder and has_all_stems(stem_folder):
        print(f"✅ Stems ready: {stem_folder}")
        return stem_folder, song_name
    else:
        print(f"❌ Could not find or create stems for: {song_name}")
        return None, song_name

def find_stem_folder(data_dir, original_name, clean_name):
    """Find the actual folder created by Demucs, matching the current song"""
    base_dirs = [
        os.path.join(data_dir, "htdemucs"),                 # when using --out data
        os.path.join(data_dir, "separated", "htdemucs"),    # default demucs layout
        os.path.join(data_dir, "seperated", "htdemucs"),    # safety for misspelling
    ]

    potential_names = [original_name.lower(), clean_name.lower()]

    for htdemucs_dir in base_dirs:
        if not os.path.isdir(htdemucs_dir):
            continue
        print(f"🔍 Checking htdemucs directory: {htdemucs_dir}")

        for folder_name in os.listdir(htdemucs_dir):
            folder_path = os.path.join(htdemucs_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            # ✅ 1. Require the folder name to match this song closely
            if any(name == folder_name.lower() for name in potential_names):
                if has_all_stems(folder_path):
                    print(f"✅ Found exact match for current song: {folder_path}")
                    return folder_path

        # ✅ 2. If no exact match, fall back to partial substring match
        for folder_name in os.listdir(htdemucs_dir):
            folder_path = os.path.join(htdemucs_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            if any(name in folder_name.lower() or folder_name.lower() in name
                   for name in potential_names):
                if has_all_stems(folder_path):
                    print(f"✅ Found partial match: {folder_path}")
                    return folder_path

    print("❌ No valid stem folder found for this song")
    return None

# def find_stem_folder(data_dir, original_name, clean_name):
#     """Find the actual folder created by Demucs"""
#     htdemucs_dir = os.path.join(data_dir, "separated", "htdemucs")
    
#     if not os.path.exists(htdemucs_dir):
#         print(f"❌ No htdemucs directory found: {htdemucs_dir}")
#         return None
    
#     print(f"🔍 Checking htdemucs directory...")
    
#     potential_names = [original_name, clean_name]
    
#     for folder_name in os.listdir(htdemucs_dir):
#         folder_path = os.path.join(htdemucs_dir, folder_name)
        
#         if os.path.isdir(folder_path):
#             print(f"   Found folder: {folder_name}")
            
#             if has_all_stems(folder_path):
#                 print(f"✅ Found valid stem folder: {folder_path}")
#                 return folder_path
            
#             for potential_name in potential_names:
#                 if (potential_name.lower() in folder_name.lower() or 
#                     folder_name.lower() in potential_name.lower()):
#                     if has_all_stems(folder_path):
#                         print(f"✅ Found matching stem folder: {folder_path}")
#                         return folder_path
    
#     print("❌ No valid stem folder found in htdemucs")
#     return None

def has_all_stems(folder_path):
    """Check if folder contains all required stem files (WAV or MP3)"""
    required_stems = ["vocals", "drums", "bass", "other"]
    
    found_count = 0
    for stem in required_stems:
        # Check for both .mp3 and .wav
        mp3_path = os.path.join(folder_path, f"{stem}.mp3")
        wav_path = os.path.join(folder_path, f"{stem}.wav")
        
        if os.path.exists(mp3_path) or os.path.exists(wav_path):
            found_count += 1
        else:
            print(f"   Missing: {stem}")
            return False
    
    if found_count == 4:
        print(f"   ✅ All stems present in: {folder_path}")
        return True
    return False

def get_stem_paths(stem_dir):
    """Return dict of stem_name -> relative_path, all MP3 or all WAV"""
    if not stem_dir or not os.path.exists(stem_dir):
        return None

    # Check which format this folder uses (MP3 preferred)
    mp3_exists = all(os.path.exists(os.path.join(stem_dir, f"{stem}.mp3"))
                     for stem in ["vocals", "drums", "bass", "other"])
    wav_exists = all(os.path.exists(os.path.join(stem_dir, f"{stem}.wav"))
                     for stem in ["vocals", "drums", "bass", "other"])

    if mp3_exists:
        ext = ".mp3"
    elif wav_exists:
        ext = ".wav"
    else:
        print(f"❌ No complete set of MP3 or WAV stems found in {stem_dir}")
        return None

    # --- compute relative paths instead of absolute ones ---
    project_root = os.path.abspath(".")  # or use 'data' if you want paths relative to data/
    stems = {
        stem: os.path.relpath(os.path.join(stem_dir, f"{stem}{ext}"), start=project_root)
        for stem in ["vocals", "drums", "bass", "other"]
    }

    return stems

# def get_stem_paths(stem_dir):
#     """Get absolute paths to all stem files"""
#     if not stem_dir or not os.path.exists(stem_dir):
#         return None
    
#     stems = {}
#     for stem_name in ["vocals", "drums", "bass", "other"]:
#         # Try MP3 first, then WAV
#         mp3_path = os.path.join(stem_dir, f"{stem_name}.mp3")
#         wav_path = os.path.join(stem_dir, f"{stem_name}.wav")
        
#         if os.path.exists(mp3_path):
#             stems[stem_name] = os.path.abspath(mp3_path)
#         elif os.path.exists(wav_path):
#             stems[stem_name] = os.path.abspath(wav_path)
    
#     return stems if stems else None
    # if stems:
    #     return {
    #         "folder": os.path.abspath(stem_dir),
    #         "stems": stems
    #     }
    # return None

def calibrate_simple(file_path, skip_stems=False):
    """Simple calibration - BPM, key, and stems only"""
    
    # Get song name from file path
    song_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Step 1: Analyze
    metadata = analyze_simple(file_path)
    metadata["stems"] = {}

    
    if not metadata:
        print("❌ Analysis failed")
        return None
    
    # Step 2: Split stems (optional)
    # if not skip_stems:
    #     stem_dir, actual_song_name = split_stems_simple(file_path)
        
    #     # Use the actual song name from stem separation if it worked
    #     if actual_song_name:
    #         song_name = actual_song_name
        
    #     if stem_dir:
    #         stem_info = get_stem_paths(stem_dir)
    #         if stem_info:
    #             metadata["stem_paths"] = stem_info
    # else:
    #     print("⏭️ Skipping stem separation")
    if not skip_stems:
        stem_dir, actual_song_name = split_stems_simple(file_path)
        
        if actual_song_name:
            song_name = actual_song_name
        
        if stem_dir:
            stems = get_stem_paths(stem_dir)
            if stems:
                metadata["stems"] = stems
    else:
        print("⏭️ Skipping stem separation")
    
    # Step 3: Save metadata (use song_name, NOT None)
    out_path = os.path.join("data", "metadata", f"{song_name}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Saved metadata to: {out_path}")
    
    # Step 4: Display summary
    print("\n" + "="*50)
    print("SIMPLE ANALYSIS SUMMARY")
    print("="*50)
    print(f"Song: {song_name}")
    print(f"BPM: {metadata['bpm']}")
    print(f"Key: {metadata['key']}")
    print(f"Duration: {metadata['duration']}s")
    
    if "stem_paths" in metadata:
        print(f"\n📁 Stems ({len(metadata['stem_paths']['stems'])} files):")
        for stem_name, stem_path in metadata['stem_paths']['stems'].items():
            if os.path.exists(stem_path):
                file_size = os.path.getsize(stem_path) / (1024*1024)  # MB
                ext = os.path.splitext(stem_path)[1]
                print(f"  • {stem_name}: {os.path.basename(stem_path)} ({file_size:.1f} MB)")
    
    print("="*50)
    
    return metadata

if __name__ == "__main__":
    import sys
    
    # Check for flags
    skip_stems = "--no-stems" in sys.argv
    
    # Get file path (filter out flags)
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    
    if len(args) > 0:
        # File path provided
        file_path = args[0]
    else:
        # Use file picker
        try:
            from tkinter import filedialog, Tk
            root = Tk()
            root.withdraw()
            root.update()
            file_path = filedialog.askopenfilename(
                initialdir=os.path.join("data", "mp3s"),
                title="Pick a song",
                filetypes=[("Audio Files", "*.mp3 *.wav")]
            )
            root.destroy()
        except:
            print("❌ No tkinter available")
            print("Usage: python -m calibrate.calibrate_simple <file_path>")
            print("       python -m calibrate.calibrate_simple <file_path> --no-stems")
            sys.exit(1)
    
    if file_path:
        calibrate_simple(file_path, skip_stems=skip_stems)
    else:
        print("❌ No file selected")
