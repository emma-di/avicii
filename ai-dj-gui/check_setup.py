#!/usr/bin/env python3
"""
Quick setup and test script for AI DJ GUI
Run this to verify everything is working
"""

import os
import sys
from pathlib import Path

def check_directories():
    """Check if required directories exist"""
    print("📁 Checking directories...")
    
    required_dirs = [
        'data/mp3s',
        'data/metadata',
        'data/separated',
        'data/mixes',
        'static/css',
        'static/images',
        'templates'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"   ✅ {dir_path}")
        else:
            print(f"   ❌ {dir_path} - creating...")
            os.makedirs(dir_path, exist_ok=True)
            all_exist = False
    
    return all_exist

def check_files():
    """Check if required files exist"""
    print("\n📄 Checking files...")
    
    required_files = [
        'app.py',
        'static/css/styles.css',
        'static/images/purple_disc.png',
        'static/images/cyan_disc.png',
        'templates/home.html',
        'templates/new_playlist.html',
        'templates/library.html',
        'templates/loading.html',
        'templates/remix.html'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING!")
            all_exist = False
    
    return all_exist

def check_dependencies():
    """Check if required Python packages are installed"""
    print("\n📦 Checking dependencies...")
    
    dependencies = {
        'flask': 'Flask',
        'librosa': 'librosa',
        'numpy': 'numpy',
        'soundfile': 'soundfile'
    }
    
    all_installed = True
    for module_name, package_name in dependencies.items():
        try:
            __import__(module_name)
            print(f"   ✅ {package_name}")
        except ImportError:
            print(f"   ❌ {package_name} - run: pip install {package_name}")
            all_installed = False
    
    return all_installed

def check_songs():
    """Check if any songs are available"""
    print("\n🎵 Checking for songs...")
    
    mp3_dir = Path('data/mp3s')
    if not mp3_dir.exists():
        print("   ❌ data/mp3s/ doesn't exist")
        return False
    
    songs = list(mp3_dir.glob('*.mp3'))
    if songs:
        print(f"   ✅ Found {len(songs)} songs:")
        for song in songs[:5]:  # Show first 5
            print(f"      • {song.name}")
        if len(songs) > 5:
            print(f"      ... and {len(songs) - 5} more")
        return True
    else:
        print("   ⚠️  No MP3 files found in data/mp3s/")
        print("      Add some songs to test the app!")
        return False

def check_metadata():
    """Check if songs have metadata"""
    print("\n📊 Checking metadata...")
    
    metadata_dir = Path('data/metadata')
    if not metadata_dir.exists():
        print("   ❌ data/metadata/ doesn't exist")
        return False
    
    metadata_files = list(metadata_dir.glob('*.json'))
    if metadata_files:
        print(f"   ✅ Found {len(metadata_files)} metadata files")
        return True
    else:
        print("   ⚠️  No metadata files found")
        print("      Run: python -m calibrate.calibrate_track")
        return False

def print_next_steps(has_songs, has_metadata):
    """Print next steps based on setup status"""
    print("\n" + "="*60)
    print("📋 NEXT STEPS")
    print("="*60)
    
    if not has_songs:
        print("\n1. Add MP3 files:")
        print("   cp /path/to/your/songs/*.mp3 data/mp3s/")
    
    if has_songs and not has_metadata:
        print("\n2. Analyze your songs:")
        print("   python -m calibrate.calibrate_track")
        print("   (Run this for each song, or batch process)")
    
    if has_songs and has_metadata:
        print("\n✅ You're ready to go!")
        print("\n🚀 Start the server:")
        print("   python app.py")
        print("\n🌐 Then open in browser:")
        print("   http://localhost:5000")
    else:
        print("\n3. After adding songs and metadata, start the server:")
        print("   python app.py")

def main():
    print("🎵 AI DJ - Setup Verification")
    print("="*60)
    
    # Run all checks
    dirs_ok = check_directories()
    files_ok = check_files()
    deps_ok = check_dependencies()
    has_songs = check_songs()
    has_metadata = check_metadata()
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    status = {
        'Directories': '✅' if dirs_ok else '⚠️',
        'Files': '✅' if files_ok else '❌',
        'Dependencies': '✅' if deps_ok else '❌',
        'Songs': '✅' if has_songs else '⚠️',
        'Metadata': '✅' if has_metadata else '⚠️'
    }
    
    for item, icon in status.items():
        print(f"   {icon} {item}")
    
    # Determine if ready
    ready_to_run = files_ok and deps_ok
    
    if ready_to_run and has_songs and has_metadata:
        print("\n🎉 Everything looks good!")
    elif ready_to_run:
        print("\n⚠️  App is set up, but you need songs/metadata")
    else:
        print("\n❌ Please fix the issues above first")
    
    print_next_steps(has_songs, has_metadata)
    print("\n" + "="*60)

if __name__ == '__main__':
    main()