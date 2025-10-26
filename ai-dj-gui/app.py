"""
Flask Backend API for AI DJ
Handles song loading and crossfading
"""

from flask import Flask, render_template, request, jsonify, session
import os
import sys
import json
import subprocess
from pathlib import Path

# Add calhacks directory to Python path so we can import modules
calhacks_path = os.path.abspath('..')  # Parent directory (calhacks2025)
if os.path.exists(calhacks_path):
    sys.path.insert(0, calhacks_path)
    print(f"✅ Added to Python path: {calhacks_path}")
else:
    print(f"⚠️  Warning: calhacks2025 directory not found at {calhacks_path}")

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production!

# Configuration - Point to data directory (sibling to ai-dj-gui)
MP3_DIR = "../data/mp3s"
METADATA_DIR = "../data/metadata"
OUTPUT_DIR = "../data/mixes"

# Ensure directories exist
os.makedirs(MP3_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.route('/')
def home():
    """Home page with two spinning vinyls"""
    return render_template('home.html')


@app.route('/new')
def new_playlist():
    """Song selection page (purple vinyl)"""
    # Scan for MP3 files
    songs = []
    
    # Friendly display names for known songs
    SONG_DISPLAY_NAMES = {
        'djgotusfallininl': {
            'title': 'DJ Got Us Fallin\' In Love',
            'artist': 'Usher ft. Pitbull',
            'bpm': 117,
            'key': 'G minor'
        },
        'djgotusfallininlove': {
            'title': 'DJ Got Us Fallin\' In Love',
            'artist': 'Usher ft. Pitbull',
            'bpm': 117,
            'key': 'G minor'
        },
        'dynamite': {
            'title': 'Dynamite',
            'artist': 'Taio Cruz',
            'bpm': 117,
            'key': 'E major'
        },
        'hotelroomservice': {
            'title': 'Hotel Room Service',
            'artist': 'Pitbull',
            'bpm': 123,
            'key': 'F# minor'
        },
        'hotel_room_service': {
            'title': 'Hotel Room Service',
            'artist': 'Pitbull',
            'bpm': 123,
            'key': 'F# minor'
        },
        'summer': {
            'title': 'Summer',
            'artist': 'Calvin Harris',
            'bpm': 129,
            'key': 'G major'
        }
    }
    
    print(f"\n📂 Looking for MP3s in: {MP3_DIR}")
    
    if not os.path.exists(MP3_DIR):
        print(f"⚠️  Warning: MP3 directory not found: {MP3_DIR}")
        print(f"   Please create it and add MP3 files!")
        print(f"   mkdir -p {MP3_DIR}")
        return render_template('new_playlist.html', songs=songs)
    
    # Scan directory for MP3 files
    mp3_files = [f for f in os.listdir(MP3_DIR) if f.endswith('.mp3')]
    
    if not mp3_files:
        print(f"⚠️  No MP3 files found in {MP3_DIR}")
        print(f"   Add some MP3s to get started!")
        return render_template('new_playlist.html', songs=songs)
    
    print(f"✅ Found {len(mp3_files)} MP3 file(s)")
    
    for file in mp3_files:
        song_name = os.path.splitext(file)[0].lower()  # Lowercase for matching
        
        # Check if we have a friendly display name
        if song_name in SONG_DISPLAY_NAMES:
            display_info = SONG_DISPLAY_NAMES[song_name]
            song_info = {
                'path': os.path.join(MP3_DIR, file),
                'title': display_info['title'],
                'artist': display_info['artist'],
                'bpm': display_info['bpm'],
                'key': display_info['key']
            }
            print(f"   ✓ {file} → {display_info['title']} ({display_info['bpm']} BPM, {display_info['key']})")
        else:
            # Fall back to metadata file or defaults
            metadata_path = os.path.join(METADATA_DIR, f"{song_name}.json")
            
            song_info = {
                'path': os.path.join(MP3_DIR, file),
                'title': song_name.replace('_', ' ').title(),
                'artist': 'Unknown Artist',
                'bpm': 120,
                'key': 'C major'
            }
            
            # Load metadata if exists
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        song_info['bpm'] = metadata.get('bpm', 120)
                        song_info['key'] = metadata.get('key', 'C major')
                        print(f"   ✓ {file} ({song_info['bpm']} BPM, {song_info['key']})")
                except Exception as e:
                    print(f"   ⚠️  Could not load metadata for {file}: {e}")
            else:
                print(f"   ⚠️  No display name or metadata for {file}")
        
        songs.append(song_info)
    
    print(f"✅ Loaded {len(songs)} songs into UI\n")
    
    return render_template('new_playlist.html', songs=songs)


@app.route('/library')
def library():
    """Playlist selection page (cyan vinyl)"""
    # TODO: Load saved playlists from database or JSON
    playlists = [
        {'name': 'Summer Vibes Mix', 'song_count': 5, 'duration': 18},
        {'name': 'Workout Playlist', 'song_count': 8, 'duration': 32},
        {'name': 'Chill Evening', 'song_count': 6, 'duration': 24},
    ]
    return render_template('library.html', playlists=playlists)


@app.route('/loading')
def loading():
    """Loading screen"""
    return render_template('loading.html')


@app.route('/api/get-current-mix')
def get_current_mix():
    """Serve the current mix audio file"""
    mix_path = session.get('current_mix_path')
    
    if not mix_path or not os.path.exists(mix_path):
        print(f"❌ Mix file not found: {mix_path}")
        return "Mix file not found", 404
    
    print(f"🎵 Serving mix: {mix_path}")
    
    # Serve the audio file
    from flask import send_file
    return send_file(
        mix_path,
        mimetype='audio/wav',
        as_attachment=False,
        download_name='current_mix.wav'
    )

@app.route('/remix')
def remix():
    """Live mixing interface (red page)"""
    # Get current mix info from session
    track1 = session.get('current_track_1', 'Track 1')
    track2 = session.get('current_track_2', 'Track 2')
    bpm = session.get('current_bpm', 128)
    key = session.get('current_key', 'C major')
    
    return render_template('remix.html', 
                         current_track_1=track1,
                         current_track_2=track2,
                         bpm=bpm,
                         key=key,
                         next_track='Loading...')


@app.route('/api/start-crossfade', methods=['POST'])
def start_crossfade():
    """
    Handle crossfade request from the UI
    This is where the magic happens!
    """
    data = request.json
    track1_path = data.get('track1')
    track2_path = data.get('track2')
    track1_info = data.get('track1_info', {})
    track2_info = data.get('track2_info', {})
    
    print(f"🎵 Starting crossfade:")
    print(f"   Track 1: {track1_path}")
    print(f"   Track 2: {track2_path}")
    
    # Store in session for the remix page
    session['current_track_1'] = track1_info.get('title', 'Track 1')
    session['current_track_2'] = track2_info.get('title', 'Track 2')
    session['current_bpm'] = track1_info.get('bpm', 128)
    session['current_key'] = track1_info.get('key', 'C major')
    session['track1_path'] = track1_path
    session['track2_path'] = track2_path
    
    # Run the crossfade in the background
    # For now, we'll just prepare it - you can trigger actual processing here
    try:
        # Example: Run the crossfade script
        output_name = f"{track1_info.get('title', 'track1')}_{track2_info.get('title', 'track2')}_mix.wav"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        
        # Store the output path for playback
        session['current_mix_path'] = output_path
        
        # Option 1: Run crossfade immediately (blocking - not recommended for production)
        # result = run_crossfade(track1_path, track2_path, output_path)
        
        # Option 2: Queue the job (recommended - use Celery or similar)
        # crossfade_task.delay(track1_path, track2_path, output_path)
        
        # For demo, just return success
        return jsonify({
            'success': True,
            'message': 'Crossfade started',
            'output': output_path
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/process-mix', methods=['POST'])
def process_mix():
    """
    Process the mix (called from loading screen)
    
    ⚠️ THIS IS WHERE THE CROSSFADE ACTUALLY RUNS! ⚠️
    """
    track1_path = session.get('track1_path')
    track2_path = session.get('track2_path')
    
    if not track1_path or not track2_path:
        return jsonify({'success': False, 'error': 'No tracks loaded'})
    
    try:
        output_path = session.get('current_mix_path')
        
        # Get song names for finding pre-made mix
        track1_name = os.path.splitext(os.path.basename(track1_path))[0].lower()
        track2_name = os.path.splitext(os.path.basename(track2_path))[0].lower()
        
        # Check if pre-made mix exists (for demo purposes)
        premade_mix = os.path.join(OUTPUT_DIR, f"{track1_name}_{track2_name}_mix.wav")
        
        if os.path.exists(premade_mix):
            print("=" * 60)
            print("🎵 USING PRE-MADE MIX (Demo Mode)")
            print(f"   Found: {premade_mix}")
            print("=" * 60)
            
            # Copy pre-made mix to output path
            import shutil
            shutil.copy(premade_mix, output_path)
            
            return jsonify({
                'success': True,
                'output': output_path,
                'mode': 'premade'
            })
        
        # Otherwise, run actual crossfade
        print("=" * 60)
        print("🎚️  RUNNING CROSSFADE ALGORITHM NOW...")
        print(f"   Track 1: {track1_path}")
        print(f"   Track 2: {track2_path}")
        print("=" * 60)
        
        # ⭐ THIS LINE RUNS YOUR crossfade_stems.py SCRIPT ⭐
        result = run_crossfade(track1_path, track2_path, output_path)
        
        print("=" * 60)
        print("✅ CROSSFADE COMPLETED!")
        print(f"   Output: {output_path}")
        print("=" * 60)
        
        return jsonify({
            'success': True,
            'output': output_path,
            'mode': 'crossfade'
        })
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ ERROR: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def run_crossfade(track1_path, track2_path, output_path, 
                  fade_beats=16, fade_curve='equal_power'):
    """
    Run the crossfade_stems.py script
    
    This is where the actual DJ mixing happens!
    """
    
    # Try multiple possible locations for crossfade script
    possible_paths = [
        '../dj_helpers/crossfade_stems.py',           # If in ai-dj-gui/
        'dj_helpers/crossfade_stems.py',              # If in calhacks2025/
        '../crossfade_stems.py',                      # If in root
        'crossfade_stems.py'                          # Same directory
    ]
    
    crossfade_script = None
    for path in possible_paths:
        if os.path.exists(path):
            crossfade_script = path
            print(f"✅ Found crossfade script at: {path}")
            break
    
    if not crossfade_script:
        # Try as module import
        print("⚠️  Crossfade script not found in expected locations")
        print("   Attempting to run as Python module...")
        cmd = [
            'python', '-m', 'dj_helpers.crossfade_stems',
            track1_path,
            track2_path,
            '--fade-beats', str(fade_beats),
            '--fade-curve', fade_curve,
            '--out-path', output_path,
            '--metadata-dir', METADATA_DIR
        ]
    else:
        # Run the script directly
        cmd = [
            'python', crossfade_script,
            track1_path,
            track2_path,
            '--fade-beats', str(fade_beats),
            '--fade-curve', fade_curve,
            '--out-path', output_path,
            '--metadata-dir', METADATA_DIR
        ]
    
    print("=" * 60)
    print(f"🎚️  Running crossfade command:")
    print(f"   {' '.join(cmd)}")
    print(f"   Working directory: {os.getcwd()}")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300  # 5 minute timeout
        )
        
        print(result.stdout)
        if result.stderr:
            print("⚠️  Warnings:", result.stderr)
        
        print("=" * 60)
        print("✅ Crossfade completed successfully!")
        print("=" * 60)
        
        return result
        
    except subprocess.TimeoutExpired:
        print("❌ Crossfade timed out (>5 minutes)")
        raise Exception("Crossfade processing timed out")
    except subprocess.CalledProcessError as e:
        print("=" * 60)
        print("❌ CROSSFADE FAILED!")
        print(f"   Error: {e}")
        print(f"   stdout: {e.stdout}")
        print(f"   stderr: {e.stderr}")
        print("=" * 60)
        raise


@app.route('/api/current-track', methods=['GET'])
def current_track():
    """
    Get current playback info for the remix page
    """
    return jsonify({
        'current_track': session.get('current_track_1', 'Track 1'),
        'next_track': session.get('current_track_2', 'Track 2'),
        'bpm': session.get('current_bpm', 128),
        'key': session.get('current_key', 'C major'),
        'left_track': session.get('current_track_1', 'Track 1'),
        'right_track': session.get('current_track_2', 'Track 2')
    })


@app.route('/api/play', methods=['POST'])
def play():
    """Resume playback"""
    # TODO: Implement actual audio playback control
    return jsonify({'success': True})


@app.route('/api/pause', methods=['POST'])
def pause():
    """Pause playback"""
    # TODO: Implement actual audio playback control
    return jsonify({'success': True})


@app.route('/api/start-playlist', methods=['POST'])
def start_playlist():
    """Start auto-mixing a playlist (cyan vinyl flow)"""
    data = request.json
    playlist_name = data.get('playlist')
    
    print(f"🎵 Starting auto-mix for playlist: {playlist_name}")
    
    # TODO: Load playlist, queue songs, start auto-mixing
    session['playlist_mode'] = True
    session['current_playlist'] = playlist_name
    
    return jsonify({
        'success': True,
        'playlist': playlist_name
    })

@app.route('/party-mix')
def party_mix():
    return render_template('party_mix.html')

if __name__ == '__main__':
    print("🎧 AI DJ Backend Starting...")
    print(f"📁 MP3 Directory: {MP3_DIR}")
    print(f"📁 Metadata Directory: {METADATA_DIR}")
    print(f"📁 Output Directory: {OUTPUT_DIR}")
    app.run(debug=True, port=5000)