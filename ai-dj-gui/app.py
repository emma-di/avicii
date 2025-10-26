"""
Flask Backend API for AI DJ
Handles song loading and crossfading
"""

from flask import Flask, render_template, request, jsonify, session, send_file
from werkzeug.utils import secure_filename
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
UPLOAD_FOLDER = MP3_DIR
ALLOWED_EXTENSIONS = {'mp3', 'wav'}

# Ensure directories exist
os.makedirs(MP3_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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


@app.route('/party-mix')
def party_mix():
    return render_template('party_mix.html')


@app.route('/remix')
def remix():
    """Live mixing interface"""
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


@app.route('/api/upload-and-mix', methods=['POST'])
def upload_and_mix():
    """
    Handle mixed uploads and library songs
    
    Can receive:
    - file1 + file2 (both uploaded)
    - file1 + file2_path (one uploaded, one from library)
    - file1_path + file2 (one from library, one uploaded)
    """
    
    track1_path = None
    track2_path = None
    track1_name = None
    track2_name = None
    
    try:
        # Process track 1
        if 'file1' in request.files and request.files['file1'].filename:
            # Track 1 is uploaded
            file1 = request.files['file1']
            
            if not allowed_file(file1.filename):
                return jsonify({'success': False, 'error': 'Invalid file type for track 1'}), 400
            
            # Clean filename - remove spaces and special chars, keep only alphanumeric, dash, underscore
            filename1 = secure_filename(file1.filename)
            # Further clean: remove spaces and special characters
            name_part, ext = os.path.splitext(filename1)
            clean_name = ''.join(c for c in name_part if c.isalnum() or c in '-_')
            clean_name = clean_name.replace(' ', '_')
            filename1 = f"{clean_name}{ext}"
            
            track1_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, filename1))
            file1.save(track1_path)
            track1_name = os.path.splitext(filename1)[0]
            
            print(f"✅ Uploaded track 1: {filename1}")
            print(f"   Path: {track1_path}")
            
            # Run calibration
            try:
                subprocess.run([
                    'python', '-m', 'calibrate.calibrate_simple',
                    track1_path,
                    '--no-stems'
                ], capture_output=True, text=True, check=True, timeout=60)
                print(f"✅ Calibrated track 1")
            except Exception as e:
                print(f"⚠️  Calibration warning for track 1: {e}")
        
        elif 'file1_path' in request.form:
            # Track 1 is from library
            track1_path = request.form['file1_path']
            track1_name = os.path.splitext(os.path.basename(track1_path))[0]
            print(f"✅ Using library track 1: {track1_name}")
        
        else:
            return jsonify({'success': False, 'error': 'No track 1 provided'}), 400
        
        # Process track 2
        if 'file2' in request.files and request.files['file2'].filename:
            # Track 2 is uploaded
            file2 = request.files['file2']
            
            if not allowed_file(file2.filename):
                return jsonify({'success': False, 'error': 'Invalid file type for track 2'}), 400
            
            # Clean filename - remove spaces and special chars, keep only alphanumeric, dash, underscore
            filename2 = secure_filename(file2.filename)
            # Further clean: remove spaces and special characters
            name_part, ext = os.path.splitext(filename2)
            clean_name = ''.join(c for c in name_part if c.isalnum() or c in '-_')
            clean_name = clean_name.replace(' ', '_')
            filename2 = f"{clean_name}{ext}"
            
            track2_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, filename2))
            file2.save(track2_path)
            track2_name = os.path.splitext(filename2)[0]
            
            print(f"✅ Uploaded track 2: {filename2}")
            print(f"   Path: {track2_path}")
            
            # Run calibration
            try:
                subprocess.run([
                    'python', '-m', 'calibrate.calibrate_simple',
                    track2_path,
                    '--no-stems'
                ], capture_output=True, text=True, check=True, timeout=60)
                print(f"✅ Calibrated track 2")
            except Exception as e:
                print(f"⚠️  Calibration warning for track 2: {e}")
        
        elif 'file2_path' in request.form:
            # Track 2 is from library
            track2_path = request.form['file2_path']
            track2_name = os.path.splitext(os.path.basename(track2_path))[0]
            print(f"✅ Using library track 2: {track2_name}")
        
        else:
            return jsonify({'success': False, 'error': 'No track 2 provided'}), 400
        
        # Store paths in session for crossfade
        session['track1_path'] = track1_path
        session['track2_path'] = track2_path
        session['current_track_1'] = track1_name
        session['current_track_2'] = track2_name
        
        # Set output path
        output_name = f"{track1_name}_{track2_name}_mix.wav"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        session['current_mix_path'] = output_path
        
        print(f"✅ Ready to mix: {track1_name} + {track2_name}")
        
        return jsonify({
            'success': True,
            'track1_name': track1_name,
            'track2_name': track2_name,
            'message': 'Tracks ready for mixing'
        })
        
    except Exception as e:
        print(f"❌ Error in upload-and-mix: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/start-crossfade', methods=['POST'])
def start_crossfade():
    """Handle crossfade request from the UI"""
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
    
    try:
        output_name = f"{track1_info.get('title', 'track1')}_{track2_info.get('title', 'track2')}_mix.wav"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        session['current_mix_path'] = output_path
        
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
    """Process the mix (called from loading screen)"""
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
    """Run the crossfade_stems.py script"""
    
    # Try multiple possible locations for crossfade script
    possible_paths = [
        '../dj_helpers/crossfade_stems.py',
        'dj_helpers/crossfade_stems.py',
        '../crossfade_stems.py',
        'crossfade_stems.py'
    ]
    
    crossfade_script = None
    for path in possible_paths:
        if os.path.exists(path):
            crossfade_script = path
            print(f"✅ Found crossfade script at: {path}")
            break
    
    if not crossfade_script:
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
            timeout=300
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


@app.route('/api/get-current-mix')
def get_current_mix():
    """Serve the current mix audio file"""
    mix_path = session.get('current_mix_path')
    
    if not mix_path or not os.path.exists(mix_path):
        print(f"❌ Mix file not found: {mix_path}")
        return "Mix file not found", 404
    
    print(f"🎵 Serving mix: {mix_path}")
    
    return send_file(
        mix_path,
        mimetype='audio/wav',
        as_attachment=False,
        download_name='current_mix.wav'
    )


@app.route('/api/current-track', methods=['GET'])
def current_track():
    """Get current playback info for the remix page"""
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
    return jsonify({'success': True})


@app.route('/api/pause', methods=['POST'])
def pause():
    """Pause playback"""
    return jsonify({'success': True})


@app.route('/api/start-playlist', methods=['POST'])
def start_playlist():
    """Start auto-mixing a playlist (cyan vinyl flow)"""
    data = request.json
    playlist_name = data.get('playlist')
    
    print(f"🎵 Starting auto-mix for playlist: {playlist_name}")
    
    session['playlist_mode'] = True
    session['current_playlist'] = playlist_name
    
    return jsonify({
        'success': True,
        'playlist': playlist_name
    })


if __name__ == '__main__':
    print("🎧 AI DJ Backend Starting...")
    print(f"📁 MP3 Directory: {MP3_DIR}")
    print(f"📁 Metadata Directory: {METADATA_DIR}")
    print(f"📁 Output Directory: {OUTPUT_DIR}")
    app.run(debug=True, port=5000)