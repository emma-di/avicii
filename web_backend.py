from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import json
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='frontend')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'data/mp3s'
METADATA_FOLDER = 'data/metadata'
ALLOWED_EXTENSIONS = {'mp3', 'wav'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(METADATA_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('frontend', path)

@app.route('/api/upload', methods=['POST'])
def upload_song():
    """Upload and analyze a song"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        song_name = os.path.splitext(filename)[0]
        metadata_path = os.path.join(METADATA_FOLDER, f"{song_name}.json")
        
        # Check if metadata already exists
        if os.path.exists(metadata_path):
            print(f"✅ Metadata already exists for {song_name}, skipping analysis")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return jsonify({
                'success': True,
                'song_name': song_name,
                'filename': filename,
                'metadata': metadata,
                'cached': True
            })
        
        # Analyze the song (import from calibrate package)
        try:
            from calibrate.analyze_audio import analyze_song
        except ImportError:
            # Try without package prefix
            import sys
            sys.path.insert(0, os.path.dirname(__file__))
            from analyze_audio import analyze_song
        
        print(f"🎵 Analyzing {filename}...")
        metadata = analyze_song(filepath)
        
        if metadata is None:
            return jsonify({'error': 'Analysis failed'}), 500
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Saved metadata to {metadata_path}")
        
        return jsonify({
            'success': True,
            'song_name': song_name,
            'filename': filename,
            'metadata': metadata,
            'cached': False
        })
    
    except Exception as e:
        print(f"❌ Error uploading song: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/library', methods=['GET'])
def get_library():
    """Get all analyzed songs"""
    try:
        songs = []
        if os.path.exists(METADATA_FOLDER):
            for filename in os.listdir(METADATA_FOLDER):
                if filename.endswith('.json'):
                    song_name = filename[:-5]
                    metadata_path = os.path.join(METADATA_FOLDER, filename)
                    
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    songs.append({
                        'name': song_name,
                        'bpm': metadata.get('bpm', 120),
                        'key': metadata.get('key', 'C major'),
                        'duration': metadata.get('duration', 0),
                        'sections': len(metadata.get('sections', []))
                    })
        
        return jsonify({'songs': songs})
    
    except Exception as e:
        print(f"Error getting library: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/metadata/<song_name>', methods=['GET'])
def get_metadata(song_name):
    """Get metadata for a specific song"""
    try:
        metadata_path = os.path.join(METADATA_FOLDER, f"{song_name}.json")
        
        if not os.path.exists(metadata_path):
            return jsonify({'error': 'Song not found'}), 404
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return jsonify(metadata)
    
    except Exception as e:
        print(f"Error getting metadata: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/audio/<filename>', methods=['GET'])
def serve_audio(filename):
    """Serve audio files"""
    try:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        return send_file(filepath)
    except Exception as e:
        print(f"Error serving audio: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/next-track', methods=['POST'])
def get_next_track():
    """Get the next compatible track - placeholder for now"""
    try:
        data = request.get_json()
        current_song = data.get('current_song')
        
        # Get all songs
        songs = []
        if os.path.exists(METADATA_FOLDER):
            for filename in os.listdir(METADATA_FOLDER):
                if filename.endswith('.json'):
                    song_name = filename[:-5]
                    if song_name != current_song:
                        with open(os.path.join(METADATA_FOLDER, filename), 'r') as f:
                            metadata = json.load(f)
                        songs.append({
                            'name': song_name,
                            'metadata': metadata,
                            'score': 75  # Placeholder score
                        })
        
        if songs:
            # Return first available song for now
            return jsonify(songs[0])
        else:
            return jsonify({'error': 'No other songs available'}), 404
    
    except Exception as e:
        print(f"Error finding next track: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🎵 Starting AI DJ Backend...")
    print("📁 Upload folder:", UPLOAD_FOLDER)
    print("📊 Metadata folder:", METADATA_FOLDER)
    print("🌐 Frontend folder:", 'frontend')
    print("\n✨ Navigate to http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
