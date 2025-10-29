CalHacks 2025  
By Emma, Sidh, Aatreyo, and Nicole  

# AUDIO PROCESSING  
To set up (for audio processing):  
        python -m venv venv39  
        To run venv: venv39\Scripts\activate  
        pip install -r requirements.txt  

After uploading desired mp3 to data/mp3s:  
        python -m calibrate.calibrate_simple  
        -> run this in the background since it will take a few mins  

RESULTS:  
    jsons stored in data//metadata  
    audio files stored in data//htdemucs  

# DJ HELPERS  

Barmap: creates JSON with beginning times of each bar  
    Uses beatmap  
    
Crossfade

# To run the demo: 

run in terminal:

    cd ai-dj-gui

    # (Optional) Install Flask if you don't have it already
    pip install flask
    
    python app.py

demo video also accessible here: https://youtu.be/up-Bnev-TF4
