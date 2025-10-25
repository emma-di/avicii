# dj_helpers/mc_voice.py
# WIP! Paused for now
import os, requests, sys

API_KEY = os.getenv("FISH_API_KEY")
URL = "https://api.fish.audio/v1/tts"

def generate_mc_audio(text, out_path="mc_output.mp3"):
    if not API_KEY:
        raise RuntimeError("FISH_API_KEY not set. In PowerShell:  $env:FISH_API_KEY='sk_...'")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "text": text,
        "format": "mp3",
        "reference_id": "4ad8a50b97e1420f980557e239da46e5",  # your chosen voice
        "model": "speech-1.5"
    }

    r = requests.post(URL, headers=headers, json=data, timeout=60)
    if not r.ok:
        # Print the server message; 402 will usually say “insufficient API credit”
        print(f"HTTP {r.status_code} {r.reason}\n{r.text}", file=sys.stderr)
        r.raise_for_status()

    with open(out_path, "wb") as f:
        f.write(r.content)
    return out_path

if __name__ == "__main__":
    print(generate_mc_audio("Ladies and gentlemen, make some noise!"))