# barmap.py
# python -m calibrate.barmap data/mp3s/Dynamite.mp3

import json
import os
import numpy as np
from dj_helpers.beatgrid import get_beatgrid

def get_barmap(path, extend_to_end=True):
    """
    Returns a simple dict of bar start times and tempo for a given track.
    
    Args:
        path: Path to audio file
        extend_to_end: If True, extends bar grid to end of song even if beat detection stops early
    """
    grid = get_beatgrid(path)
    bars = grid.bars.copy()
    
    # Check if bars go all the way to the end
    if extend_to_end and len(bars) > 1:
        last_bar = bars[-1]
        duration = grid.duration
        
        # If there's a gap between last detected bar and end of song
        gap = duration - last_bar
        
        # Calculate average bar length from detected bars
        bar_intervals = np.diff(bars)
        avg_bar_length = np.median(bar_intervals)  # Use median to be robust
        
        # If gap is more than half a bar, extend the grid
        if gap > avg_bar_length * 0.5:
            print(f"⚠️  Beat detection stopped at {last_bar:.2f}s, but song ends at {duration:.2f}s")
            print(f"   Extending bar grid by {gap:.2f}s...")
            
            # Add bars until we reach the end
            next_bar = last_bar + avg_bar_length
            while next_bar < duration:
                bars = np.append(bars, next_bar)
                next_bar += avg_bar_length
            
            print(f"   Added {len(bars) - len(grid.bars)} bars to reach end of song")
    
    barmap = {
        "bpm": grid.bpm,
        "duration": grid.duration,
        "bars": np.round(bars, 3).tolist()
    }
    return barmap


def save_barmap(path, out_dir="data/metadata", extend_to_end=True):
    """
    Generates and saves a JSON barmap file for the given audio track.
    Example: data/metadata/Dynamite_barmap.json
    
    Args:
        path: Path to audio file
        out_dir: Output directory for barmap JSON
        extend_to_end: If True, extends bar grid to end of song
    """
    song_name = os.path.splitext(os.path.basename(path))[0]
    barmap = get_barmap(path, extend_to_end=extend_to_end)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{song_name}_barmap.json")

    with open(out_path, "w") as f:
        json.dump(barmap, f, indent=2)

    print(f"✅ Saved barmap for {song_name} → {out_path}")
    print(f"   BPM: {barmap['bpm']:.1f}, Bars: {len(barmap['bars'])}")
    print(f"   Duration: {barmap['duration']:.2f}s, Last bar: {barmap['bars'][-1]:.2f}s")
    print(f"   First few bars: {barmap['bars'][:5]}")
    return barmap


# -------------------------------------------------------------
# Run directly from terminal
# -------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m calibrate.barmap <audio_path>")
        sys.exit(1)

    path = sys.argv[1]
    save_barmap(path)
