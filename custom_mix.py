#!/usr/bin/env python3
"""
Complete DJ Set Creator
Mixes all your songs in sequence!
"""

from create_dj_mix import batch_create_mixes

# Your songs in the order you want them mixed
songs = [
    "DJGotUsFallinInLove",
    "Dynamite",
    "HotelRoomService",
    "Summer"
]

print("🎵 Creating Complete DJ Set!")
print("=" * 60)
print(f"Songs in order:")
for i, song in enumerate(songs, 1):
    print(f"  {i}. {song}")
print("=" * 60)
print()

# Create all the mixes!
results = batch_create_mixes(
    track_names=songs,
    output_dir="my_dj_set",        # All mixes saved here
    use_htdemucs=False,            # Use original mp3s
    htdemucs_stem='all',           # If you change to htdemucs=True
    exclude_stems=None             # Set to ['vocals'] for instrumental
)

print("\n" + "=" * 60)
print("🎉 DJ SET COMPLETE!")
print("=" * 60)

# Show summary
successful = sum(1 for r in results if r is not None)
print(f"\n✅ Successfully created {successful}/{len(results)} mixes")

if successful > 0:
    print(f"\n📁 All mixes saved to: my_dj_set/")
    print("\nYour DJ set includes:")
    print("  1️⃣  mix_01_DJGotUsFallinInLove_to_Dynamite.wav")
    print("  2️⃣  mix_02_Dynamite_to_HotelRoomService.wav")
    print("  3️⃣  mix_03_HotelRoomService_to_Summer.wav")
    
    print("\n🎧 Play them in order for a complete DJ set experience!")
    
    # Show some stats if available
    if results[0]:
        print("\n📊 Quick Stats:")
        for i, result in enumerate(results, 1):
            if result:
                print(f"\n  Mix {i}: {result['track1']['name']} → {result['track2']['name']}")
                print(f"    Compatibility: {result['compatibility_score']:.1f}/10")
                print(f"    Duration: {result['duration']:.1f}s")

print("\n🚀 Ready to rock Cal Hacks! 🎊")