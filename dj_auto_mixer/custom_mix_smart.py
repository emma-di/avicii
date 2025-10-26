#!/usr/bin/env python3
"""
PARTY DJ Mix Creator 🎉
- Beat drops and bass boosts
- Beat-synced transitions
- Creative mixing techniques
- OBVIOUS DJ manipulation

Just run: python custom_mix.py
"""

from smart_dj_integration import create_smart_dj_set

# =============================================================================
# YOUR SONGS
# =============================================================================

songs = [
    "DJGotUsFallinInLove",
    "Dynamite",
    "HotelRoomService",
    "Summer"
]

print("🎉 PARTY DJ MIX CREATOR")
print("=" * 80)
print(f"Tracks: {len(songs)}")
for i, song in enumerate(songs, 1):
    print(f"  {i}. {song}")
print("=" * 80)
print("\n🔥 PARTY FEATURES:")
print("  → Beat drops (remove bass, then DROP it back!)")
print("  → Bass boosts (+6dB thump)")
print("  → Stutters & scratches")
print("  → Beat-synced transitions")
print("  → EQ swaps (bass swap technique)")
print("  → Channel muting effects")
print("=" * 80)

# =============================================================================
# CREATE PARTY MIX
# =============================================================================

result = create_smart_dj_set(
    track_names=songs,
    output_dir="party_mix",
    
    # VIBE-BASED LENGTHS
    exit_preference='vibe_based',  # Dynamic! (30s-150s based on energy)
    
    # PARTY EFFECTS - MAXIMUM
    add_effects=True,              # Turn on ALL effects
    effect_intensity=1.0,          # 0.5=subtle, 1.0=party, 1.5=EXTREME
    
    # AUTO-ORDERING
    auto_order=True,
    energy_curve='wave'
)

# =============================================================================
# ALTERNATIVE CONFIGURATIONS
# =============================================================================

# EXTREME EFFECTS MODE
# ====================
# result = create_smart_dj_set(
#     track_names=songs,
#     output_dir="extreme_mix",
#     exit_preference='vibe_based',
#     add_effects=True,
#     effect_intensity=1.5  # MAXIMUM effects!
# )

# FORCE ALL SONGS SHORT
# ======================
# result = create_smart_dj_set(
#     track_names=songs,
#     output_dir="quick_mix",
#     exit_preference='early',  # Force 30-60s all tracks
#     add_effects=True
# )

# CLEAN MIX (No Effects)
# ======================
# result = create_smart_dj_set(
#     track_names=songs,
#     output_dir="clean_mix",
#     exit_preference='vibe_based',
#     add_effects=False  # No effects
# )

# =============================================================================
# RESULTS
# =============================================================================

print("\n" + "=" * 80)
print("✨ COMPLETE!")
print("=" * 80)

if result['success']:
    print(f"\n✅ Your mix is ready!")
    print(f"\n📁 File: {result['output_file']}")
    print(f"⏱️  Duration: {result['total_duration']/60:.1f} minutes")
    
    print(f"\n🎵 Track breakdown:")
    for i, track_info in enumerate(result['tracks'], 1):
        print(f"   {i}. {track_info['track']:<25} → {track_info['exit_time']:.1f}s")
        print(f"      {track_info['exit_reason']}")
    
    print(f"\n🎧 Play: {result['output_file']}")
    print(f"\n🎊 Features:")
    print(f"   ✓ Variable song lengths (vibe-based)")
    print(f"   ✓ 3-6 STRONG effects per track")
    print(f"   ✓ Fades out at end")
    print(f"\n🚀 Ready to rock Cal Hacks!")
else:
    print(f"\n❌ Error: {result.get('error', 'Unknown error')}")

print("=" * 80)