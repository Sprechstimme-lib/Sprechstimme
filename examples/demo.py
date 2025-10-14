import sprechstimme as sp
from functools import partial

# ============================================================================
# INSTRUMENT CREATION
# ============================================================================

print("\n[1/5] Creating custom instruments...")

# 1. LEAD SYNTH - Bright, cutting lead with envelope
sp.new("lead")
sp.create("lead",
    wavetype=sp.waves.sawtooth,
    filters=[partial(sp.filters.low_pass, cutoff=2000, order=4)],
    envelope={
        "attack": 0.01,
        "decay": 0.1,
        "sustain": 0.6,
        "release": 0.2
    }
)

# 2. BASS - Deep, warm bass
sp.new("bass")
sp.create("bass",
    wavetype=sp.waves.square,
    filters=[partial(sp.filters.low_pass, cutoff=400, order=4)],
    envelope={
        "attack": 0.005,
        "decay": 0.1,
        "sustain": 0.9,
        "release": 0.15
    }
)

# 3. PAD - Soft atmospheric pad
sp.new("pad")
sp.create("pad",
    wavetype=sp.waves.sine,
    filters=[partial(sp.filters.low_pass, cutoff=1200, order=4)],
    envelope={
        "attack": 0.5,
        "decay": 0.3,
        "sustain": 0.7,
        "release": 1.0
    }
)

# 4. PLUCK - Short, plucky sound for arpeggios
sp.new("pluck")
sp.create("pluck",
    wavetype=sp.waves.triangle,
    filters=[partial(sp.filters.high_pass, cutoff=200, order=4)],
    envelope={
        "attack": 0.001,
        "decay": 0.15,
        "sustain": 0.1,
        "release": 0.1
    }
)

# 5. KICK - Percussive kick drum
sp.new("kick")
sp.create("kick",
    wavetype=sp.waves.sine,
    filters=[partial(sp.filters.low_pass, cutoff=150, order=4)],
    envelope={
        "attack": 0.001,
        "decay": 0.1,
        "sustain": 0.0,
        "release": 0.05
    }
)

print("   ✓ Lead synth (sawtooth + lowpass)")
print("   ✓ Bass (square + lowpass)")
print("   ✓ Pad (sine + lowpass)")
print("   ✓ Pluck (triangle + highpass)")
print("   ✓ Kick drum (sine + envelope)")

# ============================================================================
# COMPOSITION
# ============================================================================

print("\n[2/5] Composing the track...")

track = sp.Track(bpm=128)

# Key: A minor
# Chord progression: Am - F - C - G

# === INTRO - PAD CHORDS (8 beats) ===
track.add("pad", [57, 60, 64], duration=4)  # Am
track.add("pad", [53, 57, 60], duration=4)  # F

# === SECTION 1 - Add bass line (16 beats) ===
# Bass notes
track.add("bass", 45, duration=4)  # A
track.add("bass", 41, duration=4)  # F
track.add("bass", 48, duration=4)  # C
track.add("bass", 43, duration=4)  # G

# Pad chords continue
track.add("pad", [57, 60, 64], duration=4)  # Am
track.add("pad", [53, 57, 60], duration=4)  # F
track.add("pad", [48, 52, 55], duration=4)  # C
track.add("pad", [43, 47, 50], duration=4)  # G

# === SECTION 2 - Add melody and arpeggios (16 beats) ===
# Bass continues
track.add("bass", 45, duration=4)  # A
track.add("bass", 41, duration=4)  # F
track.add("bass", 48, duration=4)  # C
track.add("bass", 43, duration=4)  # G

# Lead melody
melody = [
    (69, 1), (72, 1), (76, 1), (72, 1),  # A C E C
    (74, 1), (72, 1), (69, 1), (65, 1),  # D C A F
    (67, 1), (69, 1), (72, 1), (69, 1),  # G A C A
    (71, 2), (69, 2)                      # B A
]
track.add("lead", melody, duration=16)

# Pluck arpeggios (running 16th notes feel)
arp1 = [(57, 0.25), (60, 0.25), (64, 0.25), (67, 0.25)] * 4  # Am arp
arp2 = [(53, 0.25), (57, 0.25), (60, 0.25), (65, 0.25)] * 4  # F arp
arp3 = [(48, 0.25), (52, 0.25), (55, 0.25), (60, 0.25)] * 4  # C arp
arp4 = [(43, 0.25), (47, 0.25), (50, 0.25), (55, 0.25)] * 4  # G arp

track.add("pluck", arp1, duration=4)
track.add("pluck", arp2, duration=4)
track.add("pluck", arp3, duration=4)
track.add("pluck", arp4, duration=4)

# === SECTION 3 - BUILD UP with drums (16 beats) ===
# Kick drum pattern (on beats 1 and 3)
kick_pattern = [
    (36, 1), (0, 1), (36, 1), (0, 1)  # Using 0 as rest
] * 4
# Note: We'll use low C (36) for kick
track.add("kick", [(36, 0.5), (36, 0.5)] * 8, duration=8)  # Steady kick
track.add("kick", [(36, 0.25)] * 32, duration=8)  # Build up faster

# Bass builds
track.add("bass", [(45, 0.5), (45, 0.5)] * 8, duration=8)
track.add("bass", [(45, 0.25)] * 16, duration=4)
track.add("bass", [(41, 0.25)] * 16, duration=4)

# Lead climax
climax_melody = [
    (81, 1), (84, 1), (81, 0.5), (79, 0.5), (77, 1),
    (76, 2), (74, 1), (72, 1),
    (76, 2), (77, 2),
    (72, 4)
]
track.add("lead", climax_melody, duration=16)

# === OUTRO - Final chord (8 beats) ===
track.add("pad", [57, 60, 64, 69], duration=8)  # Am add9
track.add("bass", 33, duration=8)  # Low A

print("   ✓ Intro (pad chords)")
print("   ✓ Section 1 (+ bass line)")
print("   ✓ Section 2 (+ melody & arpeggios)")
print("   ✓ Section 3 (+ drums & build-up)")
print("   ✓ Outro (final chord)")

# ============================================================================
# PLAYBACK
# ============================================================================

print("\n[3/5] Playing the composition...")
print("   (Duration: ~45 seconds)")
track.play()

# ============================================================================
# EXPORT AUDIO
# ============================================================================

print("\n[4/5] Exporting audio...")
output_file = "epic_composition.wav"
track.export(output_file)
print(f"   ✓ Saved to: {output_file}")

# ============================================================================
# VISUALIZE
# ============================================================================

print("\n[5/5] Generating musical notation...")
try:
    # Export as PDF
    track.export_score("epic_composition_score.pdf", title="Epic Sprechstimme Composition")
    print("   ✓ Score saved as: epic_composition_score.pdf")

    # Export as MusicXML
    track.export_score("epic_composition_score.musicxml", title="Epic Sprechstimme Composition")
    print("   ✓ Score saved as: epic_composition_score.musicxml")

    # Optionally display (comment out if you don't want it to open)
    # track.show_staff(title="Epic Sprechstimme Composition")

except ImportError:
    print("   ⚠ music21 not installed - skipping notation export")
    print("   Install with: pip install music21")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("DEMO COMPLETE!")
print("=" * 60)
print("\nFiles created:")
print(f"  • {output_file}")
print(f"  • epic_composition_score.pdf")
print(f"  • epic_composition_score.musicxml")
print("\nFeatures demonstrated:")
print("  • 5 custom instruments with unique timbres")
print("  • ADSR envelopes for dynamic shaping")
print("  • Filter chains (lowpass, highpass)")
print("  • Complex chord progressions")
print("  • Melodic sequences")
print("  • Arpeggiated patterns")
print("  • Rhythmic variation")
print("  • Multi-track arrangement")
print("  • Professional notation export")
print("\n" + "=" * 60)
