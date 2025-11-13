"""
Advanced Features Demo - Sprechstimme v2.0

Demonstrates the professional-grade features added to Sprechstimme:
- Advanced waveforms (FM, PWM, supersaw, wavetable, etc.)
- Professional filters (ladder, state-variable, shelving, etc.)
- Effects (chorus, reverb, phaser, compressor, etc.)
- Modulation (LFO, envelopes, etc.)
- Audio utilities (mixing, panning, etc.)
- MIDI utilities
- Audio analysis
"""

import sprechstimme as sp
import numpy as np

# Sample rate
SR = 44100


def demo_advanced_waveforms():
    """Demonstrate advanced waveform generators."""
    print("=== Advanced Waveforms Demo ===\n")

    # FM Synthesis
    print("1. FM Synthesis (bell-like tone)")
    sp.new("fm_synth", sp.waves.fm)
    sp.play("fm_synth", "C4", duration=2.0)

    # Supersaw (thick, detuned sound)
    print("2. Supersaw (thick synth sound)")
    sp.new("supersaw", sp.waves.supersaw)
    sp.play("supersaw", "A3", duration=2.0)

    # PWM (pulse width modulation)
    print("3. PWM (evolving pulse wave)")
    sp.new("pwm", sp.waves.pwm)
    sp.play("pwm", "E3", duration=2.0)

    # Morphing wave
    print("4. Morphing Wave (smooth transition between waveforms)")
    sp.new("morph", sp.waves.morphing_wave)
    sp.play("morph", "G3", duration=2.0)


def demo_professional_filters():
    """Demonstrate professional-grade filters."""
    print("\n=== Professional Filters Demo ===\n")

    # Moog ladder filter
    print("1. Moog Ladder Filter (warm, resonant)")
    sp.new("moog", sp.waves.sawtooth, filters=[sp.filters.moog(cutoff=800, resonance=0.6)])
    sp.play("moog", "A2", duration=2.0)

    # State Variable Filter
    print("2. State Variable Filter")
    sp.new("svf", sp.waves.sawtooth, filters=[sp.filters.svf(cutoff=1000, resonance=0.7, mode='low')])
    sp.play("svf", "C3", duration=2.0)

    # Peaking EQ
    print("3. Peaking EQ (boosting specific frequency)")
    sp.new("peaked", sp.waves.sawtooth, filters=[sp.filters.peak_eq(freq=1500, q=2.0, gain_db=12)])
    sp.play("peaked", "E3", duration=2.0)


def demo_effects():
    """Demonstrate professional effects."""
    print("\n=== Professional Effects Demo ===\n")

    # Create a simple melody
    sp.new("lead", sp.waves.sawtooth)

    print("1. Chorus Effect")
    melody = sp.get("lead", [("C4", 0.5), ("E4", 0.5), ("G4", 0.5), ("C5", 0.5)])
    chorus_melody = sp.effects.chorus(melody, SR, rate=1.5, depth=0.003, mix=0.6)
    sp.playback.play_array(chorus_melody, SR)

    print("2. Reverb Effect")
    reverb_melody = sp.effects.reverb(melody, SR, room_size=0.7, wet=0.4)
    sp.playback.play_array(reverb_melody, SR)

    print("3. Phaser Effect")
    phaser_melody = sp.effects.phaser(melody, SR, rate=0.5, stages=6, mix=0.7)
    sp.playback.play_array(phaser_melody, SR)


def demo_modulation():
    """Demonstrate modulation capabilities."""
    print("\n=== Modulation Demo ===\n")

    # LFO modulation
    print("1. LFO Tremolo")
    sp.new("osc", sp.waves.sine)
    signal = sp.get("osc", "A3", duration=3.0)

    lfo = sp.modulation.LFO(rate=6.0, waveform='sine', depth=0.5)
    t = np.arange(len(signal)) / SR
    modulated = sp.modulation.tremolo(signal, t, rate=6.0, depth=0.6)
    sp.playback.play_array(modulated, SR)

    # Auto-pan
    print("2. Auto-Pan (stereo effect)")
    left, right = sp.modulation.auto_pan(signal, t, rate=0.5)
    stereo = sp.utils.interleave_stereo(left, right)
    sp.playback.play_array(stereo, SR)


def demo_wavetables():
    """Demonstrate wavetable synthesis."""
    print("\n=== Wavetable Synthesis Demo ===\n")

    # Create custom wavetable
    print("1. Custom Wavetable")
    table = sp.wavetables.generate_harmonic_table(harmonics=[1.0, 0.5, 0.3, 0.2, 0.1])
    wt = sp.wavetables.Wavetable(table, SR)

    t = np.linspace(0, 2.0, int(2.0 * SR))
    audio = wt.generate(t, freq=440, amp=0.5)
    sp.playback.play_array(audio, SR)

    # PPG-style wavetable
    print("2. PPG-style Wavetable")
    ppg_table = sp.wavetables.generate_ppg_table(wave_number=16)
    wt_ppg = sp.wavetables.Wavetable(ppg_table, SR)
    audio_ppg = wt_ppg.generate(t, freq=330, amp=0.5)
    sp.playback.play_array(audio_ppg, SR)


def demo_midi_utils():
    """Demonstrate MIDI utilities."""
    print("\n=== MIDI Utilities Demo ===\n")

    # Generate scale
    print("1. Generate C Major Scale")
    scale = sp.midi_utils.generate_scale("C4", "major", octaves=2)
    print(f"Scale notes (MIDI): {scale}")

    # Generate chord progression
    print("\n2. Generate Chord Progression")
    progression = sp.midi_utils.random_chord_progression("C4", length=4)
    print(f"Chord progression: {progression}")

    # Play the progression
    sp.new("chord_synth", sp.waves.triangle, poly=True)
    song = sp.Song(bpm=100)

    for i, chord in enumerate(progression):
        song.add("chord_synth", chord, beat=i*4, duration=3.5)

    audio = song.render()
    sp.playback.play_array(audio, SR)


def demo_audio_utilities():
    """Demonstrate audio utilities."""
    print("\n=== Audio Utilities Demo ===\n")

    # Create two signals
    sp.new("osc1", sp.waves.sine)
    sp.new("osc2", sp.waves.triangle)

    sig1 = sp.get("osc1", "C4", duration=2.0)
    sig2 = sp.get("osc2", "E4", duration=2.0)

    # Mix signals
    print("1. Mixing two signals")
    mixed = sp.utils.mix([sig1, sig2], levels=[0.5, 0.5])
    sp.playback.play_array(mixed, SR)

    # Pan signal
    print("2. Panning (stereo)")
    left, right = sp.utils.pan(mixed, pan_position=0.7)  # Pan to right
    stereo = sp.utils.interleave_stereo(left, right)
    sp.playback.play_array(stereo, SR)

    # Fade in/out
    print("3. Fade in and fade out")
    faded = sp.utils.fade_in(mixed, duration=0.5, sample_rate=SR)
    faded = sp.utils.fade_out(faded, duration=0.5, sample_rate=SR)
    sp.playback.play_array(faded, SR)

    # Normalize
    print("4. Normalization")
    normalized = sp.utils.normalize(mixed, target_level=0.9)
    sp.playback.play_array(normalized, SR)


def demo_analysis():
    """Demonstrate audio analysis tools."""
    print("\n=== Audio Analysis Demo ===\n")

    # Create a test signal
    sp.new("test", sp.waves.sine)
    signal = sp.get("test", 440, duration=1.0)  # A4 = 440 Hz

    # Pitch detection
    print("1. Pitch Detection")
    detected_pitch = sp.analysis.detect_pitch_autocorrelation(signal, SR)
    print(f"Detected pitch: {detected_pitch:.2f} Hz (expected: 440 Hz)")

    detected_pitch_fft = sp.analysis.detect_pitch_fft(signal, SR)
    print(f"Detected pitch (FFT): {detected_pitch_fft:.2f} Hz")

    # Spectral analysis
    print("\n2. Spectral Centroid")
    centroid = sp.analysis.spectral_centroid(signal, SR)
    print(f"Spectral centroid: {centroid:.2f} Hz")

    # RMS energy
    print("\n3. RMS Energy")
    rms = sp.analysis.rms_energy(signal)
    print(f"RMS value: {sp.utils.rms(signal):.4f}")


def demo_new_presets():
    """Demonstrate new presets."""
    print("\n=== New Presets Demo ===\n")

    presets_to_demo = [
        "fm_bell", "supersaw", "pwm_pad", "hard_sync_lead",
        "unison_saw", "moog_bass", "formant_vocal", "plucked_string"
    ]

    for preset in presets_to_demo:
        print(f"Playing preset: {preset}")
        sp.new("demo", preset=preset)
        sp.play("demo", "A3", duration=1.5)


def demo_complete_track():
    """Create a complete track using multiple features."""
    print("\n=== Complete Track Demo ===\n")

    song = sp.Song(bpm=120)

    # Create instruments
    sp.new("bass", preset="moog_bass")
    sp.new("lead", preset="supersaw")
    sp.new("pad", preset="pwm_pad")
    sp.new("perc", preset="pink_noise_hat")

    # Bassline
    bass_pattern = [
        ("C2", 1, 7), ("C2", 1, 5), ("D#2", 1, 6), ("C2", 1, 7),
        ("C2", 1, 7), ("C2", 1, 5), ("F2", 1, 6), ("G2", 1, 7),
    ]

    # Lead melody
    lead_pattern = [
        ("C4", 0.5, 6), ("E4", 0.5, 7), ("G4", 0.5, 8), ("C5", 0.5, 7),
        ("B4", 1, 6), ("G4", 1, 5),
    ]

    # Pad chords
    pad_chords = ["C4", "Am3", "F3", "G3"]

    # Arrange the song
    for bar in range(4):
        beat_offset = bar * 8

        # Add bass
        for i, note in enumerate(bass_pattern):
            song.add("bass", note[0], beat=beat_offset + i, duration=note[1], vol=note[2])

        # Add lead (every other bar)
        if bar % 2 == 1:
            for i, note in enumerate(lead_pattern):
                song.add("lead", note[0], beat=beat_offset + i, duration=note[1], vol=note[2])

        # Add pad
        chord_idx = bar % len(pad_chords)
        song.add("pad", pad_chords[chord_idx], beat=beat_offset, duration=7.5, vol=4)

        # Add percussion
        for i in range(8):
            if i % 2 == 1:
                song.add("perc", "C6", beat=beat_offset + i, duration=0.1, vol=5)

    # Render and apply effects
    print("Rendering track...")
    audio = song.render()

    print("Applying reverb...")
    audio = sp.effects.reverb(audio, SR, room_size=0.5, wet=0.25)

    print("Applying compression...")
    audio = sp.effects.compressor(audio, SR, threshold=-15, ratio=4.0)

    print("Normalizing...")
    audio = sp.utils.normalize(audio, target_level=0.95)

    print("Playing complete track...")
    sp.playback.play_array(audio, SR)

    # Export
    print("Exporting to track.wav...")
    sp.playback.export_wav(audio, "track.wav", SR)
    print("Track exported!")


if __name__ == "__main__":
    print("Sprechstimme v2.0 - Advanced Features Demo")
    print("=" * 50)

    # Uncomment the demos you want to run:

    # demo_advanced_waveforms()
    # demo_professional_filters()
    # demo_effects()
    # demo_modulation()
    # demo_wavetables()
    # demo_midi_utils()
    # demo_audio_utilities()
    # demo_analysis()
    # demo_new_presets()
    demo_complete_track()

    print("\n" + "=" * 50)
    print("Demo complete!")
