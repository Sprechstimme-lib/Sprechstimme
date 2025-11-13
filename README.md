# Sprechstimme

A professional-grade modular Python synthesizer and sequencer for creating music programmatically.

[![PyPI version](https://badge.fury.io/py/sprechstimme.svg)](https://badge.fury.io/py/sprechstimme)
[![Python](https://img.shields.io/pypi/pyversions/sprechstimme.svg)](https://pypi.org/project/sprechstimme/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

Sprechstimme is a professional-grade, feature-rich Python library for audio synthesis and music composition. Version 2.0 brings extensive professional features rivaling commercial synthesizers, including advanced synthesis methods, professional effects, comprehensive modulation, and powerful audio analysis tools—all with an intuitive, Pythonic API.

## Features

### Core Synthesis
- **17+ Waveform Types**: Sine, square, sawtooth, triangle, pulse, noise (white/pink/brown)
- **Advanced Synthesis**: FM synthesis, PWM, supersaw, hard sync, ring modulation, additive synthesis, formant synthesis, Karplus-Strong, wavetable synthesis
- **Morphing Waveforms**: Smooth interpolation between different wave shapes
- **Unison/Detune**: Multi-voice detuned oscillators for thick, wide sounds

### Professional Filters
- **Butterworth Filters**: Low-pass, high-pass, band-pass, band-stop
- **Analog-Style Filters**: Moog ladder filter, state-variable filter (SVF)
- **EQ Filters**: Peaking, low shelf, high shelf, notch filters
- **Special Filters**: Allpass, comb filter
- **Resonance Control**: Full resonance control on analog-style filters

### Professional Effects
- **Modulation Effects**: Chorus, flanger, phaser, tremolo, vibrato
- **Time-Based Effects**: Reverb (Schroeder), delay, ping-pong delay, echo
- **Dynamics**: Compressor, limiter
- **Distortion**: Overdrive, hard distortion, bitcrusher
- **All effects** with wet/dry mix and comprehensive parameter control

### Modulation & Envelopes
- **LFO**: Multiple waveforms (sine, triangle, square, saw, random), rate and depth control
- **Advanced Envelopes**: ADSR with multiple curve types (linear, exponential, logarithmic), AR envelopes
- **Envelope Follower**: Extract amplitude envelope from any signal
- **Step Sequencer**: Rhythmic modulation patterns
- **Auto-Pan**: Automatic stereo panning

### Audio Utilities
- **Mixing**: Multi-signal mixing with level control
- **Panning**: Stereo panning with equal-power law
- **Stereo Width**: Control stereo field width
- **Fades**: Fade in/out with multiple curve types, crossfading
- **Normalization**: Peak normalization, RMS measurement
- **Time Manipulation**: Reverse, time stretch, concatenate, repeat
- **Clipping**: Hard and soft clipping
- **Conversions**: dB/linear, stereo/mono conversion

### Wavetable Synthesis
- **Preset Wavetables**: Sine, saw, square, triangle, pulse, harmonic series
- **Advanced Wavetables**: PPG-style, vowel formants, Serum-style formula-based
- **Wavetable Banks**: Smooth morphing between multiple wavetables
- **Custom Wavetables**: Create from arrays or mathematical functions

### MIDI Utilities
- **Conversions**: MIDI ↔ frequency, MIDI ↔ note names
- **Scales**: Generate scales (major, minor, modes, pentatonic, blues, etc.)
- **Chords**: Generate chords (major, minor, 7ths, 9ths, sus, dim, aug, etc.)
- **Progressions**: Random chord progression generator
- **Melody**: Random melody generation from scales
- **Quantization**: Grid quantization, swing timing
- **Velocity Curves**: Linear, exponential, logarithmic

### Audio Analysis
- **Spectrum Analysis**: FFT, power spectrum, spectrogram
- **Pitch Detection**: Autocorrelation, FFT, YIN algorithm, harmonic product spectrum
- **Spectral Features**: Centroid, rolloff, flatness
- **Onset Detection**: Spectral flux-based onset detection
- **Tempo Detection**: BPM estimation
- **Energy Analysis**: RMS, zero-crossing rate
- **Peak Finding**: Configurable peak detection

### Sequencing & Composition
- **Track Sequencing**: BPM-based timing, per-note dynamics
- **Multi-Track Songs**: Unlimited tracks, universal beat counter
- **Precise Timing**: Fractional beat positioning (e.g., beat 200.5)
- **Dynamics Control**: Per-note volume control (1-10 scale)
- **Flexible Note Input**: MIDI numbers, frequencies, note names, chord notation
- **Real-time Playback**: Instant audio playback
- **WAV Export**: High-quality audio export

### Presets
- **33+ Instrument Presets**: Acoustic, electric, synthesizers, drums, and more
- **Classic Synths**: Piano, organ, bass, leads, pads, strings, brass, bells
- **Advanced Synths**: FM bells, supersaw, PWM pads, hard sync leads, unison saws
- **Specialized**: Formant vocals, plucked strings, Moog bass, resonant filters
- **Effects Presets**: Pre-configured effect chains

## Installation

```bash
pip install sprechstimme
```

### Requirements

- Python >= 3.8
- numpy
- sounddevice
- scipy

## Quick Start

### Basic Synth Creation and Playback

```python
import sprechstimme as sp

# Create and configure a synthesizer
sp.new("lead")
sp.create("lead", wavetype=sp.waves.sawtooth)

# Play a single note (MIDI number 60 = middle C)
sp.play("lead", 60, duration=0.5)

# Play using note names
sp.play("lead", "C4", duration=0.5)

# Play a chord
sp.play("lead", [60, 64, 67], duration=1.0)  # C major chord
```

### Using Filters and Envelopes

```python
import sprechstimme as sp
from functools import partial

# Create a synth with filters
sp.new("bass")
sp.create(
    "bass",
    wavetype=sp.waves.sawtooth,
    filters=[
        partial(sp.filters.low_pass, cutoff=800),
        partial(sp.filters.echo, delay=0.3, decay=0.5)
    ],
    envelope={
        "attack": 0.01,
        "decay": 0.1,
        "sustain": 0.7,
        "release": 0.2
    }
)

sp.play("bass", "A2", duration=1.0)
```

### Sequencing with Track

```python
import sprechstimme as sp

# Create synthesizers
sp.new("lead")
sp.create("lead", wavetype=sp.waves.square)

sp.new("bass")
sp.create("bass", wavetype=sp.waves.sawtooth)

# Create a track at 120 BPM
track = sp.Track(bpm=120)

# Add notes (duration in beats)
track.add("bass", notes="C2", duration=4)
track.add("lead", notes="C4", duration=1)
track.add("lead", notes="E4", duration=1)
track.add("lead", notes="G4", duration=1)
track.add("lead", notes="C5", duration=1)

# Play the sequence
track.play()

# Export to WAV file
track.export("my_composition.wav")
```

### Multi-Track Composition with Song

The `Song` class enables multiple tracks to play simultaneously with precise beat positioning:

```python
import sprechstimme as sp
from functools import partial

# Create synthesizers
sp.new("bass")
sp.create("bass", wavetype=sp.waves.sawtooth,
    filters=[partial(sp.filters.low_pass, cutoff=200)])

sp.new("lead")
sp.create("lead", wavetype=sp.waves.square)

# Create a song at 120 BPM
song = sp.Song(bpm=120)

# Add bass notes at specific beat positions
song.add("bass_track", "bass", "C2", beat_position=0, duration=2)
song.add("bass_track", "bass", "G1", beat_position=2, duration=2)

# Add melody starting at beat 1, with half-beat precision
song.add("melody", "lead", "C4", beat_position=1, duration=0.5)
song.add("melody", "lead", "E4", beat_position=1.5, duration=0.5)
song.add("melody", "lead", "G4", beat_position=2, duration=1)

# Add chords at specific positions
song.add_chord("chords", "lead", "C4", beat_position=0, duration=4)

# Play or export
song.play()
song.export("my_song.wav")

# Get song information
print(song.get_duration())  # {'beats': 4, 'seconds': 2.0, 'formatted': '0:02.00'}
print(song.list_tracks())   # Track info with event counts
```

### Advanced: Chord Notation

```python
import sprechstimme as sp

sp.new("pad")
sp.create("pad", wavetype=sp.waves.sine)

track = sp.Track(bpm=90)

# Add chords using chord notation
track.addChord("pad", "C4", duration=2)   # C major, octave 4
track.addChord("pad", "Am3", duration=2)  # A minor, octave 3
track.addChord("pad", "F4", duration=2)   # F major, octave 4
track.addChord("pad", "G4", duration=2)   # G major, octave 4

track.play()
```

## Version 2.0 - Professional Features

Sprechstimme 2.0 introduces extensive professional-grade features for advanced synthesis and production.

### Advanced Synthesis

```python
import sprechstimme as sp

# FM Synthesis
sp.new("fm_bell", sp.waves.fm)
sp.play("fm_bell", "C4", duration=2.0)

# Supersaw (thick, detuned sound)
sp.new("supersaw", sp.waves.supersaw)
sp.play("supersaw", "A3", duration=2.0)

# PWM (Pulse Width Modulation)
sp.new("pwm", sp.waves.pwm)
sp.play("pwm", "E3", duration=2.0)

# Wavetable Synthesis
from sprechstimme import wavetables

table = wavetables.generate_harmonic_table(harmonics=[1.0, 0.5, 0.3, 0.2])
wt = wavetables.Wavetable(table)
# Use in synthesis...
```

### Professional Effects

```python
import sprechstimme as sp
from sprechstimme import effects

# Create a melody
sp.new("lead", sp.waves.sawtooth)
melody = sp.get("lead", [("C4", 0.5), ("E4", 0.5), ("G4", 0.5)])

# Apply chorus
chorus_melody = effects.chorus(melody, 44100, rate=1.5, mix=0.6)

# Apply reverb
reverb_melody = effects.reverb(melody, 44100, room_size=0.7, wet=0.4)

# Apply phaser
phaser_melody = effects.phaser(melody, 44100, rate=0.5, stages=6)

# Chain effects
processed = effects.chorus(melody, 44100)
processed = effects.reverb(processed, 44100)
processed = effects.compressor(processed, 44100, threshold=-15, ratio=4.0)

sp.playback.play_array(processed, 44100)
```

### Professional Filters

```python
import sprechstimme as sp
from functools import partial

# Moog ladder filter (warm analog sound)
sp.new("moog_bass", sp.waves.sawtooth,
    filters=[partial(sp.filters.ladder_filter, cutoff=600, resonance=0.7)])
sp.play("moog_bass", "A1", duration=2.0)

# State variable filter with resonance
sp.new("resonant", sp.waves.sawtooth,
    filters=[partial(sp.filters.state_variable_filter, cutoff=1000, resonance=0.8, mode='low')])

# Peaking EQ
sp.new("boosted", sp.waves.sawtooth,
    filters=[partial(sp.filters.peaking, freq=1500, q=2.0, gain_db=12)])
```

### Modulation & LFO

```python
import sprechstimme as sp
from sprechstimme import modulation
import numpy as np

# Create LFO
lfo = modulation.LFO(rate=5.0, waveform='sine', depth=0.8)

# Generate signal
sp.new("osc", sp.waves.sine)
signal = sp.get("osc", "A3", duration=3.0)

# Apply tremolo
t = np.arange(len(signal)) / 44100
modulated = modulation.tremolo(signal, t, rate=6.0, depth=0.6)

# Auto-pan for stereo
left, right = modulation.auto_pan(signal, t, rate=0.5)

# Custom ADSR envelope
env = modulation.ADSR(attack=0.1, decay=0.2, sustain=0.7, release=0.5, curve='exponential')
envelope = env.generate(duration=2.0, sample_rate=44100)
```

### Audio Utilities

```python
import sprechstimme as sp
from sprechstimme import utils

# Mix multiple signals
sp.new("osc1", sp.waves.sine)
sp.new("osc2", sp.waves.triangle)
sig1 = sp.get("osc1", "C4", duration=2.0)
sig2 = sp.get("osc2", "E4", duration=2.0)

mixed = utils.mix([sig1, sig2], levels=[0.6, 0.4])

# Pan to stereo
left, right = utils.pan(mixed, pan_position=0.7)  # Pan right

# Normalize
normalized = utils.normalize(mixed, target_level=0.9)

# Fade in/out
faded = utils.fade_in(mixed, duration=0.5, sample_rate=44100)
faded = utils.fade_out(faded, duration=0.5, sample_rate=44100)

# Apply gain in dB
gained = utils.apply_gain(mixed, gain_db=6.0)
```

### MIDI Utilities

```python
import sprechstimme as sp
from sprechstimme import midi_utils

# Generate scales
c_major = midi_utils.generate_scale("C4", "major", octaves=2)
a_minor = midi_utils.generate_scale("A3", "minor", octaves=1)
blues_scale = midi_utils.generate_scale("E3", "blues")

# Generate chords
c_major_chord = midi_utils.generate_chord("C4", "major")
am7_chord = midi_utils.generate_chord("A3", "minor7")
dom9_chord = midi_utils.generate_chord("G3", "dom9")

# Parse chord notation
chord = midi_utils.parse_chord_notation("Cmaj7")

# Generate random progression
progression = midi_utils.random_chord_progression("C4", length=4)

# Convert between formats
freq = midi_utils.midi_to_freq(60)  # 261.63 Hz (middle C)
midi_num = midi_utils.note_to_midi("A4")  # 69
note_name = midi_utils.midi_to_note(69)  # "A4"
```

### Audio Analysis

```python
import sprechstimme as sp
from sprechstimme import analysis

# Create test signal
sp.new("test", sp.waves.sine)
signal = sp.get("test", 440, duration=1.0)  # A4

# Pitch detection
pitch = analysis.detect_pitch_autocorrelation(signal, 44100)
print(f"Detected pitch: {pitch:.2f} Hz")

# Spectrum analysis
freqs, mags = analysis.fft_spectrum(signal, 44100)
centroid = analysis.spectral_centroid(signal, 44100)
rolloff = analysis.spectral_rolloff(signal, 44100)

# Onset detection
onsets = analysis.detect_onsets(signal, 44100)
print(f"Onset times: {onsets}")

# Tempo detection
tempo = analysis.detect_tempo(signal, 44100)
print(f"Detected tempo: {tempo} BPM")

# Spectrogram
times, freqs, spec = analysis.spectrogram(signal, 44100, window_size=2048)
```

### Using New Presets

```python
import sprechstimme as sp

# Advanced synthesis presets
sp.new("fm_bell", preset="fm_bell")
sp.new("supersaw", preset="supersaw")
sp.new("pwm_pad", preset="pwm_pad")
sp.new("hard_sync", preset="hard_sync_lead")
sp.new("unison", preset="unison_saw")

# Specialized presets
sp.new("moog", preset="moog_bass")
sp.new("resonant", preset="resonant_bass")
sp.new("vocal", preset="formant_vocal")
sp.new("pluck", preset="plucked_string")

# Use them
sp.play("supersaw", "C3", duration=2.0)
sp.play("vocal", ["C4", "E4", "G4"], duration=1.5)

# List all available presets
print(sp.presets.list_presets())
```

### Complete Production Example

```python
import sprechstimme as sp
from sprechstimme import effects, utils

# Create song
song = sp.Song(bpm=120)

# Add instruments
sp.new("bass", preset="moog_bass")
sp.new("lead", preset="supersaw")
sp.new("pad", preset="pwm_pad")

# Compose
song.add("bass", "C2", beat=0, duration=1, vol=7)
song.add("lead", "C4", beat=1, duration=0.5, vol=6)
song.add("pad", "C4", beat=0, duration=4, vol=4)  # Chord

# Render
audio = song.render()

# Apply mastering effects
audio = effects.chorus(audio, 44100, mix=0.3)
audio = effects.reverb(audio, 44100, room_size=0.5, wet=0.25)
audio = effects.compressor(audio, 44100, threshold=-15, ratio=4.0)
audio = utils.normalize(audio, target_level=0.95)

# Export
sp.playback.export_wav(audio, "master.wav", 44100)
```

For more examples, see `examples/advanced_features.py` in the repository.

## API Reference

### Core Functions

#### `new(name)`
Register a new synthesizer with default settings.

**Parameters:**
- `name` (str): Name identifier for the synth

#### `create(name, wavetype=None, filters=None, envelope=None, poly=True)`
Configure a synthesizer with specific parameters.

**Parameters:**
- `name` (str): Name of the synth to configure
- `wavetype` (callable or str): Waveform generator function or preset name ("sine", "square", "sawtooth", "triangle", "noise")
- `filters` (list): List of filter functions or preset names
- `envelope` (dict): ADSR envelope parameters `{"attack": float, "decay": float, "sustain": float, "release": float}`
- `poly` (bool): Enable polyphony (default: True)

#### `play(name, notes, duration=0.5, sample_rate=44100)`
Play notes on a synthesizer.

**Parameters:**
- `name` (str): Synth name
- `notes`: Single note (int/float/str), list of notes (chord), or list of (note, duration) tuples (sequence)
- `duration` (float): Duration in seconds (default: 0.5)
- `sample_rate` (int): Sample rate in Hz (default: 44100)

#### `get(name)`
Get configuration summary for a synth.

**Returns:** Dictionary with synth configuration

### Utility Functions

#### `midi_to_freq(midi_note)`
Convert MIDI note number to frequency in Hz.

**Parameters:**
- `midi_note` (int): MIDI note number (0-127)

**Returns:** float (frequency in Hz)

#### `note_to_freq(note)`
Convert note name to frequency.

**Parameters:**
- `note` (str): Note name (e.g., "C4", "A#3", "Bb2")

**Returns:** float (frequency in Hz)

### Track Class

#### `Track(bpm=120, sample_rate=44100)`
Create a new musical track/sequence.

**Parameters:**
- `bpm` (int): Beats per minute (default: 120)
- `sample_rate` (int): Sample rate in Hz (default: 44100)

#### `Track.add(synth, notes, duration=1)`
Add an event to the track.

**Parameters:**
- `synth` (str): Name of synth to use
- `notes`: Note(s) to play (int/float/str or list)
- `duration` (float): Duration in beats (default: 1)

#### `Track.addChord(synth, chord, duration=1)`
Add a chord to the track using chord notation.

**Parameters:**
- `synth` (str): Name of synth to use
- `chord` (str): Chord notation (e.g., "C4", "Am3", "F#5")
- `duration` (float): Duration in beats (default: 1)

#### `Track.play()`
Play the track through audio output.

#### `Track.export(filename="output.wav")`
Export track to a WAV file.

**Parameters:**
- `filename` (str): Output filename (default: "output.wav")

### Song Class

#### `Song(bpm=120, sample_rate=44100)`
Create a new multi-track song with universal beat counter.

**Parameters:**
- `bpm` (int): Beats per minute for the entire song (default: 120)
- `sample_rate` (int): Sample rate in Hz (default: 44100)

#### `Song.add_track(track_name)`
Create a new track in the song.

**Parameters:**
- `track_name` (str): Unique identifier for the track

#### `Song.add(track_name, synth, notes, beat_position, duration=1)`
Add an event to a specific track at a specific beat position.

**Parameters:**
- `track_name` (str): Name of the track to add to (auto-created if doesn't exist)
- `synth` (str): Name of the synthesizer to use
- `notes`: Note(s) to play (int/float/str or list for chords)
- `beat_position` (float): When to start this event in beats (can be fractional like 200.5)
- `duration` (float): Duration of the event in beats (default: 1)

#### `Song.add_chord(track_name, synth, chord, beat_position, duration=1)`
Add a chord event to a specific track at a specific beat position.

**Parameters:**
- `track_name` (str): Name of the track to add to
- `synth` (str): Name of the synthesizer to use
- `chord` (str): Chord notation (e.g., "C4", "Am3", "F#5")
- `beat_position` (float): When to start this event in beats
- `duration` (float): Duration of the event in beats (default: 1)

#### `Song.play()`
Render and play the entire song through audio output.

#### `Song.export(filename="output.wav")`
Render and export the song to a WAV file.

**Parameters:**
- `filename` (str): Output filename (default: "output.wav")

#### `Song.get_duration()`
Get the total duration of the song.

**Returns:** Dictionary with `beats`, `seconds`, and `formatted` time

#### `Song.list_tracks()`
Get information about all tracks in the song.

**Returns:** Dictionary with track names, event counts, and synths used

### Waveform Generators

All waveform functions have the signature: `func(t, freq=440, amp=1.0)`

- `sp.waves.sine` - Pure sine wave
- `sp.waves.square` - Square wave
- `sp.waves.sawtooth` - Sawtooth wave
- `sp.waves.triangle` - Triangle wave
- `sp.waves.noise` - White noise

### Filters

All filters accept `signal` and `sample_rate` as first two parameters.

#### `sp.filters.low_pass(signal, sample_rate, cutoff=1000.0, order=4)`
Butterworth low-pass filter.

#### `sp.filters.high_pass(signal, sample_rate, cutoff=200.0, order=4)`
Butterworth high-pass filter.

#### `sp.filters.band_pass(signal, sample_rate, low=300.0, high=3000.0, order=4)`
Butterworth band-pass filter.

#### `sp.filters.echo(signal, sample_rate, delay=0.25, decay=0.4)`
Echo/delay effect.

#### `sp.filters.simple_distortion(signal, sample_rate, gain=1.0, threshold=0.8)`
Clipping-based distortion effect.

### Filter Helpers

For convenience, use these to create partial functions with custom parameters:

```python
from functools import partial

# Low-pass with custom cutoff
sp.filters.lp(cutoff=500)

# High-pass with custom cutoff
sp.filters.hp(cutoff=100)

# Band-pass with custom range
sp.filters.bp(low=200, high=2000)

# Echo with custom timing
sp.filters.delay(delay=0.5, decay=0.6)

# Distortion with custom gain
sp.filters.dist(gain=2.0, threshold=0.7)
```

## Examples

### Creating a Simple Melody

```python
import sprechstimme as sp

sp.new("melody")
sp.create("melody", wavetype="triangle", envelope={
    "attack": 0.05,
    "decay": 0.1,
    "sustain": 0.6,
    "release": 0.3
})

notes = [
    ("C4", 0.5),
    ("D4", 0.5),
    ("E4", 0.5),
    ("F4", 0.5),
    ("G4", 1.0)
]

for note, dur in notes:
    sp.play("melody", note, duration=dur)
```

### Building a Multi-Track Composition

```python
import sprechstimme as sp
from functools import partial

# Bass synth
sp.new("bass")
sp.create("bass",
    wavetype="sawtooth",
    filters=[partial(sp.filters.low_pass, cutoff=200)],
    envelope={"attack": 0.01, "decay": 0.1, "sustain": 0.8, "release": 0.2}
)

# Lead synth
sp.new("lead")
sp.create("lead",
    wavetype="square",
    filters=[
        partial(sp.filters.low_pass, cutoff=2000),
        partial(sp.filters.echo, delay=0.25, decay=0.3)
    ],
    envelope={"attack": 0.02, "decay": 0.05, "sustain": 0.7, "release": 0.3}
)

# Create composition
track = sp.Track(bpm=128)

# Bass line
track.add("bass", "C2", duration=1)
track.add("bass", "C2", duration=1)
track.add("bass", "G1", duration=1)
track.add("bass", "G1", duration=1)

# Melody
track.add("lead", "C4", duration=0.5)
track.add("lead", "E4", duration=0.5)
track.add("lead", "G4", duration=0.5)
track.add("lead", "C5", duration=0.5)

track.play()
track.export("composition.wav")
```

## Note Format Support

Sprechstimme accepts notes in multiple formats:

- **MIDI Numbers**: `60` (middle C), `69` (A4 = 440 Hz)
- **Frequencies**: `440.0` (A4), `261.63` (C4)
- **Note Names**: `"C4"`, `"A#3"`, `"Bb2"` (supports # for sharp, b for flat)

## License

Apache License 2.0

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests on [GitHub](https://github.com/breatn/sprechstimme).

## Credits

Created by liquidsound
