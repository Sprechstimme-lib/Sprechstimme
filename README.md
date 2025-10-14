# Sprechstimme

A modular Python synthesizer and sequencer for creating music programmatically.

[![PyPI version](https://badge.fury.io/py/sprechstimme.svg)](https://badge.fury.io/py/sprechstimme)
[![Python](https://img.shields.io/pypi/pyversions/sprechstimme.svg)](https://pypi.org/project/sprechstimme/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

Sprechstimme is a lightweight, flexible Python library for audio synthesis and music sequencing. It provides an intuitive API for creating custom synthesizers, applying audio filters, and composing musical sequences—all with just a few lines of code.

## Features

- **Multiple Waveform Types**: Sine, square, sawtooth, triangle, and noise generators
- **Audio Filters**: Low-pass, high-pass, band-pass, echo, and distortion effects
- **ADSR Envelope**: Attack, Decay, Sustain, Release envelope shaping
- **Flexible Note Input**: Support for MIDI numbers, frequencies (Hz), and note names (e.g., "C4", "A#3")
- **Chord Support**: Built-in chord notation for major and minor triads
- **Track Sequencing**: Create multi-note sequences with BPM-based timing
- **Real-time Playback**: Instant audio playback via sounddevice
- **WAV Export**: Save your compositions to standard WAV files

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

View AUTHORS in github
