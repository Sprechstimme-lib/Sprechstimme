import numpy as np
from . import waves, playback

_SYNTHS = {}

def new(name):
    _SYNTHS[name] = {"wavetype": waves.sine, "filters": []}

def create(name, wavetype=waves.sine, filters=None):
    if name not in _SYNTHS:
        raise ValueError(f"Synth '{name}' does not exist. Call sprechstimme.new() first.")
    _SYNTHS[name]["wavetype"] = wavetype
    _SYNTHS[name]["filters"] = filters or []

def play(name, notes, duration=0.5, sample_rate=44100):
    """Play notes (list of MIDI note numbers or Hz) on synth."""
    if isinstance(notes, (int, float, str)):
        notes = [notes]

    synth = _SYNTHS.get(name)
    if synth is None:
        raise ValueError(f"Synth '{name}' not found")

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.zeros_like(t)

    for n in notes:
        freq = midi_to_freq(n) if isinstance(n, (int, float)) else note_to_freq(n)
        signal += synth["wavetype"](t, freq=freq)

    # normalize
    signal /= len(notes)
    playback.play_array(signal, sample_rate)

def midi_to_freq(midi_note):
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

NOTE_MAP = {
    "C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
    "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11
}

def note_to_freq(note):
    # Example: "C4" -> 261.63 Hz
    name = note[:-1]
    octave = int(note[-1])
    midi = NOTE_MAP[name.upper()] + (octave + 1) * 12
    return midi_to_freq(midi)