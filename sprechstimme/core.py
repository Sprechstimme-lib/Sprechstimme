import numpy as np
from . import waves, playback

_SYNTHS = {}

def new(name):
    _SYNTHS[name] = {"wavetype": waves.sine, "filters": []}

def create(name, wavetype=waves.sine, filters=None):
    """
    wavetype: function(t, freq, amp) or string (preset)
    filters: list of functions or strings (presets)
    """
    if name not in _SYNTHS:
        raise ValueError(f"Synth '{name}' does not exist. Call sprechstimme.new() first.")
    # Allow string for preset wave
    if isinstance(wavetype, str):
        wavetype = getattr(waves, wavetype, waves.sine)
    _SYNTHS[name]["wavetype"] = wavetype
    # Allow filter presets by string
    from . import filters as _filters
    _SYNTHS[name]["filters"] = []
    if filters:
        for f in filters:
            if isinstance(f, str):
                _SYNTHS[name]["filters"].append(getattr(_filters, f, None))
            else:
                _SYNTHS[name]["filters"].append(f)

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
        wave = synth["wavetype"](t, freq=freq)
        # Apply filters
        for f in synth["filters"]:
            if f:
                wave = f(wave, sample_rate)
        signal += wave

    # normalize
    signal /= len(notes)
    playback.play_array(signal, sample_rate)

def get(name):
    """Return info about synth: wave and filters."""
    synth = _SYNTHS.get(name)
    if not synth:
        raise ValueError(f"Synth '{name}' not found")
    wave = synth["wavetype"]
    filters = synth["filters"]
    def _func_name(f):
        if hasattr(f, "__name__"):
            return f.__name__
        return str(f)
    return {
        "wave": _func_name(wave),
        "filters": [_func_name(f) for f in filters if f]
    }

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