from .core import new, create, play, get, midi_to_freq, note_to_freq
from .track import Track
from .song import Song
from . import waves, filters, presets, effects, modulation, utils, wavetables, midi_utils, analysis

__all__ = [
    "new", "create", "play", "get",
    "Track", "Song",
    "waves", "filters", "presets", "effects", "modulation",
    "utils", "wavetables", "midi_utils", "analysis",
    "midi_to_freq", "note_to_freq"
]

__version__ = "2.0.0"
