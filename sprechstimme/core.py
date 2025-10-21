import numpy as np
import re
from functools import partial
from . import waves as waves_module
from . import filters as filters_module
from . import playback

# internal synth registry
_SYNTHS = {}

# Default sample rate
DEFAULT_SR = 44100

def new(name):
    """Register a new synth with defaults (sine, no filters)."""
    _SYNTHS[name] = {
        "wavetype": waves_module.sine,
        "filters": [],
        "envelope": None,
        "poly": True
    }

def _resolve_wave(wavetype):
    """Allow wavetype to be function or string preset name."""
    if isinstance(wavetype, str):
        return getattr(waves_module, wavetype, None) or waves_module.sine
    if callable(wavetype):
        return wavetype
    raise ValueError("wavetype must be a function or preset string")

def _resolve_filter(filt):
    """
    Accept:
     - callable filter(signal, sample_rate, **kwargs)
     - string name -> lookup in filters_module
     - functools.partial to carry parameters
    Return a callable f(signal, sample_rate) -> signal
    """
    if filt is None:
        return None
    if isinstance(filt, str):
        f = getattr(filters_module, filt, None)
        if f is None:
            raise ValueError(f"Unknown filter preset '{filt}'")
        return f
    if callable(filt):
        return filt
    raise ValueError("Filter must be callable, string preset, or partial")

def create(name, wavetype=None, filters=None, envelope=None, poly=True):
    """
    Configure synth 'name'.
    wavetype: function(t, freq=440, amp=1.0) or string preset
    filters: list of callables / preset names / partials
    envelope: dict or None, e.g. {"attack":0.01,"decay":0.05,"sustain":0.8,"release":0.1}
    poly: allow polyphony (True) or single voice (False)
    """
    if name not in _SYNTHS:
        raise ValueError(f"Synth '{name}' does not exist. Call new() first.")
    if wavetype is not None:
        _SYNTHS[name]["wavetype"] = _resolve_wave(wavetype)
    # resolve filters
    resolved = []
    if filters:
        for f in filters:
            # Allow passing partials; partial is callable so _resolve_filter handles it.
            resolved.append(_resolve_filter(f))
    _SYNTHS[name]["filters"] = [r for r in resolved if r is not None]
    # envelope
    _SYNTHS[name]["envelope"] = envelope
    _SYNTHS[name]["poly"] = bool(poly)

def get(name):
    """Return synth configuration summary."""
    synth = _SYNTHS.get(name)
    if not synth:
        raise ValueError(f"Synth '{name}' not found")
    def _name(obj):
        if obj is None:
            return None
        if hasattr(obj, "__name__"):
            return obj.__name__
        return str(obj)
    return {
        "wavetype": _name(synth["wavetype"]),
        "filters": [_name(f) for f in synth["filters"]],
        "envelope": synth["envelope"],
        "polyphony": synth["poly"]
    }

def midi_to_freq(midi_note):
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

def note_to_freq(note):
    # Accept "C4", "A#3", or "Bb3" (supports b for flat)
    NOTE_MAP = {
        "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5,
        "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11
    }
    note = note.strip().upper()
    # handle multi-char note name
    if len(note) < 2:
        raise ValueError("note must be like 'C4' or 'A#3'")
    # last character(s) are octave digits (allow negative octave e.g. C-1)
    # find where digits start
    i = len(note) - 1
    while i >= 0 and (note[i].isdigit() or note[i] == "-"):
        i -= 1
    name = note[:i+1]
    octave = int(note[i+1:])
    if name not in NOTE_MAP:
        raise ValueError(f"Unknown note name '{name}'")
    midi = NOTE_MAP[name] + (octave + 1) * 12
    return midi_to_freq(midi)

def chord_to_notes(chord_name):
    NOTE_BASES = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8,
        'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
    }
    m = re.match(r'^([A-G][b#]?)(m?)(\d+)$', chord_name)
    if not m:
        raise ValueError(f"Invalid chord name: {chord_name}")
    root, minor, octave = m.groups()
    octave = int(octave)
    root_midi = 12 + NOTE_BASES[root] + 12 * octave
    if minor:
        intervals = [0, 3, 7]  # minor triad
    else:
        intervals = [0, 4, 7]  # major triad
    return [root_midi + i for i in intervals]


def _apply_envelope(signal, sample_rate, env):
    """Simple ADSR envelope applied per-sample."""
    if not env:
        return signal
    a = float(env.get("attack", 0.01))
    d = float(env.get("decay", 0.05))
    s = float(env.get("sustain", 0.8))
    r = float(env.get("release", 0.05))
    total_len = len(signal)
    sr = sample_rate
    attack_samples = int(a * sr)
    decay_samples = int(d * sr)
    release_samples = int(r * sr)
    sustain_samples = max(0, total_len - (attack_samples + decay_samples + release_samples))
    # build envelope
    env_curve = np.zeros(total_len, dtype=float)
    idx = 0
    # attack
    if attack_samples > 0 and idx < total_len:
        actual_attack = min(attack_samples, total_len - idx)
        env_curve[idx:idx+actual_attack] = np.linspace(0.0, 1.0, actual_attack, endpoint=False)
        idx += actual_attack
    # decay
    if decay_samples > 0 and idx < total_len:
        actual_decay = min(decay_samples, total_len - idx)
        env_curve[idx:idx+actual_decay] = np.linspace(1.0, s, actual_decay, endpoint=False)
        idx += actual_decay
    # sustain
    if sustain_samples > 0 and idx < total_len:
        actual_sustain = min(sustain_samples, total_len - idx)
        env_curve[idx:idx+actual_sustain] = s
        idx += actual_sustain
    # release
    if release_samples > 0 and idx < total_len:
        actual_release = min(release_samples, total_len - idx)
        env_curve[idx:idx+actual_release] = np.linspace(s, 0.0, actual_release, endpoint=False)
        idx += actual_release
    # if idx < total_len fill remainder with 0
    env_curve[idx:] = 0.0
    return signal * env_curve

def play(name, notes, duration=0.5, sample_rate=DEFAULT_SR, vol=5.0):
    """
    Play notes on synth `name`.
    notes may be:
      - single int (MIDI) or float (Hz) or str ("C4")
      - list of notes to be treated as a chord (played together)
      - list of tuples for sequences: [(60,0.5),(64,0.5)] or [(60,0.5,3.5),(64,0.5,7.2)]
    duration is seconds for each grouped event unless provided per-note in tuple form.
    vol is dynamics from 1.0 to 10.0 (default 5.0), can be provided globally or per-note in tuple form.
    """
    # Normalize notes into a list
    if isinstance(notes, (int, float, str)):
        notes = [notes]

    synth = _SYNTHS.get(name)
    if synth is None:
        raise ValueError(f"Synth '{name}' not found")

    # If user passed sequence of (note, dur) or (note, dur, vol) tuples, handle separately:
    # detect tuple-like: first element is tuple/list of length 2 or 3 with number/str
    if len(notes) > 0 and isinstance(notes[0], (list, tuple)) and len(notes[0]) in [2, 3]:
        # treat as sequence of (note, dur) or (note, dur, vol)
        segments = []
        for item in notes:
            if len(item) == 2:
                n, dur = item
                segments.append(( [n], dur, vol))
            elif len(item) == 3:
                n, dur, note_vol = item
                segments.append(( [n], dur, note_vol))
    else:
        # Single chord or list-of-notes for same duration
        segments = [ (notes, duration, vol) ]

    out = np.array([], dtype=float)
    for seg_notes, seg_dur, seg_vol in segments:
        sec = float(seg_dur)
        # Normalize dynamics from 1-10 range to 0.1-1.0 amplitude range
        amplitude = np.clip(float(seg_vol), 1.0, 10.0) / 10.0
        t = np.linspace(0, sec, int(sample_rate * sec), endpoint=False)
        signal = np.zeros_like(t)

        for n in seg_notes:
            # Determine frequency
            if isinstance(n, int):
                freq = midi_to_freq(n)
            elif isinstance(n, float):
                # float treated as frequency in Hz
                freq = float(n)
            elif isinstance(n, str):
                freq = note_to_freq(n)
            else:
                raise ValueError(f"Unsupported note type: {type(n)} -> {n}")

            wave = synth["wavetype"](t, freq=freq, amp=amplitude)
            signal += wave

        # normalize chord by number of notes to avoid clipping
        if len(seg_notes) > 0:
            signal = signal / float(len(seg_notes))

        # apply envelope if configured
        if synth.get("envelope"):
            signal = _apply_envelope(signal, sample_rate, synth["envelope"])

        # apply filters in order
        for f in synth.get("filters", []):
            if f:
                # filters expected to be f(signal, sample_rate)
                signal = f(signal, sample_rate)

        out = np.concatenate([out, signal])

    # final normalization to avoid clipping in playback
    max_abs = np.max(np.abs(out)) if out.size else 0.0
    if max_abs > 1e-9:
        out = out / max(1.0, max_abs)

    playback.play_array(out, sample_rate)
