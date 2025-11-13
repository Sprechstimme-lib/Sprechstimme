"""MIDI utilities for note handling and conversion."""
import numpy as np


# MIDI note number to frequency conversion
def midi_to_freq(midi_note):
    """
    Convert MIDI note number to frequency in Hz.

    Args:
        midi_note: MIDI note number (0-127) or array of notes

    Returns:
        Frequency in Hz
    """
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def freq_to_midi(freq):
    """
    Convert frequency to MIDI note number.

    Args:
        freq: Frequency in Hz

    Returns:
        MIDI note number (float)
    """
    return 69 + 12 * np.log2(freq / 440.0)


def note_to_midi(note_name):
    """
    Convert note name to MIDI number.

    Args:
        note_name: Note name (e.g., "C4", "A#3", "Bb5")

    Returns:
        MIDI note number
    """
    note_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}

    note_name = note_name.strip().upper()

    # Parse note
    note = note_name[0]
    rest = note_name[1:]

    # Parse accidental
    accidental = 0
    if rest.startswith('#'):
        accidental = 1
        rest = rest[1:]
    elif rest.startswith('B'):
        accidental = -1
        rest = rest[1:]

    # Parse octave
    try:
        octave = int(rest)
    except ValueError:
        octave = 4  # Default octave

    midi = note_map[note] + accidental + (octave + 1) * 12
    return midi


def midi_to_note(midi_number, use_sharps=True):
    """
    Convert MIDI number to note name.

    Args:
        midi_number: MIDI note number
        use_sharps: Use sharps (True) or flats (False) for accidentals

    Returns:
        Note name string
    """
    notes_sharp = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    notes_flat = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

    notes = notes_sharp if use_sharps else notes_flat

    octave = (midi_number // 12) - 1
    note = notes[midi_number % 12]

    return f"{note}{octave}"


def note_to_freq(note_name):
    """
    Convert note name directly to frequency.

    Args:
        note_name: Note name (e.g., "C4", "A#3")

    Returns:
        Frequency in Hz
    """
    return midi_to_freq(note_to_midi(note_name))


# Scale and chord utilities
def get_scale_intervals(scale_type='major'):
    """
    Get intervals for a scale type.

    Args:
        scale_type: Scale type name

    Returns:
        List of semitone intervals
    """
    scales = {
        'major': [0, 2, 4, 5, 7, 9, 11],
        'minor': [0, 2, 3, 5, 7, 8, 10],
        'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
        'melodic_minor': [0, 2, 3, 5, 7, 9, 11],
        'dorian': [0, 2, 3, 5, 7, 9, 10],
        'phrygian': [0, 1, 3, 5, 7, 8, 10],
        'lydian': [0, 2, 4, 6, 7, 9, 11],
        'mixolydian': [0, 2, 4, 5, 7, 9, 10],
        'locrian': [0, 1, 3, 5, 6, 8, 10],
        'pentatonic_major': [0, 2, 4, 7, 9],
        'pentatonic_minor': [0, 3, 5, 7, 10],
        'blues': [0, 3, 5, 6, 7, 10],
        'chromatic': list(range(12)),
        'whole_tone': [0, 2, 4, 6, 8, 10],
        'diminished': [0, 2, 3, 5, 6, 8, 9, 11],
    }
    return scales.get(scale_type, scales['major'])


def generate_scale(root_note, scale_type='major', octaves=1):
    """
    Generate a scale.

    Args:
        root_note: Root note (MIDI number, frequency, or note name)
        scale_type: Scale type
        octaves: Number of octaves

    Returns:
        List of MIDI note numbers
    """
    # Convert root to MIDI
    if isinstance(root_note, str):
        root_midi = note_to_midi(root_note)
    elif isinstance(root_note, float):
        root_midi = int(freq_to_midi(root_note))
    else:
        root_midi = root_note

    intervals = get_scale_intervals(scale_type)
    scale = []

    for octave in range(octaves):
        for interval in intervals:
            scale.append(root_midi + interval + octave * 12)

    # Add final root note
    scale.append(root_midi + octaves * 12)

    return scale


def get_chord_intervals(chord_type='major'):
    """
    Get intervals for a chord type.

    Args:
        chord_type: Chord type name

    Returns:
        List of semitone intervals
    """
    chords = {
        'major': [0, 4, 7],
        'minor': [0, 3, 7],
        'diminished': [0, 3, 6],
        'augmented': [0, 4, 8],
        'sus2': [0, 2, 7],
        'sus4': [0, 5, 7],
        'major7': [0, 4, 7, 11],
        'minor7': [0, 3, 7, 10],
        'dom7': [0, 4, 7, 10],
        'maj7': [0, 4, 7, 11],
        'min7': [0, 3, 7, 10],
        'dim7': [0, 3, 6, 9],
        'aug7': [0, 4, 8, 10],
        'maj9': [0, 4, 7, 11, 14],
        'min9': [0, 3, 7, 10, 14],
        'dom9': [0, 4, 7, 10, 14],
        '6': [0, 4, 7, 9],
        'min6': [0, 3, 7, 9],
        'add9': [0, 4, 7, 14],
    }
    return chords.get(chord_type, chords['major'])


def generate_chord(root_note, chord_type='major'):
    """
    Generate a chord.

    Args:
        root_note: Root note (MIDI number, frequency, or note name)
        chord_type: Chord type

    Returns:
        List of MIDI note numbers
    """
    # Convert root to MIDI
    if isinstance(root_note, str):
        root_midi = note_to_midi(root_note)
    elif isinstance(root_note, float):
        root_midi = int(freq_to_midi(root_note))
    else:
        root_midi = root_note

    intervals = get_chord_intervals(chord_type)
    return [root_midi + interval for interval in intervals]


def parse_chord_notation(chord_str):
    """
    Parse chord notation string.

    Args:
        chord_str: Chord string (e.g., "Cmaj7", "Am", "F#dim")

    Returns:
        List of MIDI note numbers
    """
    chord_str = chord_str.strip()

    # Extract root note
    if len(chord_str) > 1 and chord_str[1] in ['#', 'b']:
        root = chord_str[:2]
        suffix = chord_str[2:]
    else:
        root = chord_str[0]
        suffix = chord_str[1:]

    # Parse suffix to determine chord type
    suffix = suffix.lower()

    if suffix in ['', 'maj', 'major']:
        chord_type = 'major'
    elif suffix in ['m', 'min', 'minor']:
        chord_type = 'minor'
    elif suffix in ['dim', 'diminished']:
        chord_type = 'diminished'
    elif suffix in ['aug', 'augmented']:
        chord_type = 'augmented'
    elif 'maj7' in suffix:
        chord_type = 'major7'
    elif 'min7' in suffix or 'm7' in suffix:
        chord_type = 'minor7'
    elif '7' in suffix:
        chord_type = 'dom7'
    elif 'sus4' in suffix:
        chord_type = 'sus4'
    elif 'sus2' in suffix:
        chord_type = 'sus2'
    else:
        chord_type = 'major'

    return generate_chord(root, chord_type)


# Rhythm utilities
def bpm_to_spb(bpm):
    """Convert BPM to seconds per beat."""
    return 60.0 / bpm


def spb_to_bpm(spb):
    """Convert seconds per beat to BPM."""
    return 60.0 / spb


def beats_to_seconds(beats, bpm):
    """Convert beats to seconds."""
    return beats * bpm_to_spb(bpm)


def seconds_to_beats(seconds, bpm):
    """Convert seconds to beats."""
    return seconds / bpm_to_spb(bpm)


# Quantization
def quantize_to_grid(time, grid_size):
    """
    Quantize time value to grid.

    Args:
        time: Time value
        grid_size: Grid size (e.g., 0.25 for 16th notes)

    Returns:
        Quantized time
    """
    return round(time / grid_size) * grid_size


def swing(grid_times, swing_amount=0.5, grid_size=0.5):
    """
    Apply swing to timing grid.

    Args:
        grid_times: Array of grid times
        swing_amount: Swing amount (0.0 = no swing, 1.0 = maximum swing)
        grid_size: Grid size (typically 0.5 for eighth notes)

    Returns:
        Swung times
    """
    swung = np.copy(grid_times)

    for i in range(len(swung)):
        # Apply swing to every other note
        if i % 2 == 1:
            offset = swing_amount * grid_size * 0.33
            swung[i] += offset

    return swung


# Velocity utilities
def velocity_to_amplitude(velocity, curve='linear'):
    """
    Convert MIDI velocity (0-127) to amplitude (0.0-1.0).

    Args:
        velocity: MIDI velocity (0-127)
        curve: Response curve ('linear', 'exponential', 'logarithmic')

    Returns:
        Amplitude (0.0 to 1.0)
    """
    normalized = velocity / 127.0

    if curve == 'exponential':
        return normalized ** 2
    elif curve == 'logarithmic':
        return np.sqrt(normalized)
    else:  # linear
        return normalized


def amplitude_to_velocity(amplitude, curve='linear'):
    """
    Convert amplitude (0.0-1.0) to MIDI velocity (0-127).

    Args:
        amplitude: Amplitude (0.0-1.0)
        curve: Response curve

    Returns:
        MIDI velocity (0-127)
    """
    if curve == 'exponential':
        normalized = np.sqrt(amplitude)
    elif curve == 'logarithmic':
        normalized = amplitude ** 2
    else:  # linear
        normalized = amplitude

    return int(normalized * 127)


# Random note generation
def random_melody(root_note, scale_type='major', length=8, octave_range=2):
    """
    Generate random melody from scale.

    Args:
        root_note: Root note
        scale_type: Scale type
        length: Number of notes
        octave_range: Octave range

    Returns:
        List of MIDI note numbers
    """
    scale = generate_scale(root_note, scale_type, octave_range)
    return list(np.random.choice(scale, size=length))


def random_chord_progression(key, length=4, allowed_chords=None):
    """
    Generate random chord progression.

    Args:
        key: Key (note name)
        length: Number of chords
        allowed_chords: List of chord types to choose from

    Returns:
        List of chord lists (each chord is a list of MIDI notes)
    """
    if allowed_chords is None:
        allowed_chords = ['major', 'minor', 'dom7', 'minor7']

    # Common scale degrees for progressions
    scale = generate_scale(key, 'major', octaves=1)
    degrees = [0, 2, 3, 4, 5, 7]  # I, ii, iii, IV, V, vi

    progression = []
    for _ in range(length):
        degree = np.random.choice(degrees)
        chord_type = np.random.choice(allowed_chords)
        root = scale[degree]
        progression.append(generate_chord(root, chord_type))

    return progression
