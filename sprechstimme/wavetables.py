"""Wavetable synthesis with preset wavetables."""
import numpy as np


class Wavetable:
    """Wavetable oscillator for wavetable synthesis."""

    def __init__(self, table, sample_rate=44100):
        """
        Args:
            table: Numpy array containing one cycle of the waveform
            sample_rate: Sample rate in Hz
        """
        self.table = np.array(table)
        self.sample_rate = sample_rate

    def generate(self, t, freq=440.0, amp=1.0):
        """
        Generate audio from wavetable.

        Args:
            t: Time array
            freq: Frequency in Hz (can be array for FM)
            amp: Amplitude

        Returns:
            Generated audio
        """
        # Calculate phase
        phase = (t * freq) % 1.0

        # Linear interpolation lookup
        table_size = len(self.table)
        indices = phase * table_size
        indices_int = indices.astype(int) % table_size
        indices_next = (indices_int + 1) % table_size
        frac = indices - indices_int

        # Interpolate
        output = (1 - frac) * self.table[indices_int] + frac * self.table[indices_next]

        return amp * output

    def __call__(self, t, freq=440.0, amp=1.0):
        """Allow wavetable to be called as a function."""
        return self.generate(t, freq, amp)


class WavetableBank:
    """Bank of wavetables for smooth morphing."""

    def __init__(self, tables):
        """
        Args:
            tables: List of wavetables (numpy arrays)
        """
        self.tables = [np.array(t) for t in tables]

    def morph(self, t, freq=440.0, amp=1.0, position=0.0):
        """
        Morph between wavetables.

        Args:
            t: Time array
            freq: Frequency
            amp: Amplitude
            position: Morph position (0.0 to len(tables)-1)

        Returns:
            Morphed audio
        """
        position = np.clip(position, 0, len(self.tables) - 1)
        idx1 = int(position)
        idx2 = min(idx1 + 1, len(self.tables) - 1)
        alpha = position - idx1

        wt1 = Wavetable(self.tables[idx1])
        wt2 = Wavetable(self.tables[idx2])

        return (1 - alpha) * wt1.generate(t, freq, amp) + alpha * wt2.generate(t, freq, amp)


# Preset wavetable generators
def generate_sine_table(size=2048):
    """Generate sine wave table."""
    phase = np.linspace(0, 2 * np.pi, size, endpoint=False)
    return np.sin(phase)


def generate_saw_table(size=2048, harmonics=32):
    """Generate bandlimited sawtooth table."""
    table = np.zeros(size)
    phase = np.linspace(0, 2 * np.pi, size, endpoint=False)

    for n in range(1, harmonics + 1):
        table += ((-1) ** (n + 1)) * np.sin(n * phase) / n

    return table / np.max(np.abs(table))


def generate_square_table(size=2048, harmonics=32):
    """Generate bandlimited square wave table."""
    table = np.zeros(size)
    phase = np.linspace(0, 2 * np.pi, size, endpoint=False)

    for n in range(1, harmonics + 1, 2):
        table += np.sin(n * phase) / n

    return table / np.max(np.abs(table))


def generate_triangle_table(size=2048, harmonics=32):
    """Generate bandlimited triangle wave table."""
    table = np.zeros(size)
    phase = np.linspace(0, 2 * np.pi, size, endpoint=False)

    for n in range(1, harmonics + 1, 2):
        table += ((-1) ** ((n - 1) // 2)) * np.sin(n * phase) / (n * n)

    return table / np.max(np.abs(table))


def generate_pulse_table(size=2048, duty=0.5, harmonics=32):
    """Generate bandlimited pulse wave table."""
    table = np.zeros(size)
    phase = np.linspace(0, 2 * np.pi, size, endpoint=False)

    for n in range(1, harmonics + 1):
        table += (np.sin(n * np.pi * duty) / (n * np.pi)) * np.cos(n * phase)

    return table / np.max(np.abs(table))


def generate_harmonic_table(size=2048, harmonic_levels=None):
    """
    Generate table from harmonic levels (additive synthesis).

    Args:
        size: Table size
        harmonic_levels: List of harmonic amplitudes [1st, 2nd, 3rd, ...]
    """
    if harmonic_levels is None:
        harmonic_levels = [1.0, 0.5, 0.25, 0.125]

    table = np.zeros(size)
    phase = np.linspace(0, 2 * np.pi, size, endpoint=False)

    for n, level in enumerate(harmonic_levels, start=1):
        table += level * np.sin(n * phase)

    return table / np.max(np.abs(table))


def generate_noise_table(size=2048):
    """Generate noise table."""
    return np.random.uniform(-1, 1, size)


def generate_vowel_table(size=2048, vowel='a'):
    """
    Generate vowel formant table.

    Args:
        size: Table size
        vowel: Vowel type ('a', 'e', 'i', 'o', 'u')
    """
    # Formant frequencies for different vowels (simplified)
    formants = {
        'a': [730, 1090, 2440],  # "ah"
        'e': [530, 1840, 2480],  # "eh"
        'i': [270, 2290, 3010],  # "ee"
        'o': [570, 840, 2410],   # "oh"
        'u': [440, 1020, 2240],  # "oo"
    }

    freqs = formants.get(vowel, formants['a'])

    # Create base sawtooth
    table = generate_saw_table(size)

    # Add formant peaks (simplified)
    phase = np.linspace(0, 2 * np.pi, size, endpoint=False)
    formant_wave = np.zeros(size)

    for freq in freqs:
        harmonic = int(freq / 100)  # Approximate harmonic number
        formant_wave += 0.3 * np.sin(harmonic * phase)

    table = table * (1.0 + formant_wave)

    return table / np.max(np.abs(table))


def generate_ppg_table(size=2048, wave_number=1):
    """
    Generate PPG-style wavetables (inspired by PPG Wave synthesizer).

    Args:
        size: Table size
        wave_number: Wave number (1-64)
    """
    phase = np.linspace(0, 2 * np.pi, size, endpoint=False)

    # Various PPG-inspired waves
    if wave_number <= 8:
        # Pulse waves with varying duty cycle
        duty = 0.1 + (wave_number / 8) * 0.4
        return generate_pulse_table(size, duty)
    elif wave_number <= 16:
        # Resonant waves
        n = (wave_number - 8) * 2 + 1
        table = np.sin(phase) + 0.5 * np.sin(n * phase)
        return table / np.max(np.abs(table))
    elif wave_number <= 24:
        # Mixed waves
        mix = (wave_number - 16) / 8
        saw = generate_saw_table(size)
        square = generate_square_table(size)
        return (1 - mix) * saw + mix * square
    else:
        # Complex harmonic content
        table = np.zeros(size)
        for n in range(1, 16):
            weight = 1.0 / (n ** (wave_number / 32))
            table += weight * np.sin(n * phase + wave_number * 0.1)
        return table / np.max(np.abs(table))


def generate_serum_table(size=2048, formula='sin(x) + sin(2*x)/2'):
    """
    Generate wavetable from mathematical formula (Serum-style).

    Args:
        size: Table size
        formula: Python expression using x (phase from 0 to 2π)
    """
    x = np.linspace(0, 2 * np.pi, size, endpoint=False)

    try:
        # Safe eval with limited scope
        safe_dict = {
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'abs': np.abs, 'exp': np.exp, 'log': np.log,
            'sqrt': np.sqrt, 'pi': np.pi, 'x': x,
            'np': np  # Allow numpy functions
        }
        table = eval(formula, {"__builtins__": {}}, safe_dict)

        # Normalize
        table = np.array(table)
        max_val = np.max(np.abs(table))
        if max_val > 1e-10:
            table = table / max_val

        return table
    except Exception as e:
        print(f"Error evaluating formula: {e}")
        return generate_sine_table(size)


# Preset wavetables
class PresetWavetables:
    """Collection of preset wavetables."""

    @staticmethod
    def basic_shapes(size=2048):
        """Basic waveform shapes."""
        return WavetableBank([
            generate_sine_table(size),
            generate_triangle_table(size),
            generate_saw_table(size),
            generate_square_table(size)
        ])

    @staticmethod
    def pulse_width(size=2048, steps=8):
        """Pulse width sweep."""
        return WavetableBank([
            generate_pulse_table(size, duty=i/(steps-1))
            for i in range(steps)
        ])

    @staticmethod
    def harmonic_series(size=2048, steps=8):
        """Harmonic series evolution."""
        tables = []
        for i in range(steps):
            harmonics = [(1.0 / (n ** (i / steps))) for n in range(1, 9)]
            tables.append(generate_harmonic_table(size, harmonics))
        return WavetableBank(tables)

    @staticmethod
    def vowels(size=2048):
        """Vowel formants."""
        return WavetableBank([
            generate_vowel_table(size, v)
            for v in ['a', 'e', 'i', 'o', 'u']
        ])

    @staticmethod
    def ppg_waves(size=2048, count=16):
        """PPG-style wavetables."""
        return WavetableBank([
            generate_ppg_table(size, i * 4)
            for i in range(count)
        ])

    @staticmethod
    def analog_stack(size=2048):
        """Analog-style waveforms with detuned oscillators."""
        return WavetableBank([
            generate_saw_table(size, harmonics=16),
            generate_saw_table(size, harmonics=32),
            generate_square_table(size, harmonics=16),
            generate_pulse_table(size, duty=0.25),
            generate_pulse_table(size, duty=0.125),
        ])


# Convenience function to create wavetable from function
def create_wavetable_from_function(func, size=2048):
    """
    Create wavetable from a function.

    Args:
        func: Function that takes phase (0 to 2π) and returns amplitude
        size: Table size

    Returns:
        Wavetable array
    """
    phase = np.linspace(0, 2 * np.pi, size, endpoint=False)
    table = func(phase)
    return table / np.max(np.abs(table))
