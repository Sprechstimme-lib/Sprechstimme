"""Modulation sources and envelopes for synthesis."""
import numpy as np


class LFO:
    """Low Frequency Oscillator for modulation."""

    def __init__(self, rate=5.0, waveform='sine', depth=1.0, phase=0.0):
        """
        Args:
            rate: LFO rate in Hz
            waveform: 'sine', 'triangle', 'square', 'saw', 'random'
            depth: Modulation depth (0.0 to 1.0)
            phase: Initial phase (0.0 to 1.0)
        """
        self.rate = rate
        self.waveform = waveform
        self.depth = depth
        self.phase = phase

    def generate(self, t):
        """Generate LFO signal for time array t."""
        phase = (t * self.rate + self.phase) % 1.0

        if self.waveform == 'sine':
            lfo = np.sin(2 * np.pi * phase)
        elif self.waveform == 'triangle':
            lfo = 2 * np.abs(2 * (phase - 0.5)) - 1
        elif self.waveform == 'square':
            lfo = np.where(phase < 0.5, 1.0, -1.0)
        elif self.waveform == 'saw':
            lfo = 2 * (phase - 0.5)
        elif self.waveform == 'random':
            # Sample and hold random values
            samples = int(len(t) * self.rate / (t[-1] - t[0]) if len(t) > 1 else 1)
            random_vals = np.random.uniform(-1, 1, max(samples, 1))
            indices = (phase * len(random_vals)).astype(int) % len(random_vals)
            lfo = random_vals[indices]
        else:
            lfo = np.sin(2 * np.pi * phase)

        return lfo * self.depth

    def __call__(self, t):
        """Allows LFO to be called as a function."""
        return self.generate(t)


class ADSR:
    """ADSR Envelope Generator - more flexible than the built-in one."""

    def __init__(self, attack=0.01, decay=0.1, sustain=0.7, release=0.3, curve='linear'):
        """
        Args:
            attack: Attack time in seconds
            decay: Decay time in seconds
            sustain: Sustain level (0.0 to 1.0)
            release: Release time in seconds
            curve: Envelope curve type - 'linear', 'exponential', 'logarithmic'
        """
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        self.release = release
        self.curve = curve

    def generate(self, duration, sample_rate=44100):
        """Generate ADSR envelope."""
        total_samples = int(duration * sample_rate)
        attack_samples = int(self.attack * sample_rate)
        decay_samples = int(self.decay * sample_rate)
        release_samples = int(self.release * sample_rate)
        sustain_samples = max(0, total_samples - attack_samples - decay_samples - release_samples)

        envelope = np.zeros(total_samples)

        # Attack
        if attack_samples > 0:
            if self.curve == 'exponential':
                envelope[:attack_samples] = 1 - np.exp(-5 * np.linspace(0, 1, attack_samples))
            elif self.curve == 'logarithmic':
                envelope[:attack_samples] = np.log(1 + np.linspace(0, np.e - 1, attack_samples))
            else:  # linear
                envelope[:attack_samples] = np.linspace(0, 1, attack_samples)

        # Decay
        start_idx = attack_samples
        end_idx = start_idx + decay_samples
        if decay_samples > 0:
            if self.curve == 'exponential':
                decay_curve = np.exp(-3 * np.linspace(0, 1, decay_samples))
                envelope[start_idx:end_idx] = self.sustain + (1 - self.sustain) * decay_curve
            else:  # linear
                envelope[start_idx:end_idx] = np.linspace(1, self.sustain, decay_samples)

        # Sustain
        start_idx = end_idx
        end_idx = start_idx + sustain_samples
        envelope[start_idx:end_idx] = self.sustain

        # Release
        start_idx = end_idx
        if release_samples > 0 and start_idx < total_samples:
            actual_release = min(release_samples, total_samples - start_idx)
            if self.curve == 'exponential':
                envelope[start_idx:start_idx + actual_release] = self.sustain * np.exp(-5 * np.linspace(0, 1, actual_release))
            else:  # linear
                envelope[start_idx:start_idx + actual_release] = np.linspace(self.sustain, 0, actual_release)

        return envelope


class AR:
    """Attack-Release Envelope - simpler than ADSR."""

    def __init__(self, attack=0.01, release=0.3):
        self.attack = attack
        self.release = release

    def generate(self, duration, sample_rate=44100):
        """Generate AR envelope."""
        total_samples = int(duration * sample_rate)
        attack_samples = int(self.attack * sample_rate)
        release_samples = int(self.release * sample_rate)
        sustain_samples = max(0, total_samples - attack_samples - release_samples)

        envelope = np.zeros(total_samples)

        # Attack
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)

        # Sustain at 1.0
        envelope[attack_samples:attack_samples + sustain_samples] = 1.0

        # Release
        start_idx = attack_samples + sustain_samples
        if release_samples > 0 and start_idx < total_samples:
            actual_release = min(release_samples, total_samples - start_idx)
            envelope[start_idx:start_idx + actual_release] = np.linspace(1, 0, actual_release)

        return envelope


class EnvelopeFollower:
    """Envelope follower - extracts amplitude envelope from a signal."""

    def __init__(self, attack=0.01, release=0.1):
        """
        Args:
            attack: Attack time constant in seconds
            release: Release time constant in seconds
        """
        self.attack = attack
        self.release = release

    def process(self, signal, sample_rate=44100):
        """Extract envelope from signal."""
        # Full-wave rectification
        rectified = np.abs(signal)

        # Smooth with asymmetric filter
        envelope = np.zeros_like(rectified)
        envelope[0] = rectified[0]

        attack_coeff = np.exp(-1.0 / (self.attack * sample_rate))
        release_coeff = np.exp(-1.0 / (self.release * sample_rate))

        for i in range(1, len(rectified)):
            if rectified[i] > envelope[i-1]:
                # Attack
                envelope[i] = attack_coeff * envelope[i-1] + (1 - attack_coeff) * rectified[i]
            else:
                # Release
                envelope[i] = release_coeff * envelope[i-1] + (1 - release_coeff) * rectified[i]

        return envelope


class StepSequencer:
    """Step sequencer for creating rhythmic modulation patterns."""

    def __init__(self, steps, rate=4.0):
        """
        Args:
            steps: List of values for each step
            rate: Steps per second
        """
        self.steps = np.array(steps)
        self.rate = rate

    def generate(self, t):
        """Generate step sequence for time array t."""
        if len(t) == 0:
            return np.array([])

        step_indices = (t * self.rate).astype(int) % len(self.steps)
        return self.steps[step_indices]


def apply_modulation(signal, modulator, depth=1.0, mode='multiply'):
    """
    Apply modulation to a signal.

    Args:
        signal: Input signal array
        modulator: Modulation signal array (same length as signal)
        depth: Modulation depth (0.0 to 1.0)
        mode: 'multiply' (AM), 'add', or 'ring' (ring modulation)
    """
    if len(signal) != len(modulator):
        raise ValueError("Signal and modulator must have the same length")

    if mode == 'multiply':
        # Amplitude modulation
        return signal * (1.0 + depth * modulator)
    elif mode == 'add':
        # Additive modulation
        return signal + depth * modulator
    elif mode == 'ring':
        # Ring modulation
        return signal * modulator * depth
    else:
        return signal


# Convenience functions
def tremolo(signal, t, rate=5.0, depth=0.5):
    """Apply tremolo (amplitude modulation) to signal."""
    lfo = LFO(rate=rate, waveform='sine', depth=depth)
    mod = lfo.generate(t)
    return signal * (1.0 - depth + depth * (mod + 1.0) / 2.0)


def vibrato_signal(t, freq, rate=5.0, depth=0.02):
    """
    Generate a frequency-modulated sine wave (vibrato).

    Args:
        t: Time array
        freq: Base frequency
        rate: Vibrato rate in Hz
        depth: Vibrato depth as fraction of frequency

    Returns:
        Vibrato signal
    """
    lfo = LFO(rate=rate, waveform='sine', depth=depth)
    freq_mod = freq * (1.0 + lfo.generate(t))
    phase = np.cumsum(freq_mod) / 44100  # Assuming 44100 sample rate
    return np.sin(2 * np.pi * phase)


def auto_pan(signal, t, rate=0.5):
    """
    Create auto-panning effect (returns left and right channels).

    Args:
        signal: Mono input signal
        t: Time array
        rate: Pan rate in Hz

    Returns:
        (left_channel, right_channel)
    """
    lfo = LFO(rate=rate, waveform='sine', depth=1.0)
    pan = (lfo.generate(t) + 1.0) / 2.0  # 0 to 1

    left = signal * np.sqrt(1.0 - pan)
    right = signal * np.sqrt(pan)

    return left, right
