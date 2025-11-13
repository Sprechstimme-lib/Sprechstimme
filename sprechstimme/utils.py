"""Audio utilities for mixing, processing, and analysis."""
import numpy as np


def normalize(signal, target_level=1.0):
    """
    Normalize signal to target level.

    Args:
        signal: Input signal array
        target_level: Target peak level (default 1.0)

    Returns:
        Normalized signal
    """
    max_abs = np.max(np.abs(signal))
    if max_abs > 1e-10:
        return signal * (target_level / max_abs)
    return signal


def mix(signals, levels=None):
    """
    Mix multiple signals together.

    Args:
        signals: List of signal arrays (must all be same length)
        levels: List of mixing levels (default: equal mix)

    Returns:
        Mixed signal
    """
    if not signals:
        return np.array([])

    length = len(signals[0])
    if not all(len(s) == length for s in signals):
        raise ValueError("All signals must have the same length")

    if levels is None:
        levels = [1.0 / len(signals)] * len(signals)

    if len(levels) != len(signals):
        raise ValueError("Number of levels must match number of signals")

    output = np.zeros(length)
    for signal, level in zip(signals, levels):
        output += signal * level

    return output


def pan(signal, pan_position=0.5):
    """
    Pan a mono signal to stereo.

    Args:
        signal: Mono input signal
        pan_position: Pan position (0.0 = full left, 0.5 = center, 1.0 = full right)

    Returns:
        (left_channel, right_channel)
    """
    pan_position = np.clip(pan_position, 0.0, 1.0)

    # Equal power panning
    left = signal * np.sqrt(1.0 - pan_position)
    right = signal * np.sqrt(pan_position)

    return left, right


def stereo_width(left, right, width=1.0):
    """
    Adjust stereo width.

    Args:
        left: Left channel
        right: Right channel
        width: Stereo width (0.0 = mono, 1.0 = normal, 2.0 = extra wide)

    Returns:
        (left_out, right_out)
    """
    mid = (left + right) / 2.0
    side = (left - right) / 2.0

    side = side * width

    left_out = mid + side
    right_out = mid - side

    return left_out, right_out


def fade_in(signal, duration=1.0, sample_rate=44100, curve='linear'):
    """
    Apply fade-in to signal.

    Args:
        signal: Input signal
        duration: Fade duration in seconds
        sample_rate: Sample rate in Hz
        curve: Fade curve ('linear', 'exponential', 'logarithmic')

    Returns:
        Faded signal
    """
    fade_samples = int(duration * sample_rate)
    fade_samples = min(fade_samples, len(signal))

    if curve == 'exponential':
        envelope = 1 - np.exp(-5 * np.linspace(0, 1, fade_samples))
    elif curve == 'logarithmic':
        envelope = np.log(1 + np.linspace(0, np.e - 1, fade_samples))
    else:  # linear
        envelope = np.linspace(0, 1, fade_samples)

    output = np.copy(signal)
    output[:fade_samples] *= envelope

    return output


def fade_out(signal, duration=1.0, sample_rate=44100, curve='linear'):
    """
    Apply fade-out to signal.

    Args:
        signal: Input signal
        duration: Fade duration in seconds
        sample_rate: Sample rate in Hz
        curve: Fade curve ('linear', 'exponential', 'logarithmic')

    Returns:
        Faded signal
    """
    fade_samples = int(duration * sample_rate)
    fade_samples = min(fade_samples, len(signal))

    if curve == 'exponential':
        envelope = np.exp(-5 * np.linspace(0, 1, fade_samples))
    elif curve == 'logarithmic':
        envelope = 1 - np.log(1 + np.linspace(0, np.e - 1, fade_samples))
    else:  # linear
        envelope = np.linspace(1, 0, fade_samples)

    output = np.copy(signal)
    output[-fade_samples:] *= envelope

    return output


def crossfade(signal1, signal2, duration=1.0, sample_rate=44100, curve='linear'):
    """
    Crossfade between two signals.

    Args:
        signal1: First signal (fades out)
        signal2: Second signal (fades in)
        duration: Crossfade duration in seconds
        sample_rate: Sample rate in Hz
        curve: Fade curve

    Returns:
        Crossfaded signal
    """
    fade_samples = int(duration * sample_rate)

    # Ensure signals are long enough
    min_length = min(len(signal1), len(signal2))
    fade_samples = min(fade_samples, min_length)

    if curve == 'exponential':
        fade_out_curve = np.exp(-5 * np.linspace(0, 1, fade_samples))
        fade_in_curve = 1 - np.exp(-5 * np.linspace(0, 1, fade_samples))
    elif curve == 'logarithmic':
        fade_out_curve = 1 - np.log(1 + np.linspace(0, np.e - 1, fade_samples))
        fade_in_curve = np.log(1 + np.linspace(0, np.e - 1, fade_samples))
    else:  # linear
        fade_out_curve = np.linspace(1, 0, fade_samples)
        fade_in_curve = np.linspace(0, 1, fade_samples)

    # Create output
    max_length = max(len(signal1), len(signal2))
    output = np.zeros(max_length)

    # Add faded portions
    overlap_start = len(signal1) - fade_samples

    output[:overlap_start] = signal1[:overlap_start]
    output[overlap_start:overlap_start + fade_samples] = (
        signal1[overlap_start:overlap_start + fade_samples] * fade_out_curve +
        signal2[:fade_samples] * fade_in_curve
    )
    output[overlap_start + fade_samples:] = signal2[fade_samples:]

    return output


def mono_to_stereo(signal, width=0.0):
    """
    Convert mono signal to stereo with optional width.

    Args:
        signal: Mono input signal
        width: Stereo width (0.0 = identical channels, 1.0 = maximum decorrelation)

    Returns:
        (left, right)
    """
    if width == 0.0:
        return signal, signal

    # Simple decorrelation using a short delay
    delay_samples = int(width * 10)  # Up to 10 sample delay

    left = signal
    right = np.roll(signal, delay_samples)

    return left, right


def stereo_to_mono(left, right, method='average'):
    """
    Convert stereo to mono.

    Args:
        left: Left channel
        right: Right channel
        method: 'average' or 'left' or 'right'

    Returns:
        Mono signal
    """
    if method == 'left':
        return left
    elif method == 'right':
        return right
    else:  # average
        return (left + right) / 2.0


def resample_simple(signal, original_rate, target_rate):
    """
    Simple linear interpolation resampling.

    Args:
        signal: Input signal
        original_rate: Original sample rate
        target_rate: Target sample rate

    Returns:
        Resampled signal
    """
    ratio = target_rate / original_rate
    new_length = int(len(signal) * ratio)

    old_indices = np.linspace(0, len(signal) - 1, new_length)
    new_signal = np.interp(old_indices, np.arange(len(signal)), signal)

    return new_signal


def reverse(signal):
    """
    Reverse signal.

    Args:
        signal: Input signal

    Returns:
        Reversed signal
    """
    return signal[::-1]


def time_stretch(signal, factor, sample_rate=44100):
    """
    Simple time stretching (changes duration without changing pitch much).

    Args:
        signal: Input signal
        factor: Stretch factor (0.5 = half speed, 2.0 = double speed)
        sample_rate: Sample rate

    Returns:
        Time-stretched signal
    """
    # Simple implementation using resampling
    new_length = int(len(signal) * factor)
    old_indices = np.linspace(0, len(signal) - 1, new_length)
    stretched = np.interp(old_indices, np.arange(len(signal)), signal)

    return stretched


def clip(signal, threshold=1.0):
    """
    Hard clip signal at threshold.

    Args:
        signal: Input signal
        threshold: Clipping threshold

    Returns:
        Clipped signal
    """
    return np.clip(signal, -threshold, threshold)


def soft_clip(signal, threshold=0.8):
    """
    Soft clip signal using tanh.

    Args:
        signal: Input signal
        threshold: Soft clipping threshold

    Returns:
        Soft-clipped signal
    """
    return threshold * np.tanh(signal / threshold)


def apply_gain(signal, gain_db):
    """
    Apply gain in dB.

    Args:
        signal: Input signal
        gain_db: Gain in decibels

    Returns:
        Signal with gain applied
    """
    gain_linear = 10 ** (gain_db / 20)
    return signal * gain_linear


def db_to_linear(db):
    """Convert decibels to linear amplitude."""
    return 10 ** (db / 20)


def linear_to_db(linear):
    """Convert linear amplitude to decibels."""
    return 20 * np.log10(np.abs(linear) + 1e-10)


def rms(signal):
    """Calculate RMS (Root Mean Square) level of signal."""
    return np.sqrt(np.mean(signal ** 2))


def peak(signal):
    """Calculate peak level of signal."""
    return np.max(np.abs(signal))


def concatenate(signals, gap=0.0, sample_rate=44100):
    """
    Concatenate multiple signals with optional gap.

    Args:
        signals: List of signal arrays
        gap: Gap duration in seconds between signals
        sample_rate: Sample rate in Hz

    Returns:
        Concatenated signal
    """
    if not signals:
        return np.array([])

    gap_samples = int(gap * sample_rate)
    gap_signal = np.zeros(gap_samples)

    result = signals[0]
    for signal in signals[1:]:
        if gap_samples > 0:
            result = np.concatenate([result, gap_signal, signal])
        else:
            result = np.concatenate([result, signal])

    return result


def repeat(signal, times):
    """
    Repeat signal multiple times.

    Args:
        signal: Input signal
        times: Number of repetitions

    Returns:
        Repeated signal
    """
    return np.tile(signal, times)


def trim_silence(signal, threshold_db=-40.0, sample_rate=44100, frame_length=2048):
    """
    Trim silence from beginning and end of signal.

    Args:
        signal: Input signal
        threshold_db: Silence threshold in dB
        sample_rate: Sample rate in Hz
        frame_length: Frame length for analysis

    Returns:
        Trimmed signal
    """
    threshold_linear = db_to_linear(threshold_db)

    # Find first non-silent frame
    start = 0
    for i in range(0, len(signal), frame_length):
        frame = signal[i:i + frame_length]
        if rms(frame) > threshold_linear:
            start = i
            break

    # Find last non-silent frame
    end = len(signal)
    for i in range(len(signal) - frame_length, 0, -frame_length):
        frame = signal[i:i + frame_length]
        if rms(frame) > threshold_linear:
            end = i + frame_length
            break

    return signal[start:end]


def split_stereo(stereo_signal):
    """
    Split interleaved stereo signal into left and right channels.

    Args:
        stereo_signal: Interleaved stereo array [L, R, L, R, ...]

    Returns:
        (left, right)
    """
    left = stereo_signal[0::2]
    right = stereo_signal[1::2]
    return left, right


def interleave_stereo(left, right):
    """
    Interleave left and right channels into stereo signal.

    Args:
        left: Left channel
        right: Right channel

    Returns:
        Interleaved stereo array [L, R, L, R, ...]
    """
    stereo = np.empty(len(left) + len(right), dtype=left.dtype)
    stereo[0::2] = left
    stereo[1::2] = right
    return stereo
