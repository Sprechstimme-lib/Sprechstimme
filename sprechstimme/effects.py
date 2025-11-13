"""Professional audio effects for synthesis."""
import numpy as np
from functools import partial


def chorus(signal, sample_rate, rate=1.5, depth=0.003, mix=0.5, voices=3):
    """
    Chorus effect - creates a thicker sound by mixing delayed copies with slight pitch variation.

    Args:
        signal: Input signal
        sample_rate: Sample rate in Hz
        rate: LFO rate in Hz
        depth: Delay modulation depth in seconds
        mix: Wet/dry mix (0.0 = dry, 1.0 = wet)
        voices: Number of chorus voices
    """
    output = np.copy(signal)
    t = np.arange(len(signal)) / sample_rate

    for i in range(voices):
        # Each voice has slightly different rate and phase
        lfo_rate = rate * (1.0 + i * 0.1)
        phase = i / voices
        lfo = depth * sample_rate * (np.sin(2 * np.pi * lfo_rate * t + phase * 2 * np.pi) + 1) / 2

        # Create modulated delay
        delayed = np.zeros_like(signal)
        for j in range(len(signal)):
            delay_samples = int(lfo[j]) + 1
            if j >= delay_samples:
                delayed[j] = signal[j - delay_samples]

        output += delayed / voices

    return (1 - mix) * signal + mix * output


def flanger(signal, sample_rate, rate=0.5, depth=0.002, feedback=0.5, mix=0.5):
    """
    Flanger effect - swept comb filter creating a jet-like sound.

    Args:
        signal: Input signal
        sample_rate: Sample rate in Hz
        rate: LFO rate in Hz
        depth: Delay modulation depth in seconds
        feedback: Feedback amount (-0.9 to 0.9)
        mix: Wet/dry mix (0.0 = dry, 1.0 = wet)
    """
    t = np.arange(len(signal)) / sample_rate
    lfo = depth * sample_rate * (np.sin(2 * np.pi * rate * t) + 1) / 2

    output = np.zeros_like(signal)
    feedback_buffer = np.zeros_like(signal)

    for i in range(len(signal)):
        delay_samples = int(lfo[i]) + 1
        if i >= delay_samples:
            delayed = feedback_buffer[i - delay_samples]
        else:
            delayed = 0

        feedback_buffer[i] = signal[i] + feedback * delayed
        output[i] = signal[i] + delayed

    # Normalize
    max_abs = np.max(np.abs(output))
    if max_abs > 1e-9:
        output = output / max(1.0, max_abs)

    return (1 - mix) * signal + mix * output


def phaser(signal, sample_rate, rate=0.5, depth=1.0, stages=4, feedback=0.5, mix=0.5):
    """
    Phaser effect - series of allpass filters creating a sweeping sound.

    Args:
        signal: Input signal
        sample_rate: Sample rate in Hz
        rate: LFO rate in Hz
        depth: Modulation depth
        stages: Number of allpass filter stages (2, 4, 6, etc.)
        feedback: Feedback amount
        mix: Wet/dry mix
    """
    t = np.arange(len(signal)) / sample_rate
    lfo = 200 + 1000 * depth * (np.sin(2 * np.pi * rate * t) + 1) / 2

    output = np.copy(signal)
    feedback_signal = np.zeros_like(signal)

    # Simple allpass approximation
    for stage in range(stages):
        filtered = np.zeros_like(output)
        for i in range(1, len(output)):
            # Simple first-order allpass
            a = (lfo[i] - 1) / (lfo[i] + 1)
            filtered[i] = a * output[i] + output[i-1] - a * filtered[i-1]
        output = filtered

    output = output + feedback * np.roll(output, 100)

    # Normalize
    max_abs = np.max(np.abs(output))
    if max_abs > 1e-9:
        output = output / max(1.0, max_abs)

    return (1 - mix) * signal + mix * output


def reverb(signal, sample_rate, room_size=0.5, damping=0.5, wet=0.3):
    """
    Simple reverb effect using Schroeder reverberator.

    Args:
        signal: Input signal
        sample_rate: Sample rate in Hz
        room_size: Room size (0.0 to 1.0)
        damping: High frequency damping (0.0 to 1.0)
        wet: Wet/dry mix (0.0 = dry, 1.0 = wet)
    """
    # Comb filter delays (in samples) - scaled by room size
    comb_delays = [int(d * room_size) for d in [1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116]]
    comb_gains = [0.773, 0.802, 0.753, 0.733]

    # Allpass delays
    allpass_delays = [int(d * room_size) for d in [225, 556, 441, 341]]
    allpass_gains = [0.7, 0.7, 0.7, 0.7]

    output = np.copy(signal)

    # Parallel comb filters
    comb_output = np.zeros_like(signal)
    for delay, gain in zip(comb_delays[:4], comb_gains):
        delay = max(delay, 1)
        comb = np.zeros_like(signal)
        for i in range(delay, len(signal)):
            comb[i] = signal[i] + gain * comb[i - delay] * (1 - damping)
        comb_output += comb / len(comb_gains)

    # Series allpass filters
    output = comb_output
    for delay, gain in zip(allpass_delays, allpass_gains):
        delay = max(delay, 1)
        filtered = np.zeros_like(output)
        for i in range(delay, len(output)):
            filtered[i] = -gain * output[i] + output[i - delay] + gain * filtered[i - delay]
        output = filtered

    # Normalize
    max_abs = np.max(np.abs(output))
    if max_abs > 1e-9:
        output = output / max(1.0, max_abs)

    return (1 - wet) * signal + wet * output


def compressor(signal, sample_rate, threshold=-20.0, ratio=4.0, attack=0.005, release=0.1, makeup_gain=0.0):
    """
    Dynamic range compressor.

    Args:
        signal: Input signal
        sample_rate: Sample rate in Hz
        threshold: Threshold in dB
        ratio: Compression ratio (e.g., 4.0 = 4:1)
        attack: Attack time in seconds
        release: Release time in seconds
        makeup_gain: Makeup gain in dB
    """
    # Convert to dB
    signal_db = 20 * np.log10(np.abs(signal) + 1e-10)

    # Calculate gain reduction
    gain_reduction = np.zeros_like(signal_db)
    for i in range(len(signal_db)):
        if signal_db[i] > threshold:
            gain_reduction[i] = (signal_db[i] - threshold) * (1 - 1/ratio)

    # Apply attack/release envelope
    envelope = np.zeros_like(gain_reduction)
    attack_coeff = np.exp(-1.0 / (attack * sample_rate))
    release_coeff = np.exp(-1.0 / (release * sample_rate))

    for i in range(1, len(gain_reduction)):
        if gain_reduction[i] > envelope[i-1]:
            envelope[i] = attack_coeff * envelope[i-1] + (1 - attack_coeff) * gain_reduction[i]
        else:
            envelope[i] = release_coeff * envelope[i-1] + (1 - release_coeff) * gain_reduction[i]

    # Convert back to linear and apply
    gain_linear = 10 ** (-(envelope - makeup_gain) / 20)
    output = signal * gain_linear

    return output


def limiter(signal, sample_rate, threshold=-1.0, release=0.05):
    """
    Limiter - prevents signal from exceeding threshold.

    Args:
        signal: Input signal
        sample_rate: Sample rate in Hz
        threshold: Threshold in dB
        release: Release time in seconds
    """
    threshold_linear = 10 ** (threshold / 20)
    signal_abs = np.abs(signal)

    gain = np.ones_like(signal)
    release_coeff = np.exp(-1.0 / (release * sample_rate))

    for i in range(len(signal)):
        if signal_abs[i] > threshold_linear:
            gain[i] = threshold_linear / signal_abs[i]

        if i > 0:
            # Smooth gain changes
            if gain[i] < gain[i-1]:
                gain[i] = gain[i]  # Immediate attack
            else:
                gain[i] = release_coeff * gain[i-1] + (1 - release_coeff) * gain[i]

    return signal * gain


def tremolo(signal, sample_rate, rate=5.0, depth=0.5, waveform='sine'):
    """
    Tremolo effect - amplitude modulation.

    Args:
        signal: Input signal
        sample_rate: Sample rate in Hz
        rate: Tremolo rate in Hz
        depth: Modulation depth (0.0 to 1.0)
        waveform: LFO waveform ('sine', 'triangle', 'square')
    """
    t = np.arange(len(signal)) / sample_rate

    if waveform == 'sine':
        lfo = np.sin(2 * np.pi * rate * t)
    elif waveform == 'triangle':
        lfo = 2 * np.abs(2 * ((t * rate) % 1.0 - 0.5)) - 1
    elif waveform == 'square':
        lfo = np.sign(np.sin(2 * np.pi * rate * t))
    else:
        lfo = np.sin(2 * np.pi * rate * t)

    modulation = 1.0 - depth + depth * (lfo + 1.0) / 2.0
    return signal * modulation


def vibrato(signal, sample_rate, rate=5.0, depth=0.005):
    """
    Vibrato effect - pitch modulation.

    Args:
        signal: Input signal
        sample_rate: Sample rate in Hz
        rate: Vibrato rate in Hz
        depth: Pitch modulation depth in semitones
    """
    t = np.arange(len(signal)) / sample_rate
    lfo = depth * np.sin(2 * np.pi * rate * t)

    # Time-varying delay
    delay = (2 ** (lfo / 12) - 1) * sample_rate / 440  # Convert semitones to delay

    output = np.zeros_like(signal)
    for i in range(len(signal)):
        delay_samples = delay[i]
        idx = i - delay_samples

        if 0 <= idx < len(signal) - 1:
            # Linear interpolation
            idx_int = int(idx)
            frac = idx - idx_int
            output[i] = (1 - frac) * signal[idx_int] + frac * signal[idx_int + 1]

    return output


def delay_effect(signal, sample_rate, delay_time=0.375, feedback=0.4, mix=0.5):
    """
    Delay effect with feedback.

    Args:
        signal: Input signal
        sample_rate: Sample rate in Hz
        delay_time: Delay time in seconds
        feedback: Feedback amount (0.0 to 0.95)
        mix: Wet/dry mix
    """
    delay_samples = int(delay_time * sample_rate)
    output = np.copy(signal)
    feedback_buffer = np.zeros_like(signal)

    for i in range(len(signal)):
        if i >= delay_samples:
            delayed = feedback_buffer[i - delay_samples]
        else:
            delayed = 0

        feedback_buffer[i] = signal[i] + feedback * delayed
        output[i] = signal[i] + delayed * mix

    # Normalize
    max_abs = np.max(np.abs(output))
    if max_abs > 1e-9:
        output = output / max(1.0, max_abs)

    return output


def ping_pong_delay(signal, sample_rate, delay_time=0.375, feedback=0.4, mix=0.5):
    """
    Ping-pong stereo delay effect.

    Args:
        signal: Input signal (mono)
        sample_rate: Sample rate in Hz
        delay_time: Delay time in seconds
        feedback: Feedback amount
        mix: Wet/dry mix

    Returns:
        (left_channel, right_channel)
    """
    delay_samples = int(delay_time * sample_rate)

    left = np.copy(signal)
    right = np.copy(signal)

    left_buffer = np.zeros_like(signal)
    right_buffer = np.zeros_like(signal)

    for i in range(len(signal)):
        if i >= delay_samples:
            # Ping-pong: left feeds right, right feeds left
            delayed_left = right_buffer[i - delay_samples]
            delayed_right = left_buffer[i - delay_samples]
        else:
            delayed_left = 0
            delayed_right = 0

        left_buffer[i] = signal[i] + feedback * delayed_left
        right_buffer[i] = signal[i] + feedback * delayed_right

        left[i] = (1 - mix) * signal[i] + mix * left_buffer[i]
        right[i] = (1 - mix) * signal[i] + mix * right_buffer[i]

    # Normalize
    max_abs = max(np.max(np.abs(left)), np.max(np.abs(right)))
    if max_abs > 1e-9:
        left = left / max(1.0, max_abs)
        right = right / max(1.0, max_abs)

    return left, right


def bitcrusher(signal, sample_rate, bit_depth=8, sample_rate_reduction=1.0):
    """
    Bitcrusher - reduces bit depth and sample rate for lo-fi effect.

    Args:
        signal: Input signal
        sample_rate: Sample rate in Hz
        bit_depth: Bit depth (1 to 16)
        sample_rate_reduction: Sample rate reduction factor (1.0 = no reduction)
    """
    # Reduce bit depth
    levels = 2 ** bit_depth
    quantized = np.round(signal * levels) / levels

    # Reduce sample rate
    if sample_rate_reduction > 1.0:
        step = int(sample_rate_reduction)
        output = np.copy(quantized)
        for i in range(0, len(output), step):
            output[i:i+step] = quantized[i]
        return output

    return quantized


def overdrive(signal, sample_rate, drive=5.0, tone=0.5):
    """
    Overdrive distortion effect.

    Args:
        signal: Input signal
        sample_rate: Sample rate in Hz
        drive: Drive amount (1.0 to 10.0)
        tone: Tone control (0.0 = dark, 1.0 = bright)
    """
    # Apply drive
    driven = signal * drive

    # Soft clipping with tanh
    clipped = np.tanh(driven)

    # Simple tone control (low-pass filter)
    if tone < 1.0:
        alpha = tone
        filtered = np.zeros_like(clipped)
        filtered[0] = clipped[0]
        for i in range(1, len(clipped)):
            filtered[i] = alpha * clipped[i] + (1 - alpha) * filtered[i-1]
        clipped = filtered

    # Normalize
    max_abs = np.max(np.abs(clipped))
    if max_abs > 1e-9:
        clipped = clipped / max(1.0, max_abs)

    return clipped


def distortion(signal, sample_rate, amount=5.0, mix=1.0):
    """
    Hard distortion effect.

    Args:
        signal: Input signal
        sample_rate: Sample rate in Hz
        amount: Distortion amount
        mix: Wet/dry mix
    """
    distorted = np.tanh(signal * amount)
    return (1 - mix) * signal + mix * distorted


# Convenience partial creators
chorus_fx = lambda rate=1.5, depth=0.003, mix=0.5: partial(chorus, rate=rate, depth=depth, mix=mix)
flanger_fx = lambda rate=0.5, depth=0.002, feedback=0.5, mix=0.5: partial(flanger, rate=rate, depth=depth, feedback=feedback, mix=mix)
phaser_fx = lambda rate=0.5, stages=4, mix=0.5: partial(phaser, rate=rate, stages=stages, mix=mix)
reverb_fx = lambda room_size=0.5, wet=0.3: partial(reverb, room_size=room_size, wet=wet)
compressor_fx = lambda threshold=-20.0, ratio=4.0: partial(compressor, threshold=threshold, ratio=ratio)
tremolo_fx = lambda rate=5.0, depth=0.5: partial(tremolo, rate=rate, depth=depth)
delay_fx = lambda delay_time=0.375, feedback=0.4, mix=0.5: partial(delay_effect, delay_time=delay_time, feedback=feedback, mix=mix)
overdrive_fx = lambda drive=5.0, tone=0.5: partial(overdrive, drive=drive, tone=tone)
