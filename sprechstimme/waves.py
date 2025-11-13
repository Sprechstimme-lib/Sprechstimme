import numpy as np

def sine(t, freq=440, amp=1.0):
    return amp * np.sin(2 * np.pi * freq * t)

def square(t, freq=440, amp=1.0):
    return amp * np.sign(np.sin(2 * np.pi * freq * t))

def sawtooth(t, freq=440, amp=1.0):
    return amp * (2 * (t * freq - np.floor(0.5 + t * freq)))

def triangle(t, freq=440, amp=1.0):
    return amp * (2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - 1)

def noise(t, freq=0, amp=1.0):
    return amp * np.random.uniform(-1.0, 1.0, size=t.shape)

def pulse(t, freq=440, amp=1.0, duty=0.5):
    return amp * np.where((t * freq) % 1 < duty, 1.0, -1.0)

def ring_mod(t, freq=440, freq2=660, amp=1.0):
    return amp * np.sin(2 * np.pi * freq * t) * np.sin(2 * np.pi * freq2 * t)

# Advanced waveform generators
def fm(t, carrier_freq=440, modulator_freq=220, mod_index=2.0, amp=1.0):
    """Frequency modulation synthesis."""
    return amp * np.sin(2 * np.pi * carrier_freq * t + mod_index * np.sin(2 * np.pi * modulator_freq * t))

def pwm(t, freq=440, amp=1.0, lfo_freq=5, depth=0.4):
    """Pulse width modulation - pulse wave with varying duty cycle."""
    duty = 0.5 + depth * np.sin(2 * np.pi * lfo_freq * t)
    duty = np.clip(duty, 0.05, 0.95)
    return amp * np.where((t * freq) % 1 < duty, 1.0, -1.0)

def supersaw(t, freq=440, amp=1.0, detune=0.1, voices=7):
    """Supersaw - multiple detuned sawtooth waves for thick sound."""
    output = np.zeros_like(t)
    for i in range(voices):
        detune_factor = 1.0 + detune * (i - voices // 2) / voices
        output += sawtooth(t, freq * detune_factor, amp / voices)
    return output

def unison(t, freq=440, amp=1.0, detune=0.05, voices=5, waveform='saw'):
    """Unison - multiple detuned voices for a thick, wide sound."""
    output = np.zeros_like(t)
    wave_func = {'saw': sawtooth, 'square': square, 'sine': sine, 'triangle': triangle}[waveform]
    for i in range(voices):
        detune_factor = 1.0 + detune * (i - voices // 2) / voices
        output += wave_func(t, freq * detune_factor, amp / voices)
    return output

def additive(t, freq=440, amp=1.0, harmonics=None):
    """Additive synthesis - sum of harmonics with individual amplitudes."""
    if harmonics is None:
        harmonics = [1.0, 0.5, 0.25, 0.125]  # Default organ-like
    output = np.zeros_like(t)
    for i, h_amp in enumerate(harmonics):
        output += h_amp * sine(t, freq * (i + 1), amp / len(harmonics))
    return output

def formant(t, freq=440, amp=1.0, formant_freqs=[800, 1150, 2900]):
    """Formant synthesis - vowel-like sounds."""
    output = sawtooth(t, freq, amp)
    # Simple formant filtering simulation using additive synthesis
    formant_wave = np.zeros_like(t)
    for f_freq in formant_freqs:
        formant_wave += sine(t, f_freq, 0.3)
    return output * (1.0 + 0.3 * formant_wave)

def karplus_strong(t, freq=440, amp=1.0, decay=0.996, sample_rate=44100):
    """Karplus-Strong algorithm for plucked string synthesis."""
    period_samples = int(sample_rate / freq)
    output = np.zeros(len(t))
    # Initialize with noise burst
    output[:period_samples] = np.random.uniform(-1, 1, period_samples)
    # Feedback loop
    for i in range(period_samples, len(t)):
        output[i] = decay * 0.5 * (output[i - period_samples] + output[i - period_samples - 1])
    return amp * output

def hard_sync(t, freq=440, sync_freq=220, amp=1.0):
    """Hard sync oscillator - master resets slave oscillator."""
    slave = sawtooth(t, freq, 1.0)
    master_phase = (t * sync_freq) % 1.0
    # Reset slave when master crosses zero
    resets = np.diff(np.floor(master_phase), prepend=0) > 0
    reset_indices = np.where(resets)[0]
    # Apply resets
    for i in range(len(reset_indices) - 1):
        start = reset_indices[i]
        end = reset_indices[i + 1]
        segment_t = t[start:end] - t[start]
        slave[start:end] = sawtooth(segment_t, freq, 1.0)
    return amp * slave

def wavetable(t, freq=440, amp=1.0, table=None):
    """Wavetable synthesis - reads through a custom waveform table."""
    if table is None:
        table = sine(np.linspace(0, 1/440, 1024), 440, 1.0)  # Default to sine
    phase = (t * freq) % 1.0
    indices = (phase * len(table)).astype(int) % len(table)
    return amp * table[indices]

def morphing_wave(t, freq=440, amp=1.0, morph=0.5):
    """Morphs between different waveforms (0=sine, 0.33=triangle, 0.66=saw, 1=square)."""
    if morph < 0.33:
        # Morph between sine and triangle
        alpha = morph / 0.33
        return amp * ((1 - alpha) * sine(t, freq, 1.0) + alpha * triangle(t, freq, 1.0))
    elif morph < 0.66:
        # Morph between triangle and sawtooth
        alpha = (morph - 0.33) / 0.33
        return amp * ((1 - alpha) * triangle(t, freq, 1.0) + alpha * sawtooth(t, freq, 1.0))
    else:
        # Morph between sawtooth and square
        alpha = (morph - 0.66) / 0.34
        return amp * ((1 - alpha) * sawtooth(t, freq, 1.0) + alpha * square(t, freq, 1.0))

def pink_noise(t, freq=0, amp=1.0):
    """Pink noise (1/f noise) - more natural than white noise."""
    # Simple pink noise approximation using multiple octaves
    output = np.zeros_like(t)
    for octave in range(8):
        white = np.random.uniform(-1, 1, len(t))
        output += white / (octave + 1)
    return amp * output / 4.0

def brown_noise(t, freq=0, amp=1.0):
    """Brown noise (red noise) - even lower frequency emphasis."""
    white = np.random.uniform(-1, 1, len(t))
    output = np.cumsum(white)
    output = output / np.max(np.abs(output))  # Normalize
    return amp * output
    
