import numpy as np
from scipy.signal import butter, lfilter, iirnotch, iirpeak, firwin, filtfilt
from functools import partial

def _butter_filter(signal, sample_rate, cutoff, btype="low", order=4):
    nyq = 0.5 * sample_rate
    if isinstance(cutoff, (list, tuple)) and btype == "band":
        normal = [c / nyq for c in cutoff]
    else:
        normal = cutoff / nyq
    b, a = butter(order, normal, btype=btype, analog=False)
    return lfilter(b, a, signal)

def low_pass(signal, sample_rate, cutoff=1000.0, order=4):
    return _butter_filter(signal, sample_rate, cutoff, btype="low", order=order)

def high_pass(signal, sample_rate, cutoff=200.0, order=4):
    return _butter_filter(signal, sample_rate, cutoff, btype="high", order=order)

def band_pass(signal, sample_rate, low=300.0, high=3000.0, order=4):
    return _butter_filter(signal, sample_rate, (low, high), btype="band", order=order)

def echo(signal, sample_rate, delay=0.25, decay=0.4):
    delay_samples = int(delay * sample_rate)
    out = np.copy(signal)
    if delay_samples <= 0 or delay_samples >= len(signal):
        return out
    echo_sig = np.zeros_like(signal)
    echo_sig[delay_samples:] = signal[:-delay_samples] * decay
    out = out + echo_sig
    # normalize
    max_abs = np.max(np.abs(out)) if out.size else 0.0
    if max_abs > 1e-9:
        out = out / max(1.0, max_abs)
    return out

def simple_distortion(signal, sample_rate, gain=1.0, threshold=0.8):
    out = signal * gain
    out = np.clip(out, -threshold, threshold)
    return out / max(1.0, threshold)

# Professional-grade filters
def notch(signal, sample_rate, freq=1000.0, q=30.0):
    """Notch filter - removes a specific frequency."""
    nyq = 0.5 * sample_rate
    w0 = freq / nyq
    b, a = iirnotch(w0, q)
    return lfilter(b, a, signal)

def peaking(signal, sample_rate, freq=1000.0, q=1.0, gain_db=6.0):
    """Peaking EQ filter - boost or cut at a specific frequency."""
    # Simple peaking filter implementation
    nyq = 0.5 * sample_rate
    w0 = freq / nyq
    A = 10 ** (gain_db / 40.0)
    alpha = np.sin(2 * np.pi * w0) / (2 * q)

    b0 = 1 + alpha * A
    b1 = -2 * np.cos(2 * np.pi * w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(2 * np.pi * w0)
    a2 = 1 - alpha / A

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1, a1 / a0, a2 / a0])

    return lfilter(b, a, signal)

def low_shelf(signal, sample_rate, freq=200.0, gain_db=6.0):
    """Low shelf filter - boost or cut low frequencies."""
    nyq = 0.5 * sample_rate
    w0 = freq / nyq
    A = 10 ** (gain_db / 40.0)
    alpha = np.sin(2 * np.pi * w0) / 2 * np.sqrt((A + 1/A) * (1/0.7 - 1) + 2)

    cos_w0 = np.cos(2 * np.pi * w0)

    b0 = A * ((A+1) - (A-1)*cos_w0 + 2*np.sqrt(A)*alpha)
    b1 = 2*A * ((A-1) - (A+1)*cos_w0)
    b2 = A * ((A+1) - (A-1)*cos_w0 - 2*np.sqrt(A)*alpha)
    a0 = (A+1) + (A-1)*cos_w0 + 2*np.sqrt(A)*alpha
    a1 = -2 * ((A-1) + (A+1)*cos_w0)
    a2 = (A+1) + (A-1)*cos_w0 - 2*np.sqrt(A)*alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1, a1 / a0, a2 / a0])

    return lfilter(b, a, signal)

def high_shelf(signal, sample_rate, freq=4000.0, gain_db=6.0):
    """High shelf filter - boost or cut high frequencies."""
    nyq = 0.5 * sample_rate
    w0 = freq / nyq
    A = 10 ** (gain_db / 40.0)
    alpha = np.sin(2 * np.pi * w0) / 2 * np.sqrt((A + 1/A) * (1/0.7 - 1) + 2)

    cos_w0 = np.cos(2 * np.pi * w0)

    b0 = A * ((A+1) + (A-1)*cos_w0 + 2*np.sqrt(A)*alpha)
    b1 = -2*A * ((A-1) + (A+1)*cos_w0)
    b2 = A * ((A+1) + (A-1)*cos_w0 - 2*np.sqrt(A)*alpha)
    a0 = (A+1) - (A-1)*cos_w0 + 2*np.sqrt(A)*alpha
    a1 = 2 * ((A-1) - (A+1)*cos_w0)
    a2 = (A+1) - (A-1)*cos_w0 - 2*np.sqrt(A)*alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1, a1 / a0, a2 / a0])

    return lfilter(b, a, signal)

def allpass(signal, sample_rate, freq=1000.0, q=0.707):
    """Allpass filter - changes phase without affecting amplitude."""
    nyq = 0.5 * sample_rate
    w0 = freq / nyq
    alpha = np.sin(2 * np.pi * w0) / (2 * q)

    b0 = 1 - alpha
    b1 = -2 * np.cos(2 * np.pi * w0)
    b2 = 1 + alpha
    a0 = 1 + alpha
    a1 = -2 * np.cos(2 * np.pi * w0)
    a2 = 1 - alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1, a1 / a0, a2 / a0])

    return lfilter(b, a, signal)

def comb_filter(signal, sample_rate, delay=0.05, feedback=0.5):
    """Comb filter - creates resonant peaks at harmonics of delay time."""
    delay_samples = int(delay * sample_rate)
    output = np.copy(signal)

    for i in range(delay_samples, len(signal)):
        output[i] = signal[i] + feedback * output[i - delay_samples]

    # Normalize to prevent clipping
    max_abs = np.max(np.abs(output))
    if max_abs > 1e-9:
        output = output / max(1.0, max_abs)

    return output

def state_variable_filter(signal, sample_rate, cutoff=1000.0, resonance=0.707, mode='low'):
    """State variable filter - provides low, high, and band outputs with resonance."""
    nyq = 0.5 * sample_rate
    f = 2 * np.sin(np.pi * cutoff / sample_rate)
    q = 1.0 / resonance

    low = np.zeros_like(signal)
    high = np.zeros_like(signal)
    band = np.zeros_like(signal)
    notch_out = np.zeros_like(signal)

    for i in range(len(signal)):
        if i == 0:
            low[i] = 0
            band[i] = 0
        else:
            low[i] = low[i-1] + f * band[i-1]
            high[i] = signal[i] - low[i] - q * band[i-1]
            band[i] = f * high[i] + band[i-1]
            notch_out[i] = high[i] + low[i]

    if mode == 'low':
        return low
    elif mode == 'high':
        return high
    elif mode == 'band':
        return band
    elif mode == 'notch':
        return notch_out
    else:
        return low

def band_stop(signal, sample_rate, low=300.0, high=3000.0, order=4):
    """Band stop (notch) filter - removes a range of frequencies."""
    return _butter_filter(signal, sample_rate, (low, high), btype="bandstop", order=order)

def ladder_filter(signal, sample_rate, cutoff=1000.0, resonance=0.5):
    """Moog-style ladder filter - warm, musical resonance."""
    nyq = 0.5 * sample_rate
    fc = cutoff / sample_rate

    # Simplified ladder filter model
    f = fc * 1.16
    fb = resonance * (1.0 - 0.15 * f * f)

    y1 = y2 = y3 = y4 = 0
    output = np.zeros_like(signal)

    for i in range(len(signal)):
        input_val = signal[i] - fb * y4

        y1 = y1 + f * (np.tanh(input_val) - np.tanh(y1))
        y2 = y2 + f * (np.tanh(y1) - np.tanh(y2))
        y3 = y3 + f * (np.tanh(y2) - np.tanh(y3))
        y4 = y4 + f * (np.tanh(y3) - np.tanh(y4))

        output[i] = y4

    return output

# convenience partial creators so user can pass strings + params by using functools.partial:
from functools import partial as _partial
lp = lambda cutoff=1000.0, order=4: _partial(low_pass, cutoff=cutoff, order=order)
hp = lambda cutoff=200.0, order=4: _partial(high_pass, cutoff=cutoff, order=order)
bp = lambda low=300.0, high=3000.0, order=4: _partial(band_pass, low=low, high=high, order=order)
delay = lambda delay=0.25, decay=0.4: _partial(echo, delay=delay, decay=decay)
dist = lambda gain=1.0, threshold=0.8: _partial(simple_distortion, gain=gain, threshold=threshold)

# Additional convenience helpers for new filters
notch_filter = lambda freq=1000.0, q=30.0: _partial(notch, freq=freq, q=q)
peak_eq = lambda freq=1000.0, q=1.0, gain_db=6.0: _partial(peaking, freq=freq, q=q, gain_db=gain_db)
low_shelf_eq = lambda freq=200.0, gain_db=6.0: _partial(low_shelf, freq=freq, gain_db=gain_db)
high_shelf_eq = lambda freq=4000.0, gain_db=6.0: _partial(high_shelf, freq=freq, gain_db=gain_db)
svf = lambda cutoff=1000.0, resonance=0.707, mode='low': _partial(state_variable_filter, cutoff=cutoff, resonance=resonance, mode=mode)
moog = lambda cutoff=1000.0, resonance=0.5: _partial(ladder_filter, cutoff=cutoff, resonance=resonance)