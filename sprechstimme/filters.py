import numpy as np
from scipy.signal import butter, lfilter
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

# convenience partial creators so user can pass strings + params by using functools.partial:
from functools import partial as _partial
lp = lambda cutoff=1000.0, order=4: _partial(low_pass, cutoff=cutoff, order=order)
hp = lambda cutoff=200.0, order=4: _partial(high_pass, cutoff=cutoff, order=order)
bp = lambda low=300.0, high=3000.0, order=4: _partial(band_pass, low=low, high=high, order=order)
delay = lambda delay=0.25, decay=0.4: _partial(echo, delay=delay, decay=decay)
dist = lambda gain=1.0, threshold=0.8: _partial(simple_distortion, gain=gain, threshold=threshold)