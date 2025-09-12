import numpy as np
from scipy.signal import butter, lfilter

def low_pass_filter(signal, sample_rate, cutoff=1000.0, order=5):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)

def high_pass_filter(signal, sample_rate, cutoff=500.0, order=5):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return lfilter(b, a, signal)

def band_pass_filter(signal, sample_rate, low=300.0, high=3000.0, order=5):
    nyquist = 0.5 * sample_rate
    low = low / nyquist
    high = high / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, signal)

def echo_filter(signal, sample_rate, delay=0.5, decay=0.5):
    delay_samples = int(delay * sample_rate)
    echo_signal = np.zeros_like(signal)
    echo_signal[delay_samples:] = signal[:-delay_samples] * decay
    return signal + echo_signal