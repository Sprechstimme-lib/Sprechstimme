"""Audio analysis tools for spectrum analysis and pitch detection."""
import numpy as np


def fft_spectrum(signal, sample_rate=44100):
    """
    Compute FFT spectrum of signal.

    Args:
        signal: Input signal
        sample_rate: Sample rate in Hz

    Returns:
        (frequencies, magnitudes)
    """
    n = len(signal)
    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, 1/sample_rate)
    mags = np.abs(fft)

    return freqs, mags


def power_spectrum(signal, sample_rate=44100):
    """
    Compute power spectrum of signal.

    Args:
        signal: Input signal
        sample_rate: Sample rate in Hz

    Returns:
        (frequencies, power in dB)
    """
    freqs, mags = fft_spectrum(signal, sample_rate)
    power = 20 * np.log10(mags + 1e-10)

    return freqs, power


def spectrogram(signal, sample_rate=44100, window_size=2048, hop_size=512):
    """
    Compute spectrogram of signal.

    Args:
        signal: Input signal
        sample_rate: Sample rate in Hz
        window_size: FFT window size
        hop_size: Hop size between windows

    Returns:
        (times, frequencies, spectrogram_matrix)
    """
    num_windows = (len(signal) - window_size) // hop_size + 1

    # Hann window
    window = np.hanning(window_size)

    spectrogram_data = []
    for i in range(num_windows):
        start = i * hop_size
        end = start + window_size
        if end > len(signal):
            break

        windowed = signal[start:end] * window
        fft = np.fft.rfft(windowed)
        mags = np.abs(fft)
        spectrogram_data.append(mags)

    spectrogram_matrix = np.array(spectrogram_data).T

    times = np.arange(num_windows) * hop_size / sample_rate
    freqs = np.fft.rfftfreq(window_size, 1/sample_rate)

    return times, freqs, spectrogram_matrix


def autocorrelation(signal):
    """
    Compute autocorrelation of signal.

    Args:
        signal: Input signal

    Returns:
        Autocorrelation array
    """
    # FFT-based autocorrelation (faster)
    n = len(signal)
    fft = np.fft.fft(signal, n=2*n)
    acf = np.fft.ifft(fft * np.conj(fft))[:n]
    acf = np.real(acf)
    acf = acf / acf[0]  # Normalize

    return acf


def detect_pitch_autocorrelation(signal, sample_rate=44100, min_freq=80, max_freq=1000):
    """
    Detect pitch using autocorrelation method.

    Args:
        signal: Input signal
        sample_rate: Sample rate in Hz
        min_freq: Minimum frequency to detect
        max_freq: Maximum frequency to detect

    Returns:
        Detected frequency in Hz (or None if no pitch detected)
    """
    acf = autocorrelation(signal)

    # Find peaks in autocorrelation
    min_lag = int(sample_rate / max_freq)
    max_lag = int(sample_rate / min_freq)

    if max_lag >= len(acf):
        max_lag = len(acf) - 1

    # Find maximum in valid range
    valid_acf = acf[min_lag:max_lag]
    if len(valid_acf) == 0:
        return None

    peak_lag = np.argmax(valid_acf) + min_lag

    # Check if peak is significant
    if acf[peak_lag] < 0.3:  # Threshold for pitch confidence
        return None

    frequency = sample_rate / peak_lag
    return frequency


def detect_pitch_fft(signal, sample_rate=44100):
    """
    Detect pitch using FFT method (finds dominant frequency).

    Args:
        signal: Input signal
        sample_rate: Sample rate in Hz

    Returns:
        Detected frequency in Hz
    """
    freqs, mags = fft_spectrum(signal, sample_rate)

    # Find peak frequency
    peak_idx = np.argmax(mags)
    peak_freq = freqs[peak_idx]

    return peak_freq


def detect_pitch_yin(signal, sample_rate=44100, threshold=0.15):
    """
    YIN pitch detection algorithm.

    Args:
        signal: Input signal
        sample_rate: Sample rate in Hz
        threshold: Detection threshold (lower = more sensitive)

    Returns:
        Detected frequency in Hz (or None if no pitch detected)
    """
    # Difference function
    def difference_function(signal, max_tau):
        diff = np.zeros(max_tau)
        for tau in range(1, max_tau):
            for i in range(max_tau):
                diff[tau] += (signal[i] - signal[i + tau]) ** 2
        return diff

    # Cumulative mean normalized difference function
    def cmnd(diff):
        cmnd_vals = np.zeros(len(diff))
        cmnd_vals[0] = 1
        running_sum = 0
        for tau in range(1, len(diff)):
            running_sum += diff[tau]
            cmnd_vals[tau] = diff[tau] / (running_sum / tau)
        return cmnd_vals

    max_tau = len(signal) // 2
    diff = difference_function(signal, max_tau)
    cmnd_vals = cmnd(diff)

    # Find first minimum below threshold
    tau = -1
    for i in range(2, len(cmnd_vals)):
        if cmnd_vals[i] < threshold:
            tau = i
            break

    if tau == -1:
        return None

    # Parabolic interpolation for better accuracy
    if tau > 0 and tau < len(cmnd_vals) - 1:
        s0 = cmnd_vals[tau - 1]
        s1 = cmnd_vals[tau]
        s2 = cmnd_vals[tau + 1]
        adjustment = (s2 - s0) / (2 * (2 * s1 - s2 - s0))
        tau = tau + adjustment

    frequency = sample_rate / tau
    return frequency


def detect_onsets(signal, sample_rate=44100, window_size=2048, hop_size=512, threshold=0.3):
    """
    Detect onset times in signal.

    Args:
        signal: Input signal
        sample_rate: Sample rate in Hz
        window_size: Analysis window size
        hop_size: Hop size between windows
        threshold: Detection threshold

    Returns:
        List of onset times in seconds
    """
    # Compute spectral flux
    num_windows = (len(signal) - window_size) // hop_size + 1
    window = np.hanning(window_size)

    prev_spectrum = None
    flux = []

    for i in range(num_windows):
        start = i * hop_size
        end = start + window_size
        if end > len(signal):
            break

        windowed = signal[start:end] * window
        spectrum = np.abs(np.fft.rfft(windowed))

        if prev_spectrum is not None:
            # Spectral flux = sum of positive differences
            diff = spectrum - prev_spectrum
            diff[diff < 0] = 0
            flux.append(np.sum(diff))
        else:
            flux.append(0)

        prev_spectrum = spectrum

    flux = np.array(flux)

    # Normalize
    if np.max(flux) > 0:
        flux = flux / np.max(flux)

    # Find peaks above threshold
    onsets = []
    for i in range(1, len(flux) - 1):
        if flux[i] > threshold and flux[i] > flux[i-1] and flux[i] > flux[i+1]:
            onset_time = i * hop_size / sample_rate
            onsets.append(onset_time)

    return onsets


def rms_energy(signal, window_size=2048, hop_size=512):
    """
    Compute RMS energy over time.

    Args:
        signal: Input signal
        window_size: Window size
        hop_size: Hop size

    Returns:
        (times, rms_values)
    """
    num_windows = (len(signal) - window_size) // hop_size + 1
    rms_vals = []

    for i in range(num_windows):
        start = i * hop_size
        end = start + window_size
        if end > len(signal):
            break

        window = signal[start:end]
        rms = np.sqrt(np.mean(window ** 2))
        rms_vals.append(rms)

    times = np.arange(len(rms_vals)) * hop_size / 44100

    return times, np.array(rms_vals)


def zero_crossing_rate(signal, window_size=2048, hop_size=512):
    """
    Compute zero crossing rate over time.

    Args:
        signal: Input signal
        window_size: Window size
        hop_size: Hop size

    Returns:
        (times, zcr_values)
    """
    num_windows = (len(signal) - window_size) // hop_size + 1
    zcr_vals = []

    for i in range(num_windows):
        start = i * hop_size
        end = start + window_size
        if end > len(signal):
            break

        window = signal[start:end]
        # Count zero crossings
        zcr = np.sum(np.abs(np.diff(np.sign(window)))) / (2 * window_size)
        zcr_vals.append(zcr)

    times = np.arange(len(zcr_vals)) * hop_size / 44100

    return times, np.array(zcr_vals)


def spectral_centroid(signal, sample_rate=44100):
    """
    Compute spectral centroid (brightness measure).

    Args:
        signal: Input signal
        sample_rate: Sample rate in Hz

    Returns:
        Spectral centroid in Hz
    """
    freqs, mags = fft_spectrum(signal, sample_rate)

    # Weighted average of frequencies
    centroid = np.sum(freqs * mags) / np.sum(mags)

    return centroid


def spectral_rolloff(signal, sample_rate=44100, rolloff_percent=0.85):
    """
    Compute spectral rolloff frequency.

    Args:
        signal: Input signal
        sample_rate: Sample rate in Hz
        rolloff_percent: Percentage of total energy (default 0.85 = 85%)

    Returns:
        Rolloff frequency in Hz
    """
    freqs, mags = fft_spectrum(signal, sample_rate)

    total_energy = np.sum(mags)
    threshold = rolloff_percent * total_energy

    cumsum = np.cumsum(mags)
    rolloff_idx = np.where(cumsum >= threshold)[0]

    if len(rolloff_idx) > 0:
        return freqs[rolloff_idx[0]]
    else:
        return freqs[-1]


def spectral_flatness(signal):
    """
    Compute spectral flatness (noisiness measure).
    Values close to 1 indicate noise-like, close to 0 indicate tonal.

    Args:
        signal: Input signal

    Returns:
        Spectral flatness (0 to 1)
    """
    freqs, mags = fft_spectrum(signal)

    # Avoid log of zero
    mags = mags + 1e-10

    # Geometric mean / arithmetic mean
    geometric_mean = np.exp(np.mean(np.log(mags)))
    arithmetic_mean = np.mean(mags)

    if arithmetic_mean > 0:
        flatness = geometric_mean / arithmetic_mean
    else:
        flatness = 0

    return flatness


def harmonic_product_spectrum(signal, sample_rate=44100, num_harmonics=5):
    """
    Harmonic Product Spectrum for pitch detection.

    Args:
        signal: Input signal
        sample_rate: Sample rate in Hz
        num_harmonics: Number of harmonics to use

    Returns:
        Detected fundamental frequency in Hz
    """
    freqs, mags = fft_spectrum(signal, sample_rate)

    # Initialize HPS
    hps = mags.copy()

    # Downsample and multiply
    for h in range(2, num_harmonics + 1):
        decimated = mags[::h]
        min_len = min(len(hps), len(decimated))
        hps[:min_len] *= decimated[:min_len]

    # Find peak
    peak_idx = np.argmax(hps)
    fundamental = freqs[peak_idx]

    return fundamental


def detect_tempo(signal, sample_rate=44100):
    """
    Simple tempo detection using onset detection.

    Args:
        signal: Input signal
        sample_rate: Sample rate in Hz

    Returns:
        Estimated tempo in BPM (or None if cannot detect)
    """
    onsets = detect_onsets(signal, sample_rate)

    if len(onsets) < 2:
        return None

    # Compute inter-onset intervals
    intervals = np.diff(onsets)

    if len(intervals) == 0:
        return None

    # Median interval
    median_interval = np.median(intervals)

    # Convert to BPM
    bpm = 60.0 / median_interval

    return bpm


def get_peaks(data, threshold=0.5, min_distance=10):
    """
    Find peaks in data.

    Args:
        data: Input data array
        threshold: Minimum peak value (normalized)
        min_distance: Minimum distance between peaks

    Returns:
        Array of peak indices
    """
    # Normalize
    if np.max(data) > 0:
        normalized = data / np.max(data)
    else:
        return np.array([])

    peaks = []
    for i in range(1, len(normalized) - 1):
        if normalized[i] > threshold and normalized[i] > normalized[i-1] and normalized[i] > normalized[i+1]:
            # Check minimum distance
            if not peaks or i - peaks[-1] >= min_distance:
                peaks.append(i)

    return np.array(peaks)
