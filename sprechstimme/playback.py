import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os

def play_array(arr, sample_rate=44100):
    """
    Play a numpy array (1D) via sounddevice. Ensure -1..1 range.
    Blocks until finished.
    """
    if arr is None or arr.size == 0:
        return
    # ensure float32
    arr = np.asarray(arr, dtype=np.float32)
    max_abs = np.max(np.abs(arr))
    if max_abs > 0:
        if max_abs > 1.0:
            arr = arr / max_abs
    sd.play(arr, samplerate=sample_rate)
    sd.wait()

def save_wav(filename, arr, sample_rate=44100):
    if arr is None or arr.size == 0:
        raise ValueError("No audio data to save")
    arr = np.asarray(arr, dtype=np.float32)
    max_abs = np.max(np.abs(arr))
    if max_abs > 0:
        arr = arr / max(1.0, max_abs)
    arr_16 = (arr * 32767).astype('<i2')
    write(filename, sample_rate, arr_16)
    print(f"Saved to {os.path.abspath(filename)}")