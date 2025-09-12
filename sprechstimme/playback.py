import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os

def play_array(arr, sample_rate=44100):
    sd.play(arr, samplerate=sample_rate)
    sd.wait()

def save_wav(filename, arr, sample_rate=44100):
    arr_16 = np.int16(arr / np.max(np.abs(arr)) * 32767)
    write(filename, sample_rate, arr_16)
    print(f"Saved to {os.path.abspath(filename)}")