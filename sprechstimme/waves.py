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
    return amp * np.sin(2 * np.pi * freq1 * t) * np.sin(2 * np.pi * freq2 * t)
    
