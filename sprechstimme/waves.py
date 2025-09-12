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