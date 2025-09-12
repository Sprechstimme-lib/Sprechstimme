import numpy as np
from .core import play, midi_to_freq
from .playback import play_array, save_wav

class Track:
    def __init__(self, bpm=120, sample_rate=44100):
        self.bpm = bpm
        self.sample_rate = sample_rate
        self.events = []

    def add(self, synth, notes, duration=1):
        self.events.append((synth, notes, duration))

    def play(self):
        for synth, notes, duration in self.events:
            sec = (60 / self.bpm) * duration
            play(synth, notes, duration=sec, sample_rate=self.sample_rate)

    def export(self, filename="output.wav"):
        buffers = []
        for synth, notes, duration in self.events:
            sec = (60 / self.bpm) * duration
            # You could refactor to generate signal directly instead of playing
            from .core import _SYNTHS
            t = np.linspace(0, sec, int(self.sample_rate * sec), endpoint=False)
            synth_data = np.zeros_like(t)
            for n in notes:
                freq = midi_to_freq(n) if isinstance(n, (int, float)) else n
                synth_data += _SYNTHS[synth]["wavetype"](t, freq=freq)
            buffers.append(synth_data)

        combined = np.concatenate(buffers)
        save_wav(filename, combined, sample_rate=self.sample_rate)