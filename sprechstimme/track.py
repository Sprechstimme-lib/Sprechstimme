import numpy as np
from .core import midi_to_freq, note_to_freq, chord_to_notes, _SYNTHS, DEFAULT_SR
from .playback import save_wav

class Track:
    def __init__(self, bpm=120, sample_rate=DEFAULT_SR):
        self.bpm = bpm
        self.sample_rate = sample_rate
        self.events = []  # list of (synth_name, notes, beats, vol)

    def add(self, synth, notes, duration=1, vol=5.0):
        """
        notes: int/float/str, or list of them (chord), or list of (note,dur) or (note,dur,vol) for sequences.
        duration: number of beats (relative to bpm). If notes is sequence of tuples, that overrides.
        vol: dynamics from 1.0 to 10.0 (default 5.0).
        """
        self.events.append((synth, notes, duration, vol))

    def addChord(self, synth, chord, duration=1, vol=5.0):

        notes = chord_to_notes(chord)

        self.events.append((synth, notes, duration, vol))
    
    def _render_event(self, synth_name, notes, sec, vol=5.0):
        sr = self.sample_rate
        t = np.linspace(0, sec, int(sr * sec), endpoint=False)
        out = np.zeros_like(t)
        synth = _SYNTHS.get(synth_name)
        if synth is None:
            raise ValueError(f"Synth '{synth_name}' not registered")
        # If notes is sequence of (note,dur) or (note,dur,vol) tuples -> render them sequentially inside this event
        if len(notes) > 0 and isinstance(notes[0], (list, tuple)) and len(notes[0]) in [2, 3]:
            # render each subsegment and concatenate
            segs = []
            for item in notes:
                if len(item) == 2:
                    n, dur = item
                    note_vol = vol
                elif len(item) == 3:
                    n, dur, note_vol = item
                else:
                    raise ValueError("Sequence items must be (note, dur) or (note, dur, vol)")
                seg_sec = (60.0 / self.bpm) * dur
                seg_amplitude = np.clip(float(note_vol), 1.0, 10.0) / 10.0
                seg_t = np.linspace(0, seg_sec, int(sr * seg_sec), endpoint=False)
                seg_sig = np.zeros_like(seg_t)
                if isinstance(n, (list, tuple)):
                    note_list = n
                else:
                    note_list = [n]
                for nn in note_list:
                    if isinstance(nn, int):
                        freq = midi_to_freq(nn)
                    elif isinstance(nn, float):
                        freq = float(nn)
                    elif isinstance(nn, str):
                        freq = note_to_freq(nn)
                    else:
                        raise ValueError("Unsupported note type in sequence")
                    seg_sig += synth["wavetype"](seg_t, freq=freq, amp=seg_amplitude)
                if len(note_list) > 0:
                    seg_sig = seg_sig / float(len(note_list))
                if synth.get("envelope"):
                    from .core import _apply_envelope
                    seg_sig = _apply_envelope(seg_sig, sr, synth["envelope"])
                for f in synth.get("filters", []):
                    if f:
                        seg_sig = f(seg_sig, sr)
                segs.append(seg_sig)
            return np.concatenate(segs)
        else:
            # normal chord/notes for this event duration sec
            amplitude = np.clip(float(vol), 1.0, 10.0) / 10.0
            for n in notes if isinstance(notes, (list, tuple)) else [notes]:
                if isinstance(n, int):
                    freq = midi_to_freq(n)
                elif isinstance(n, float):
                    freq = float(n)
                elif isinstance(n, str):
                    freq = note_to_freq(n)
                else:
                    raise ValueError("Unsupported note type")
                out += synth["wavetype"](t, freq=freq, amp=amplitude)
            if (isinstance(notes, (list, tuple)) and len(notes) > 0):
                out = out / float(len(notes))
            if synth.get("envelope"):
                from .core import _apply_envelope
                out = _apply_envelope(out, sr, synth["envelope"])
            for f in synth.get("filters", []):
                if f:
                    out = f(out, sr)
            return out

    def play(self):
        """Play the track by rendering events sequentially and playing the combined buffer."""
        buffers = []
        for synth, notes, beats, vol in self.events:
            sec = (60.0 / self.bpm) * beats
            buffers.append(self._render_event(synth, notes, sec, vol))
        if not buffers:
            return
        combined = np.concatenate(buffers)
        # normalize
        max_abs = np.max(np.abs(combined)) if combined.size else 0.0
        if max_abs > 1e-9:
            combined = combined / max(1.0, max_abs)
        from .playback import play_array
        play_array(combined, self.sample_rate)

    def export(self, filename="output.wav"):
        buffers = []
        for synth, notes, beats, vol in self.events:
            sec = (60.0 / self.bpm) * beats
            buffers.append(self._render_event(synth, notes, sec, vol))
        combined = np.concatenate(buffers) if buffers else np.array([], dtype=float)
        save_wav(filename, combined, sample_rate=self.sample_rate)
