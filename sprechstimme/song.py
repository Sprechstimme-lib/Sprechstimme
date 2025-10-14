import numpy as np
from .core import midi_to_freq, note_to_freq, chord_to_notes, _SYNTHS, DEFAULT_SR
from .playback import save_wav

class Song:
    """
    A multi-track composition with a universal beat counter.

    Allows multiple tracks to play simultaneously, with events positioned
    at specific beat locations (e.g., beat 200.5).
    """

    def __init__(self, bpm=120, sample_rate=DEFAULT_SR):
        """
        Initialize a new Song.

        Parameters:
        - bpm: Beats per minute for the entire song
        - sample_rate: Audio sample rate in Hz
        """
        self.bpm = bpm
        self.sample_rate = sample_rate
        self.tracks = {}  # track_name -> list of (beat_position, synth, notes, duration_in_beats)
        self.total_beats = 0  # automatically calculated from all events

    def add_track(self, track_name):
        """
        Create a new track in the song.

        Parameters:
        - track_name: Unique identifier for the track
        """
        if track_name in self.tracks:
            raise ValueError(f"Track '{track_name}' already exists")
        self.tracks[track_name] = []

    def add(self, track_name, synth, notes, beat_position, duration=1):
        """
        Add an event to a specific track at a specific beat position.

        Parameters:
        - track_name: Name of the track to add to
        - synth: Name of the synthesizer to use
        - notes: Note(s) to play (int/float/str or list for chords)
        - beat_position: When to start this event (in beats, can be fractional like 200.5)
        - duration: Duration of the event in beats (default: 1)
        """
        if track_name not in self.tracks:
            # Auto-create track if it doesn't exist
            self.add_track(track_name)

        self.tracks[track_name].append((beat_position, synth, notes, duration))

        # Update total beats if this event extends the song
        end_beat = beat_position + duration
        if end_beat > self.total_beats:
            self.total_beats = end_beat

    def add_chord(self, track_name, synth, chord, beat_position, duration=1):
        """
        Add a chord event to a specific track at a specific beat position.

        Parameters:
        - track_name: Name of the track to add to
        - synth: Name of the synthesizer to use
        - chord: Chord notation (e.g., "C4", "Am3", "F#5")
        - beat_position: When to start this event (in beats)
        - duration: Duration of the event in beats (default: 1)
        """
        notes = chord_to_notes(chord)
        self.add(track_name, synth, notes, beat_position, duration)

    def _render_event(self, synth_name, notes, sec):
        """
        Render a single event (synth + notes) for a given duration.

        Returns a numpy array of audio samples.
        """
        sr = self.sample_rate
        t = np.linspace(0, sec, int(sr * sec), endpoint=False)
        out = np.zeros_like(t)

        synth = _SYNTHS.get(synth_name)
        if synth is None:
            raise ValueError(f"Synth '{synth_name}' not registered")

        # Handle different note input formats
        note_list = notes if isinstance(notes, (list, tuple)) else [notes]

        for n in note_list:
            if isinstance(n, int):
                freq = midi_to_freq(n)
            elif isinstance(n, float):
                freq = float(n)
            elif isinstance(n, str):
                freq = note_to_freq(n)
            else:
                raise ValueError(f"Unsupported note type: {type(n)}")

            out += synth["wavetype"](t, freq=freq, amp=1.0)

        # Normalize by number of notes to avoid clipping
        if len(note_list) > 0:
            out = out / float(len(note_list))

        # Apply envelope if configured
        if synth.get("envelope"):
            from .core import _apply_envelope
            out = _apply_envelope(out, sr, synth["envelope"])

        # Apply filters in order
        for f in synth.get("filters", []):
            if f:
                out = f(out, sr)

        return out

    def _render_track(self, track_events):
        """
        Render all events in a track, returning a single audio buffer
        that spans the entire song duration.
        """
        # Calculate total samples needed for the entire song
        total_seconds = (60.0 / self.bpm) * self.total_beats
        total_samples = int(self.sample_rate * total_seconds)

        # Initialize empty buffer for this track
        track_buffer = np.zeros(total_samples, dtype=float)

        # Render each event and place it at the correct position
        for beat_pos, synth, notes, duration in track_events:
            # Calculate timing
            start_sec = (60.0 / self.bpm) * beat_pos
            event_sec = (60.0 / self.bpm) * duration

            # Calculate sample positions
            start_sample = int(start_sec * self.sample_rate)

            # Render the event
            event_audio = self._render_event(synth, notes, event_sec)
            event_length = len(event_audio)

            # Place the event in the track buffer (with bounds checking)
            end_sample = min(start_sample + event_length, total_samples)
            actual_length = end_sample - start_sample

            if actual_length > 0:
                # Mix the event into the track buffer
                track_buffer[start_sample:end_sample] += event_audio[:actual_length]

        return track_buffer

    def _render(self):
        """
        Render all tracks and mix them together.

        Returns a mixed audio buffer containing all tracks.
        """
        if not self.tracks or self.total_beats == 0:
            return np.array([], dtype=float)

        # Render each track
        track_buffers = []
        for track_name, track_events in self.tracks.items():
            if track_events:  # Only render non-empty tracks
                track_buffer = self._render_track(track_events)
                track_buffers.append(track_buffer)

        if not track_buffers:
            return np.array([], dtype=float)

        # Mix all tracks together
        mixed = np.sum(track_buffers, axis=0)

        # Normalize to prevent clipping
        max_abs = np.max(np.abs(mixed))
        if max_abs > 1e-9:
            mixed = mixed / max(1.0, max_abs)

        return mixed

    def play(self):
        """
        Render and play the entire song through audio output.
        """
        audio = self._render()
        if audio.size == 0:
            print("Warning: Song is empty, nothing to play")
            return

        from .playback import play_array
        play_array(audio, self.sample_rate)

    def export(self, filename="output.wav"):
        """
        Render and export the song to a WAV file.

        Parameters:
        - filename: Output filename (default: "output.wav")
        """
        audio = self._render()
        save_wav(filename, audio, sample_rate=self.sample_rate)

    def get_duration(self):
        """
        Get the total duration of the song.

        Returns a dictionary with duration in beats, seconds, and formatted time.
        """
        total_seconds = (60.0 / self.bpm) * self.total_beats
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60

        return {
            "beats": self.total_beats,
            "seconds": total_seconds,
            "formatted": f"{minutes}:{seconds:05.2f}"
        }

    def list_tracks(self):
        """
        Get information about all tracks in the song.

        Returns a dictionary with track names and event counts.
        """
        return {
            track_name: {
                "events": len(events),
                "synths": list(set(synth for _, synth, _, _ in events))
            }
            for track_name, events in self.tracks.items()
        }
