from music_vae.music.note_container_lib import steps_per_bar_in_quantized_container
from music_vae.music import settings

MELODY_NOTE_OFF = settings.MELODY_NOTE_OFF
MELODY_NO_EVENT = settings.MELODY_NO_EVENT
MIN_MIDI_PITCH = settings.MIN_MIDI_PITCH


class Melody:
    def __init__(self, events=None, start_step=0, steps_per_bar=None, steps_per_quarter=None):
        self.events = events or []
        self.steps_per_bar = steps_per_bar
        self.steps_per_quarter = steps_per_quarter

        self.start_step = start_step
        self.end_step = start_step + len(self.events)

        # Replace MELODY_NOTE_OFF events with MELODY_NO_EVENT before first note.
        if len(self.events) > 0:
            cleaned_events = list(self.events)
            for i, e in enumerate(self.events):
                if e not in (MELODY_NO_EVENT, MELODY_NOTE_OFF):
                    break
                cleaned_events[i] = MELODY_NO_EVENT
            self.events = cleaned_events

    def __len__(self):
        return len(self.events)

    def __iter__(self):
        return iter(self.events)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.events[item]
        elif isinstance(item, slice):
            events = self.events.__getitem__(item)
            return type(self)(events=events,
                              start_step=self.start_step + (item.start or 0),
                              steps_per_bar=self.steps_per_bar,
                              steps_per_quarter=self.steps_per_quarter)

    def add_note(self, pitch, start_step, end_step):
        assert end_step > start_step

        self.set_length(end_step + 1)

        self.events[start_step] = pitch
        self.events[end_step] = MELODY_NOTE_OFF

        for i in range(start_step + 1, end_step):
            self.events[i] = MELODY_NO_EVENT

    def set_length(self, steps):
        old_len = len(self)

        if steps > old_len:
            self.events.extend([MELODY_NO_EVENT] * (steps - old_len))
        else:
            del self.events[steps:]

        self.end_step = self.start_step + steps

        if steps > old_len:
            # When extending the music on the right end any sustained notes.
            for i in reversed(range(old_len)):
                if self.events[i] == MELODY_NOTE_OFF:
                    break
                elif self.events[i] != MELODY_NO_EVENT:
                    self.events[old_len] = MELODY_NOTE_OFF
                    break

    def get_last_on_off_events(self):
        last_off = len(self)
        for i in range(len(self) - 1, -1, -1):
            if self.events[i] == MELODY_NOTE_OFF:
                last_off = i
            if self.events[i] >= MIN_MIDI_PITCH:
                return i, last_off

        raise ValueError('No events in the stream')

    @classmethod
    def from_quantized_note_container(cls, qnc, instrument, search_start_step, gap_bars, filter_drums, pad_end=True):
        melody = cls()

        melody.steps_per_bar = steps_per_bar = int(steps_per_bar_in_quantized_container(qnc))
        melody.steps_per_quarter = qnc.steps_per_quarter

        # Sort track by note start times, and secondarily by pitch descending.
        notes = sorted([n for n in qnc.notes
                        if n.instrument == instrument and n.quantized_start_step >= search_start_step],
                       key=lambda note: (note.quantized_start_step, -note.pitch))

        if not notes:
            return None

        # music start at beginning of a bar
        start_step = notes[0].quantized_start_step
        melody_start_step = start_step - (start_step - search_start_step) % steps_per_bar

        for i, note in enumerate(notes):
            if (filter_drums and note.is_drum) or note.velocity == 0:
                continue

            start_index = note.quantized_start_step - melody_start_step
            end_index = note.quantized_end_step - melody_start_step

            if not melody.events:
                melody.add_note(note.pitch, start_index, end_index)
                continue

            # If `start_index` comes before or lands on an already added note's start
            # step, we cannot add it. In that case either discard the music or keep
            # the highest pitch.
            last_on, last_off = melody.get_last_on_off_events()
            on_distance = start_index - last_on
            off_distance = start_index - last_off
            if on_distance == 0:
                continue
            elif on_distance < 0:
                raise Exception()

            # If a gap of `gap` or more steps is found, end the music.
            if len(melody) and off_distance >= gap_bars * steps_per_bar:
                break

            # Add the note-on and off events to the music.
            melody.add_note(note.pitch, start_index, end_index)

        if not melody.events:
            return None

        melody.start_step = melody_start_step

        # Strip final MELODY_NOTE_OFF event.
        if melody.events[-1] == MELODY_NOTE_OFF:
            del melody.events[-1]

        length = len(melody)
        # Optionally round up `_end_step` to a multiple of `steps_per_bar`.
        if pad_end:
            length += -len(melody) % steps_per_bar
        melody.set_length(length)

        return melody
