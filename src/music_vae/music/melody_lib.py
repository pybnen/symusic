# most methods/functionality is taken from magenta
# https://github.com/magenta/magenta
# see: https://github.com/magenta/magenta/blob/master/magenta/music/sequences_lib.py
#   and https://github.com/magenta/magenta/blob/master/magenta/models/music_vae/data.py
import numpy as np
import copy
import pretty_midi

from music_vae.music import settings
from music_vae.music.melody import Melody
from music_vae.music.note_container_lib import NoteContainer,\
    split_on_time_signature_tempo_change, quantize_note_container,\
    steps_per_bar_in_quantized_container


MELODY_NOTE_OFF = settings.MELODY_NOTE_OFF
MELODY_NO_EVENT = settings.MELODY_NO_EVENT
MIN_MIDI_PITCH = settings.MIN_MIDI_PITCH


class MelodyExtractor:

    def __init__(self,
                 min_pitch=settings.PIANO_MIN_MIDI_PITCH,
                 max_pitch=settings.PIANO_MAX_MIDI_PITCH,
                 valid_programs=settings.MELODY_PROGRAMS,
                 max_bars=100,
                 slice_bars=2,
                 steps_per_quarter=4,
                 gap_bars=1.0,
                 pad_end=True):

        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.valid_programs = valid_programs
        self.max_bars = max_bars
        self.slice_bars = slice_bars
        self.steps_per_quarter = steps_per_quarter
        self.gap_bars = gap_bars
        self.pad_end=pad_end

    def filter_notes(self, nc):
        def filter(note):
            if self.valid_programs is not None and note.program not in self.valid_programs:
                return False
            return self.min_pitch <= note.pitch <= self.max_pitch

        notes = list(nc.notes)
        del nc.notes[:]
        nc.notes.extend([n for n in notes if filter(n)])

    def extract_melodies(self, midi_path):
        try:
            pm = pretty_midi.PrettyMIDI(midi_path)
        except EOFError:
            return []

        note_containers = split_on_time_signature_tempo_change(NoteContainer.from_pretty_midi(pm))

        total_melodies = []
        for nc in note_containers:
            if not (self.valid_time_signature(nc) and self.valid_tempos(nc)):
                continue
            self.filter_notes(nc)
            qnc = quantize_note_container(nc, self.steps_per_quarter)
            melodies = extract_melodies(qnc, gap_bars=self.gap_bars,
                                        max_bars=self.max_bars,
                                        pad_end=self.pad_end)
            total_melodies.extend(melodies)

        # slice melodies
        # unique melodies
        # finished

        return total_melodies

    @staticmethod
    def valid_time_signature(nc):
        """Checks if all existing time signature changes are 4/4"""
        if len(nc.time_signatures) == 0:
            # if no time signature is given it is assumed 4/4
            # see: http://midi.teragonaudio.com/tech/midifile/time.htm
            return True
        return True
        # there is at least one time signature, all time signatures must be 4/4
        for time_signature in nc.time_signatures:
            if time_signature.denominator != 4 or time_signature.numerator != 4:
                return False
        return True

    @staticmethod
    def valid_tempos(nc):
        """Checks if all existing tempos are equal"""
        if len(nc.tempos) > 0:
            qpm = settings.DEFAULT_QPM
            for tempo in nc.tempos:
                if tempo.time == 0.0:
                    qpm = tempo.qpm
                if tempo.qpm != qpm:
                    return False
        return True


def extract_melodies(qnc, min_bars=1, gap_bars=1.0, max_bars=100, pad_end=True):
    melodies = []

    instruments = set(n.instrument for n in qnc.notes)
    steps_per_bar = int(steps_per_bar_in_quantized_container(qnc))

    # TODO make not hard coded, but this it the way it work in magenta
    max_steps_truncate = max_bars * 16 if max_bars is not None else None

    for instrument in instruments:
        instrument_search_start_step = 0

        while True:
            melody = Melody.from_quantized_note_container(qnc, instrument=instrument,
                                                          search_start_step=instrument_search_start_step,
                                                          gap_bars=gap_bars,
                                                          filter_drums=True,
                                                          pad_end=pad_end)

            if melody is None:
                break

            # Start search for next music on next bar boundary (inclusive).
            instrument_search_start_step = melody.end_step + melody.end_step % steps_per_bar

            # Require a certain music length.
            if len(melody) < steps_per_bar * min_bars:
                continue

            # Discard melodies that are too long.
            if max_steps_truncate is not None and len(melody) > max_steps_truncate:
                truncated_length = max_steps_truncate
                if pad_end:
                    truncated_length -= max_steps_truncate % melody.steps_per_bar
                melody.set_length(truncated_length)

            melodies.append(melody)

    return melodies


# -----------------------------


def melody_to_midi(melody, velocity=80, program=0):
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    notes = []
    time_per_step = 1/8.0

    pitch = MELODY_NOTE_OFF
    start_time = cur_time = 0.0

    assert(MIN_MIDI_PITCH > MELODY_NOTE_OFF > MELODY_NO_EVENT)

    # add MELODY_NOTE_OFF event at end of melody
    melody_copy = copy.copy(melody)
    if isinstance(melody_copy, (tuple, list)):
        melody_copy += (MELODY_NOTE_OFF,)
    elif isinstance(melody_copy, np.ndarray):
        melody_copy = np.concatenate((melody_copy, [MELODY_NOTE_OFF]))
    else:
        assert(False)

    for event in melody_copy:
        assert (event >= MELODY_NO_EVENT)

        if event >= MELODY_NOTE_OFF:
            # create previous note
            if pitch >= MIN_MIDI_PITCH:
                note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=cur_time)
                notes.append(note)

            # update current note
            pitch = event
            start_time = cur_time

        cur_time += time_per_step

    instrument.notes = notes
    pm.instruments.append(instrument)

    return pm


def melody_to_pianoroll(melody, velocity=80):
    pianoroll = np.zeros((len(melody), 128))
    prev_event = -1
    for i, event in enumerate(melody):
        if event >= -1:
            if event >= 0:
                pianoroll[i, event] = velocity
            prev_event = event

        if event == -2 and prev_event >= 0:
            pianoroll[i, prev_event] = velocity
    return pianoroll