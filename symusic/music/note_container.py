from collections import namedtuple

InstrumentInfo = namedtuple("InstrumentInfo", "name instrument")


class TimeSignature:
    def __init__(self, time, numerator, denominator):
        self.time = time
        self.numerator = numerator
        self.denominator = denominator


class Tempo:
    def __init__(self, time, qpm):
        self.time = time
        self.qpm = qpm


class Note:
    def __init__(self, instrument, program, start_time, end_time, pitch, velocity, is_drum):
        self.instrument = instrument
        self.program = program
        self.start_time = start_time
        self.end_time = end_time
        self.pitch = pitch
        self.velocity = velocity
        self.is_drum = is_drum

        self.quantized_start_step = None
        self.quantized_end_step = None


class NoteContainer:
    def __init__(self, notes=None, total_time=None, instrument_infos=None, tempos=None, time_signatures=None):
        self.notes = notes or []
        self.total_time = total_time
        self.instrument_infos = instrument_infos or []
        self.tempos = tempos or []
        self.time_signatures = time_signatures or []

        self.steps_per_quarter = None
        self.total_quantized_steps = None

    @classmethod
    def from_pretty_midi(cls, pm):
        tempos = [Tempo(time=time, qpm=qpm) for time, qpm in zip(*pm.get_tempo_changes())]
        time_signatures = [TimeSignature(time=ts.time, numerator=ts.numerator, denominator=ts.denominator)
                           for ts in pm.time_signature_changes]

        notes = []
        instrument_infos = []
        total_time = 0.0

        for num_instrument, midi_instrument in enumerate(pm.instruments):
            if midi_instrument.name:
                instrument_infos.append(InstrumentInfo(name=midi_instrument.name,
                                                       instrument=num_instrument))

            for midi_note in midi_instrument.notes:
                if midi_note.end > total_time:
                    total_time = midi_note.end

                note = Note(instrument=num_instrument,
                            program=midi_instrument.program,
                            start_time=midi_note.start,
                            end_time=midi_note.end,
                            pitch=midi_note.pitch,
                            velocity=midi_note.velocity,
                            is_drum=midi_instrument.is_drum)
                notes.append(note)

        return cls(notes, total_time=total_time,
                   instrument_infos=instrument_infos,
                   tempos=tempos,
                   time_signatures=time_signatures)
