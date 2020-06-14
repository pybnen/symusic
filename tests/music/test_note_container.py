from glob import glob
import pickle
import numpy as np
import pretty_midi

from magenta.music import sequences_lib
from magenta.scripts.convert_dir_to_note_sequences import convert_midi
import magenta.music as mm

from music_vae.music.note_container import NoteContainer
from music_vae.music.note_container_lib import (split_on_time_signature_tempo_change,
                                                quantize_note_container,
                                                MultipleTimeSignatureError)

MIDI_PATH_1 = r".\files\midi\midi1.mid"
MIDI_PATH_2 = r".\files\midi\midi2.mid"
MIDI_PATH_3 = r".\files\midi\midi3.mid"
MIDI_PATH_4 = r".\files\midi\midi4.mid"

midi_paths = glob("./files/midi/*.mid")
midi_big_paths = glob("./files/midi_big/*.mid")


def assert_nc_ns(nc, ns, quantized=False):
    assert nc.total_time == ns.total_time

    for ns_note, nc_note in zip(ns.notes, nc.notes):
        assert_note_equal(ns_note, nc_note, quantized)

    for ns_ts, nc_ts in zip(ns.time_signatures, nc.time_signatures):
        assert ns_ts.time == nc_ts.time
        assert ns_ts.denominator == nc_ts.denominator
        assert ns_ts.numerator == nc_ts.numerator

    for ns_t, nc_t in zip(ns.tempos, nc.tempos):
        assert ns_t.time == nc_t.time
        assert ns_t.qpm == nc_t.qpm

    if quantized:
        assert nc.total_quantized_steps == ns.total_quantized_steps
        assert nc.steps_per_quarter == ns.quantization_info.steps_per_quarter


def assert_note_equal(n1, n2, quantized=False):
    assert n1.instrument == n2.instrument
    assert n1.program == n2.program
    assert n1.start_time == n2.start_time
    assert n1.end_time == n2.end_time
    assert n1.pitch == n2.pitch
    assert n1.velocity == n2.velocity
    assert n1.is_drum == n2.is_drum

    if quantized:
        n1.quantized_start_step = n2.quantized_start_step
        n1.quantized_end_step = n2.quantized_end_step


def create_notes(velocity=100):
    notes = []
    notes.append(pretty_midi.Note(velocity=velocity, pitch=70, start=0.0, end=1.0))
    notes.append(pretty_midi.Note(velocity=velocity, pitch=80, start=1.0, end=2.0))
    notes.append(pretty_midi.Note(velocity=velocity, pitch=75, start=1.5, end=2.5))
    notes.append(pretty_midi.Note(velocity=velocity, pitch=60, start=5.1, end=5.34))

    return notes


def create_simple_midi(tempo=120.):
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=0)
    pm.instruments.append(instrument)
    instrument.notes = create_notes()
    return pm


def test_note_container():
    for midi_path in midi_paths:
        pm = pretty_midi.PrettyMIDI(midi_path)
        nc = NoteContainer.from_pretty_midi(pm)
        for nc_tempo, pm_time, pm_tempo in zip(nc.tempos, *pm.get_tempo_changes()):
            assert np.isclose(nc_tempo.time, pm_time)
            assert np.isclose(nc_tempo.qpm, pm_tempo)

        for nc_ts, pm_ts in zip(nc.time_signatures, pm.time_signature_changes):
            assert np.isclose(nc_ts.time, pm_ts.time)
            assert nc_ts.numerator == pm_ts.numerator
            assert nc_ts.denominator == pm_ts.denominator

        total_time = 0.0
        midi_notes = []
        for num_instrument, midi_instrument in enumerate(pm.instruments):
            for midi_note in midi_instrument.notes:
                if midi_note.end > total_time:
                    total_time = midi_note.end

                midi_notes.append((midi_instrument.program, num_instrument,
                                   midi_instrument.is_drum, midi_note))

        assert np.isclose(total_time, nc.total_time)

        for nc_note, (program, instrument, is_drum, midi_note) in zip(nc.notes, midi_notes):
            assert nc_note.instrument == instrument
            assert nc_note.program == program
            assert nc_note.start_time == midi_note.start
            assert nc_note.end_time == midi_note.end
            assert nc_note.pitch == midi_note.pitch
            assert nc_note.velocity == midi_note.velocity
            assert nc_note.is_drum == is_drum


def test_split_on_time_signature_tempo_change():
    midi_path = "./files/midi/3a0eee9e3a8e59524c36e6f8e4a0c698.mid"
    pm = pretty_midi.PrettyMIDI(midi_path)
    nc = NoteContainer.from_pretty_midi(pm)

    split_ncs = split_on_time_signature_tempo_change(nc)
    # target_split_times = [0.0, 0.410958, 26.7122, 27.328707, 58.850897925]
    assert len(split_ncs) == 4

    for midi_path in midi_paths:
        pm = pretty_midi.PrettyMIDI(midi_path)
        nc = NoteContainer.from_pretty_midi(pm)
        total_notes = len(nc.notes)
        split_ncs = split_on_time_signature_tempo_change(nc)
        # this must not always hold but is a good sanity check
        assert total_notes == np.sum([len(n.notes) for n in split_ncs])


def test_quantize(pm, target_steps):
    nc = NoteContainer.from_pretty_midi(pm)
    qnc = quantize_note_container(nc, 4)

    for (start, end), note in zip(target_steps, qnc.notes):
        assert note.quantized_start_step == start
        assert note.quantized_end_step == end


def test_quantize_note_container():
    test_quantize(create_simple_midi(120.), [(0, 8), (8, 16), (12, 20), (41, 43)])
    test_quantize(create_simple_midi(140.), [(0, 9), (9, 19), (14, 23), (48, 50)])

    pm = pretty_midi.PrettyMIDI(MIDI_PATH_1)
    with open("./files/targets/midi1_target_steps.pkl", "rb") as f:
        target_steps = pickle.load(f)
    test_quantize(pm, target_steps)

    try:
        test_quantize(pretty_midi.PrettyMIDI("./files/midi/3a0eee9e3a8e59524c36e6f8e4a0c698.mid"), None)
        assert False
    except Exception as e:
        assert isinstance(e, MultipleTimeSignatureError)


def test_split_and_quantize():
    for midi_path in midi_big_paths:
        # print(midi_path)
        ns = convert_midi("", '', midi_path)
        note_sequences = sequences_lib.split_note_sequence_on_time_changes(ns)

        pm = pretty_midi.PrettyMIDI(midi_path)
        nc = NoteContainer.from_pretty_midi(pm)

        split_ncs = split_on_time_signature_tempo_change(nc)

        for nc, ns in zip(split_ncs, note_sequences):
            assert_nc_ns(nc, ns)

            qnc = quantize_note_container(nc, 4)
            qns = mm.quantize_note_sequence(ns, 4)
            assert_nc_ns(qnc, qns, quantized=True)


if __name__ == "__main__":
    test_note_container()
    test_split_on_time_signature_tempo_change()
    test_quantize_note_container()
    test_split_and_quantize()
