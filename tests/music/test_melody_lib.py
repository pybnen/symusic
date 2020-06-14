from glob import glob
import pretty_midi

from magenta.music import sequences_lib
from magenta.scripts.convert_dir_to_note_sequences import convert_midi
import magenta.models.music_vae.data as music_vae_data
import magenta.music as mm
from magenta.pipelines.melody_pipelines import extract_melodies

from music_vae.music import melody_lib
from music_vae.music.note_container_lib import split_on_time_signature_tempo_change, quantize_note_container
from music_vae.music.note_container import NoteContainer

MIDI_PATH_1 = r".\files\midi\midi1.mid"
MIDI_PATH_2 = r".\files\midi\midi2.mid"
MIDI_PATH_3 = r".\files\midi\midi3.mid"
MIDI_PATH_4 = r".\files\midi\midi4.mid"

midi_paths = glob("./files/midi/*.mid")
midi_big_paths = glob("./files/midi_big/*.mid")

def test_filter_notes():
    me = melody_lib.MelodyExtractor(min_pitch=60, max_pitch=80)

    def cnt_out_of_pitch_range(nc):
        return len([n for n in nc.notes if me.max_pitch < n.pitch or n.pitch < me.min_pitch])

    def cnt_invalid_programs(nc):
        return len([n for n in nc.notes if me.valid_programs is not None and n.program not in me.valid_programs])

    for midi_path in midi_paths:
        pm = pretty_midi.PrettyMIDI(midi_path)
        note_container = NoteContainer.from_pretty_midi(pm)

        me.filter_notes(note_container)
        assert cnt_out_of_pitch_range(note_container) == 0
        assert cnt_invalid_programs(note_container) == 0


def magent_melody_extractor(midi_path, max_bars=100):
    data_converter = music_vae_data.OneHotMelodyConverter(
        valid_programs=music_vae_data.MEL_PROGRAMS,
        skip_polyphony=False,
        max_bars=max_bars,  # Truncate long melodies before slicing.
        slice_bars=16,

        steps_per_quarter=4)

    ns = convert_midi("", '', midi_path)
    all_melodies = []
    try:
        note_sequences = sequences_lib.split_note_sequence_on_time_changes(ns)
        for ns in note_sequences:
            # filter notes --------------------------------------------------------
            def filter(note):
                if (data_converter._valid_programs is not None and note.program not in data_converter._valid_programs):
                    return False
                return data_converter._min_pitch <= note.pitch <= data_converter._max_pitch

            notes = list(ns.notes)
            del ns.notes[:]
            ns.notes.extend([n for n in notes if filter(n)])

            # quantize sequence ---------------------------------------------------
            quantized_sequence = mm.quantize_note_sequence(ns, data_converter._steps_per_quarter)

            # extract melodies ----------------------------------------------------
            melodies, _ = data_converter._event_extractor_fn(quantized_sequence)
            all_melodies.extend(melodies)
    except:
        pass
    return all_melodies


def melody_extractor(midi_path, max_bars=100):
    melody_extractor = melody_lib.MelodyExtractor(max_bars=max_bars)

    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except (EOFError, OSError):
        return []
    note_containers = split_on_time_signature_tempo_change(NoteContainer.from_pretty_midi(pm))

    total_melodies = []
    for nc in note_containers:
        melody_extractor.filter_notes(nc)
        qnc = quantize_note_container(nc, melody_extractor.steps_per_quarter)
        melodies = melody_lib.extract_melodies(qnc, gap_bars=melody_extractor.gap_bars,
                                               max_steps_truncate=melody_extractor.max_steps_truncate,
                                               pad_end=melody_extractor.pad_end)
        total_melodies.extend(melodies)
    return total_melodies


def test_melody_extractor(midi_path, max_bars=100):
    magenta_melodies = magent_melody_extractor(midi_path, max_bars=max_bars)
    melodies = melody_extractor(midi_path, max_bars=max_bars)

    assert len(magenta_melodies) == len(melodies)

    for magenta_melody, melody in zip(magenta_melodies, melodies):
        # min_len = min(len(magenta_melody._events), len(melody.events))
        # assert(magenta_melody._events[:min_len] == melody.events[:min_len])
        assert magenta_melody._events == melody.events

        for i in range(len(melody)):
            assert magenta_melody[i:i+128]._events == melody[i:i+128].events


def test_melody_extractor_one():
    test_melody_extractor(MIDI_PATH_3)


def test_melody_extractor_many():
    for midi_path in midi_paths:
        # print(midi_path)
        test_melody_extractor(midi_path, max_bars=100)
        test_melody_extractor(midi_path, max_bars=None)

    test_melody_extractor(r".\files\midi_warning\1a2c37d1ef02797d25ac9488b3a0b871.mid", max_bars=100)
    test_melody_extractor(r".\files\midi_warning\1a7a85e46e7742a7a40a18f327a757fa.mid", max_bars=100)


if __name__ == "__main__":
    test_filter_notes()
    # test_melody_extractor_one()
    test_melody_extractor_many()
