import pretty_midi
import uuid

from symusic.music.settings import MELODY_PROGRAMS
import symusic.music.melody_lib as melody_lib
from symusic.gui.globals import STEPS_PER_BAR, MELODY_LENGTH, audio_download_url, audio_filesystem_dir

import matplotlib
import matplotlib.pyplot as plt
import pypianoroll as pp
from io import BytesIO
import base64
import numpy as np
from midi2audio import FluidSynth

matplotlib.use("Agg")


def midi_to_track_options(midi_path):
    pm = pretty_midi.PrettyMIDI(midi_path)

    def instrument_to_option(idx, instrument):
        label = "{} - {} ({})".format(idx,
                                      pretty_midi.program_to_instrument_name(instrument.program),
                                      instrument.program)
        return dict(label=label, value=idx)

    return [instrument_to_option(idx, instrument) for (idx, instrument) in enumerate(pm.instruments)
            if not instrument.is_drum and instrument.program in MELODY_PROGRAMS]


def calc_max_start_bar(midi_path, track_idx):
    melody_extractor = melody_lib.MelodyExtractor(max_bars=None,
                                                  valid_programs=MELODY_PROGRAMS,
                                                  gap_bars=float("inf"))
    melodies = melody_extractor.extract_melodies(midi_path)
    melody = None
    for m in melodies:
        if m.instrument == track_idx:
            melody = m
            break

    if melody is None:
        return -1

    pianoroll = melody_lib.melody_to_pianoroll(melody)
    return max(int(len(pianoroll) // STEPS_PER_BAR - MELODY_LENGTH / STEPS_PER_BAR), 0)


def fig_to_base64(fig):
    buf = BytesIO()  # in-memory files
    plt.savefig(buf, format="png")  # save to the above file object
    data = base64.b64encode(buf.getbuffer()).decode("utf8")  # encode to html elements
    plt.close()
    return "data:image/png;base64,{}".format(data)


def midi_to_melody(midi_path, track_idx, start_bar=None):
    melody_extractor = melody_lib.MelodyExtractor(max_bars=None,
                                                  valid_programs=MELODY_PROGRAMS,
                                                  gap_bars=float("inf"))
    melodies = melody_extractor.extract_melodies(midi_path)
    melody = None
    for m in melodies:
        if m.instrument == track_idx:
            melody = m
            break

    if melody is None:
        return None, None

    if start_bar is not None:
        start_step = start_bar * STEPS_PER_BAR
        melody = melody[start_step:start_step + MELODY_LENGTH]

    return np.array(melody, dtype=np.int64), melody.program


def melody_to_graph(melody, figsize=None, start_bar=0):
    pianoroll = melody_lib.melody_to_pianoroll(melody)

    figsize = figsize or (10, 7)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    pp.plot_pianoroll(ax, pianoroll)
    ticks = np.arange(0, pianoroll.shape[0], STEPS_PER_BAR * 2)

    ax.set_xticks(ticks)
    ax.set_xticklabels(np.arange(start_bar, start_bar + len(ticks) * 2, 2))
    # ax.set_xticklabels(np.arange(0, len(ticks)*2, 2))
    ax.grid(True, axis='x')

    plt.tight_layout()

    return fig_to_base64(fig)


def melody_to_audio(melody, midi_program=0):
    pm = melody_lib.melody_to_midi(melody, program=midi_program)

    unique_filename = str(uuid.uuid4())
    midi_path = audio_filesystem_dir + unique_filename + ".mid"
    audi_path = audio_filesystem_dir + unique_filename + ".wav"

    pm.write(midi_path)
    FluidSynth().midi_to_audio(midi_path, audi_path)

    return audio_download_url + unique_filename + ".wav"

