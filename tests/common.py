from glob import glob
import os

test_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/tests/"))

MIDI_PATH_1 = os.path.join(test_data_dir, "midi/midi1.mid")
MIDI_PATH_2 = os.path.join(test_data_dir, "midi/midi2.mid")
MIDI_PATH_3 = os.path.join(test_data_dir, "midi/midi3.mid")
MIDI_PATH_4 = os.path.join(test_data_dir, "midi/midi3.mid")

MIDI_PATH_5 = os.path.join(test_data_dir, "midi/3a0eee9e3a8e59524c36e6f8e4a0c698.mid")

midi_paths = glob(os.path.join(test_data_dir, "midi/*.mid"))
assert len(midi_paths) == 14

midi_big_paths = glob(os.path.join(test_data_dir, "midi_big/*.mid"))
assert len(midi_big_paths) == 118

midi_warning_paths = glob(os.path.join(test_data_dir, "midi_warning/*.mid"))
assert len(midi_warning_paths) == 2

midi1_target_steps_path = os.path.join(test_data_dir, "targets/midi1_target_steps.pkl")

MIDI_DIR = os.path.join(test_data_dir, "midi_samples")

ckpt_out_path = os.path.join(test_data_dir, "ckpt.pth")