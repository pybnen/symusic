from glob import glob
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from torch.utils.data import DataLoader

from symusic.music.melody_lib import MelodyExtractor
from symusic.musicvae.datasets.melody_dataset import MelodyIterator, MelodyDataset, MapMelodyToIndex

# noinspection PyUnresolvedReferences
from common import midi_paths, MIDI_DIR, MIDI_PATH_1, MIDI_PATH_2


def assert_iter(melody_iter, slice_steps, should_be_equal):
    old_mels = None
    for i in range(3):
        melody_iter.reset()
        mels = np.stack([mel for mel in melody_iter])
        assert mels.shape[-1] == slice_steps
        if old_mels is not None:
            if should_be_equal:
                assert np.all(old_mels == mels)
            else:
                assert not np.all(old_mels == mels)
        old_mels = mels


def test_melody_iterator():
    slice_bars = 8
    slice_steps = 16 * slice_bars
    extractor = MelodyExtractor()

    # assert melodies are different on train = True
    melody_iter = MelodyIterator(midi_paths, extractor, slice_steps=slice_steps, max_melodies_per_sample=5, train=True)
    assert_iter(melody_iter, slice_steps, False)

    # assert melodies are the same on train = False
    melody_iter = MelodyIterator(midi_paths, extractor, slice_steps=slice_steps, max_melodies_per_sample=5, train=False)
    assert_iter(melody_iter, slice_steps, True)

    # assert melodies are the same if max_melodies_per_sample is None (nothing there to sample)
    melody_iter = MelodyIterator(midi_paths, extractor, slice_steps=slice_steps, max_melodies_per_sample=None, train=True)
    assert_iter(melody_iter, slice_steps, True)


def test_melody_iterator_no_melodies():
    """Test if the removal of files that contain no valid
    melodies works."""
    slice_bars = 8
    slice_steps = 16 * slice_bars
    extractor = MelodyExtractor()

    midi_files = [MIDI_PATH_2, MIDI_PATH_1] # midi2 contains melodies, midi1 not
    melody_iter = MelodyIterator(midi_files, extractor, slice_steps=slice_steps, max_melodies_per_sample=None, train=False)
    assert len(melody_iter.midi_files) == 2
    assert_iter(melody_iter, slice_steps, True)
    assert len(melody_iter.midi_files) == 1

    midi_files = [MIDI_PATH_2, MIDI_PATH_1, MIDI_PATH_2] # midi2 contains melodies, midi1 not
    melody_iter = MelodyIterator(midi_files, extractor, slice_steps=slice_steps, max_melodies_per_sample=5, train=False)
    assert len(melody_iter.midi_files) == 3
    assert_iter(melody_iter, slice_steps, True)
    assert len(melody_iter.midi_files) == 2


def test_melody_dataset():
    ds = MelodyDataset(midi_dir=MIDI_DIR, slice_bars=16, train=False, transforms=MapMelodyToIndex(has_sos_token=True))
    assert len(ds.midi_files) == 150
    print(f"Use dataset from \"{ds.midi_dir}\", it contains {len(ds.midi_files)} files.")

    print("Dataset")
    print("-------")
    for i, data in enumerate(ds):
        print(i, data)
        if i > 4:
            break


    print("\nData Loader")
    print("-----------")
    dl = DataLoader(ds, batch_size=16, num_workers=2)
    for i, batch in enumerate(dl):
        print(batch.shape)
        if i > 2:
            break


if __name__ == "__main__":
    test_melody_iterator_no_melodies()
    test_melody_iterator()
    test_melody_dataset()
