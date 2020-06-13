import time
from glob import glob

from torch.utils.data import DataLoader
from torchvision import transforms

from music_vae.music.melody_lib import MelodyExtractor
from music_vae.datasets.melody_dataset import MelodyIterator
from music_vae.datasets.melody_dataset import MelodyDataset, MapMelodyToIndex

MIDI_DIR = r"C:\Users\yggdrasil\Studium Informatik\12Semester\Project\data\lmd_full_samples\midi"

midi_paths = glob("./files/midi/*.mid")
midi_big_paths = glob("./files/midi_big/*.mid")


def test_melody_iterator():
    extractor = MelodyExtractor(slice_bars=8)
    melody_iter = MelodyIterator(midi_paths, extractor, max_melodies_per_sample=5, train=False)
    for mel in melody_iter:
        assert len(mel) == 8*16


def test_melody_dataset():
    ds = MelodyDataset(midi_dir=MIDI_DIR, slice_bars=16, train=False, transforms=MapMelodyToIndex(has_sos_token=True))
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
    # test_melody_iterator()
    test_melody_dataset()
