from copy import copy

import torch
# noinspection PyUnresolvedReferences
from torch.utils.data import Dataset, IterableDataset
from glob import glob
import numpy as np

from music_vae.music import settings
from music_vae.music.melody_lib import MelodyExtractor


class MelodyDataset(IterableDataset):

    def __init__(self, midi_dir,
                 min_pitch=settings.PIANO_MIN_MIDI_PITCH,
                 max_pitch=settings.PIANO_MAX_MIDI_PITCH,
                 valid_programs=settings.MELODY_PROGRAMS,
                 max_bars=100,
                 slice_bars=2,
                 steps_per_quarter=4,
                 quarters_per_bar=4,
                 train=True,
                 transforms=None,
                 max_melodies_per_sample=5):
        """
        melody_length: length of music sample in bars
        transforms: transform music
        """
        super().__init__()
        self.midi_dir = midi_dir
        self.midi_files = glob(midi_dir + "/**/*.mid", recursive=True)
        self.transforms = transforms

        self.steps_per_bar = steps_per_quarter * quarters_per_bar
        self.slice_bars = slice_bars
        self.slice_steps = self.steps_per_bar * slice_bars if slice_bars else None
        self.max_melodies_per_sample = max_melodies_per_sample
        self.train = train
        self.melody_extractor = MelodyExtractor(max_bars=max_bars, min_pitch=min_pitch, max_pitch=max_pitch,
                                                valid_programs=valid_programs,
                                                steps_per_quarter=steps_per_quarter, quarters_per_bar=quarters_per_bar)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # single-process data loading, return the full iterator
            files = self.midi_files
        else:
            # in a worker process, split workload
            def chunks(lst, n):
                """Yield n number of striped chunks from lst."""
                for i in range(0, n):
                    yield lst[i::n]

            file_chunks = list(chunks(self.midi_files, worker_info.num_workers))

            worker_id = worker_info.id
            files = file_chunks[worker_id]

        return MelodyIterator(files, self.melody_extractor, self.slice_steps, self.max_melodies_per_sample,
                              self.train, self.transforms)


class MelodyIterator(object):
    def __init__(self, midi_files, melody_extractor, slice_steps, max_melodies_per_sample, train=True, transforms=None):
        self.midi_files = midi_files
        self.file_idx = 0
        self.melody_stack = []
        self.train = train
        self.transforms = transforms

        self.melody_extractor = melody_extractor
        self.slice_steps = slice_steps
        self.max_melodies_per_sample = max_melodies_per_sample
        self.steps_per_bar = melody_extractor.steps_per_bar

    def reset(self):
        self.file_idx = 0
        self.melody_stack = []

    def __iter__(self):
        return self

    def extract_melodies(self, midi_path):
        melodies = self.melody_extractor.extract_melodies(midi_path)

        if self.slice_steps is not None:
            sliced_melodies = []
            for melodie in melodies:
                for i in range(self.slice_steps, len(melodie) + 1, self.steps_per_bar):
                    sliced_melodies.append(melodie[i - self.slice_steps: i])
        else:
            sliced_melodies = melodies

        unique_event_tuples = list(set(tuple(melodie) for melodie in sliced_melodies))
        unique_event_tuples = maybe_sample_items(unique_event_tuples,
                                                 self.max_melodies_per_sample,
                                                 self.train)
        return [np.array(list(t), dtype=np.int64) for t in unique_event_tuples]

    def __next__(self):
        if len(self.melody_stack) == 0:
            while len(self.melody_stack) == 0 and self.file_idx < len(self.midi_files):
                # get next file
                file = self.midi_files[self.file_idx]

                # try to extract melodies from file and put melodies on stack
                melodies = self.extract_melodies(file)
                self.melody_stack.extend(melodies)

                if len(melodies) == 0:
                    # if midi doesn't contain any melodies remove from array
                    # NOTE: this doesn't work as intended, because a new MelodyIterator is created every time
                    #   the __iter__ method of MelodyDataset is called.
                    del self.midi_files[self.file_idx]
                else:
                    self.file_idx += 1

            if len(self.melody_stack) == 0:
                # no more melodies on stack and files exhausted
                raise StopIteration

        melody = self.melody_stack.pop()
        if self.transforms is not None:
            melody = self.transforms(melody)
        return melody


def maybe_sample_items(seq, sample_size, randomize):
    if not sample_size or len(seq) <= sample_size:
        return seq
    if randomize:
        indices = set(np.random.choice(len(seq), size=sample_size, replace=False))
        return [seq[i] for i in indices]
    else:
        return seq[:sample_size]


class MapMelodyToIndex:

    def __init__(self,
                 min_pitch=settings.PIANO_MIN_MIDI_PITCH,
                 max_pitch=settings.PIANO_MAX_MIDI_PITCH,
                 num_special_events=settings.MELODY_NUM_SPECIAL_EVENTS,
                 has_sos_token=True):

        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.num_special_events = num_special_events
        self.has_sos_token = has_sos_token

    def __call__(self, melody):
        """Maps melody event to an index.

        Special events are mapped from 0 to num_special_events - 1
        the range [min_pitch ... max_pitch] is mapped
        to [num_special_events ... max_pitch - min_pitch + num_special_events]
        """
        is_special_event = np.logical_and(melody >= -self.num_special_events, melody < settings.MIN_MIDI_PITCH)
        is_in_pitch_range = np.logical_and(melody >= self.min_pitch, melody <= self.max_pitch)
        assert np.all(np.logical_or(is_special_event, is_in_pitch_range))

        indices = copy(melody)
        # set min_pitch to idx 0, the next pitch to 1 and so on...
        indices[indices >= 0] -= self.min_pitch
        # shift all pitch events to the right for the number of special events
        indices += self.num_special_events

        return indices

    def sequence_to_melody(self, seq):
        melody = copy(seq)
        melody -= self.num_special_events
        melody[melody >= 0] += self.min_pitch
        return melody.detach().cpu().numpy()

    def dict_size(self):
        """Returns the number if indexes

        if there is a sos token the number increases by one."""
        return self.num_special_events + self.max_pitch - self.min_pitch + 1 + int(self.has_sos_token)

    def get_sos_token(self):
        if self.has_sos_token:
            return self.dict_size() - 1
        return None

class FixedLengthMelodyDataset(Dataset):
    def __init__(self, **kwargs):
        """
        transforms: transform music
        """
        super().__init__()
        self.melody_dir = kwargs.get("melody_dir", None)
        self.transforms = kwargs.get("transforms", None)
        self.melody_files = glob(self.melody_dir + "/**/*.npy", recursive=True)

    def __len__(self):
        return len(self.melody_files)

    def __getitem__(self, idx):
        melody = np.load(self.melody_files[idx]).astype(np.int64)
        if self.transforms is not None:
            melody = self.transforms(melody)
        return melody


class MelodyEncode:

    def __init__(self, n_classes, num_special_events=2):
        self.n_classes = n_classes
        self.num_special_events = num_special_events

    def __call__(self, sequence):
        # TODO as of now expects integer values from [-2, x],
        #  but the pitch range could also be restricted, e.g. [-2, -1] union [22, 90]
        sequence = sequence + self.num_special_events

        seq_length = sequence.shape[0]
        one_hot = np.zeros((seq_length, self.n_classes), dtype=np.float32)
        one_hot[np.arange(seq_length), sequence.astype(np.int32)] = 1.0
        return one_hot


class MelodyDecode:
    def __call__(self, encoded_melody):
        return encoded_melody - 2


