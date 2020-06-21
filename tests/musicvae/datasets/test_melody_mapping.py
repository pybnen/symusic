import numpy as np

from symusic.musicvae.datasets.melody_dataset import MapMelodyToIndex


def get_random_melody(length, min_pitch, max_pitch, n_special):
    # sample from min_pitch to max_pitch
    melody = np.random.randint(min_pitch, max_pitch, size=length)
    # set some special events
    for special in range(-n_special, 0):
        melody[np.random.choice(np.arange(length), int(length * 0.3), replace=False)] = special
    return melody


def test_mel_to_idx():
    mel_to_idx = MapMelodyToIndex(has_sos_token=True)
    assert mel_to_idx.dict_size() == 91
    assert mel_to_idx.get_sos_token() == 90

    mel = np.array([mel_to_idx.min_pitch, mel_to_idx.max_pitch, -mel_to_idx.num_special_events])
    indices = mel_to_idx(mel)
    assert np.all(indices == [mel_to_idx.num_special_events, mel_to_idx.dict_size() - 2, 0])
    assert np.all(indices == [mel_to_idx.num_special_events,
                              mel_to_idx.max_pitch - mel_to_idx.min_pitch + mel_to_idx.num_special_events,
                              0])
    mel_to_idx = MapMelodyToIndex(has_sos_token=False)
    assert mel_to_idx.dict_size() == 90
    assert np.all(indices == [mel_to_idx.num_special_events, mel_to_idx.dict_size() - 1, 0])
    assert np.all(indices == [mel_to_idx.num_special_events,
                              mel_to_idx.max_pitch - mel_to_idx.min_pitch + mel_to_idx.num_special_events,
                              0])

    n_specials = 2
    for _ in range(100):
        min_pitch = np.random.randint(0, 127)
        max_pitch = np.random.randint(min_pitch + 1, 128)
        assert min_pitch <= max_pitch
        mel_to_idx = MapMelodyToIndex(min_pitch, max_pitch, n_specials, has_sos_token=np.random.choice([True, False]))

        dict_size = max_pitch - min_pitch + 1 + n_specials + int(mel_to_idx.has_sos_token)
        assert mel_to_idx.dict_size() == dict_size
        if mel_to_idx.has_sos_token:
            assert mel_to_idx.get_sos_token() == dict_size -1

        for _ in range(100):
            mel = get_random_melody(128, mel_to_idx.min_pitch, mel_to_idx.max_pitch, mel_to_idx.num_special_events)
            indices = mel_to_idx(mel)
            if mel_to_idx.has_sos_token:
                assert np.all(np.logical_and(indices >= 0, indices < mel_to_idx.dict_size() - 1))
            else:
                assert np.all(np.logical_and(indices >= 0, indices < mel_to_idx.dict_size()))


if __name__ == "__main__":
    test_mel_to_idx()