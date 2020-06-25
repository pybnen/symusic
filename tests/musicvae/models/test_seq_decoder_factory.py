import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

import symusic.musicvae.models.base as base
import symusic.musicvae.models.seq_decoder as seq_decoder_module
from symusic.musicvae.models.seq_decoder import seq_decoder_factory

def test_factory_greedy():
    z_size = 111

    device = torch.device("cpu")
    params = {
        "decoder_args": {
            "key": "another",
            "params": {
                "embed_size": 43,
                "hidden_size": 222,
                "n_layers": 1,
                "z_size": z_size
            }
        },
        "output_size": 101,
        "z_size": z_size,
        "device": device
    }
    seq_decoder = seq_decoder_factory.create("greedy", **params)
    assert isinstance(seq_decoder, seq_decoder_module.GreedySeqDecoder)
    assert seq_decoder.z_size == z_size

    assert isinstance(seq_decoder.decoder, base.AnotherDecoder)
    assert seq_decoder.decoder.output_size == 101
    assert seq_decoder.decoder.embed_size == 43
    assert seq_decoder.decoder.hidden_size == 222
    assert seq_decoder.decoder.z_size == z_size
    assert seq_decoder.decoder.n_layers == 1
    assert seq_decoder.decoder.dropout_prob == 0.


def test_factory_sample():
    z_size = 111

    device = torch.device("cpu")
    params = {
        "decoder_args": {
            "key": "another",
            "params": {
                "embed_size": 43,
                "hidden_size": 222,
                "n_layers": 4,
                "z_size": z_size,
                "dropout_prob": 0.3
            }
        },
        "output_size": 81,
        "temperature": 0.4,
        "z_size": z_size,
        "device": device
    }
    seq_decoder = seq_decoder_factory.create("sample", **params)
    assert isinstance(seq_decoder, seq_decoder_module.GreedySeqDecoder)
    assert seq_decoder.z_size == z_size
    assert seq_decoder.temperature == 0.4

    assert isinstance(seq_decoder.decoder, base.AnotherDecoder)
    assert seq_decoder.decoder.output_size == 81
    assert seq_decoder.decoder.embed_size == 43
    assert seq_decoder.decoder.hidden_size == 222
    assert seq_decoder.decoder.z_size == z_size
    assert seq_decoder.decoder.n_layers == 4
    assert seq_decoder.decoder.dropout_prob == .3


def test_factory_hier():
    z_size = 38
    c_size = 23

    device = torch.device("cpu")
    params = {
        "conductor_args": {
            "key": "conductor",
            "params": {
                "hidden_size": 13,
                "n_layers": 3,
                "c_size": c_size
            }
        },
        "seq_decoder_args": {
            "key": "greedy",
            "params": {
                "decoder_args": {
                    "key": "another",
                    "params": {
                        "embed_size": 12,
                        "hidden_size": 12,
                        "n_layers": 2,
                        "z_size": c_size
                    }
                },
                "z_size": c_size
            }
        },
        "output_size": 91,
        "n_subsequences": 8,
        "z_size": z_size,
        "device": device
    }

    seq_decoder = seq_decoder_factory.create("hier", **params)
    assert isinstance(seq_decoder, seq_decoder_module.HierarchicalSeqDecoder)
    assert seq_decoder.n_subsequences == 8
    assert seq_decoder.z_size == z_size

    assert isinstance(seq_decoder.conductor, base.Conductor)
    assert seq_decoder.conductor.hidden_size == 13
    assert seq_decoder.conductor.c_size == c_size
    assert seq_decoder.conductor.n_layers == 3

    assert isinstance(seq_decoder.seq_decoder, seq_decoder_module.GreedySeqDecoder)
    assert seq_decoder.seq_decoder.z_size == c_size

    assert isinstance(seq_decoder.seq_decoder.decoder, base.AnotherDecoder)
    assert seq_decoder.seq_decoder.decoder.output_size == 91
    assert seq_decoder.seq_decoder.decoder.embed_size == 12
    assert seq_decoder.seq_decoder.decoder.hidden_size == 12
    assert seq_decoder.seq_decoder.decoder.z_size == c_size
    assert seq_decoder.seq_decoder.decoder.n_layers == 2
    assert seq_decoder.seq_decoder.decoder.dropout_prob == 0.


if __name__ == "__main__":
    test_factory_greedy()
    test_factory_sample()
    test_factory_hier()

