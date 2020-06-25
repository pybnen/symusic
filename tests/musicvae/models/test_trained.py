import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np

import symusic.musicvae.utils as utils
import symusic.musicvae.models.trained as trained
import symusic.musicvae.models.base as base
from symusic.musicvae.models.seq_decoder import seq_decoder_factory

# from symusic.musicvae.models.trained import TrainedModel
# from symusic.musicvae.models.base import SimpleSeqDecoder, AnotherDecoder, Seq2Seq, Encoder
# from symusic.musicvae.models.hier import HierarchicalSeqDecoder, Conductor
from symusic.musicvae.datasets.melody_dataset import MapMelodyToIndex

# noinspection PyUnresolvedReferences
from common import ckpt_out_path

Z_SIZE = 33


class FakeState:
    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


def create_checkpoint(model):
    utils.save_ckpt(ckpt_out_path, 12, model, FakeState(), FakeState())


def assert_trained(model):
    batch_size = 4
    seq_len = 32

    create_checkpoint(model)

    device = torch.device("cpu")
    melody_dict = MapMelodyToIndex()
    model = trained.TrainedModel(ckpt_out_path, melody_dict, device)

    # get melody
    melodies = np.random.randint(melody_dict.min_pitch,
                                 melody_dict.max_pitch + 1,
                                 size=(batch_size, seq_len),
                                 dtype=np.int64)

    # test encode
    z, mu, logvar = model.encode(melodies)
    assert z.shape == mu.shape == logvar.shape == torch.Size([batch_size, Z_SIZE])

    # test sample
    tokens, _ = model.sample(n=10, length=32)
    assert tokens.shape == torch.Size([10, 32])

    # test decode
    tokens, _ = model.decode(z, length=32)
    assert tokens.shape == torch.Size([batch_size, 32])

    # test interpolate
    tokens, _ = model.interpolate(melodies[0], melodies[1], num_steps=12, length=32)
    assert tokens.shape == torch.Size([12, 32])


def test_greedy_trained():
    embed_size = 12
    input_size = output_size = 91
    device = torch.device("cpu")
    encoder = base.Encoder(input_size=input_size, embed_size=embed_size, hidden_size=32, z_size=Z_SIZE, n_layers=2)

    params = {
        "output_size": output_size,
        "z_size": Z_SIZE,
        "device": device,
        "decoder_args": {
            "key": "another",
            "params": {
                "z_size": Z_SIZE,
                "embed_size": embed_size,
                "hidden_size": 64,
                "n_layers": 2
            }
        }
    }
    seq_decoder = seq_decoder_factory.create("greedy", **params)
    model = base.Seq2Seq(encoder, seq_decoder, 90)
    assert_trained(model)


def test_sample_trained():
    embed_size = 12
    input_size = output_size = 91
    device = torch.device("cpu")
    encoder = base.Encoder(input_size=input_size, embed_size=embed_size, hidden_size=32, z_size=Z_SIZE, n_layers=2)

    params = {
        "output_size": output_size,
        "z_size": Z_SIZE,
        "device": device,
        "temperature": 3.0,
        "decoder_args": {
            "key": "another",
            "params": {
                "z_size": Z_SIZE,
                "embed_size": embed_size,
                "hidden_size": 64,
                "n_layers": 2
            }
        }
    }
    seq_decoder = seq_decoder_factory.create("sample", **params)
    model = base.Seq2Seq(encoder, seq_decoder, 90)
    assert_trained(model)


def test_hier_trained():
    embed_size = 12
    input_size = output_size = 91
    c_size = 134
    n_subseq = 2
    device = torch.device("cpu")
    encoder = base.Encoder(input_size=input_size, embed_size=embed_size, hidden_size=32, z_size=Z_SIZE, n_layers=2)

    params = {
        "output_size": output_size,
        "z_size": Z_SIZE,
        "n_subsequences": n_subseq,
        "device": device,
        "conductor_args": {
            "key": "conductor",
            "params": {
                "hidden_size": 23,
                "c_size": c_size,
                "n_layers": 3
            }
        },
        "seq_decoder_args": {
            "key": "sample",
            "params": {
                "z_size": c_size,
                "device": device,
                "temperature": 3.0,
                "decoder_args": {
                    "key": "another",
                    "params": {
                        "embed_size": embed_size,
                        "hidden_size": 64,
                        "n_layers": 2,
                        "dropout_prob": 0.3,
                        "z_size": c_size
                    }
                }
            }
        },
    }
    seq_decoder = seq_decoder_factory.create("hier", **params)
    model = base.Seq2Seq(encoder, seq_decoder, 90)
    assert_trained(model)

if __name__ == "__main__":
    test_greedy_trained()
    test_sample_trained()
    # test_hier_trained()