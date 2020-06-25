import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

import symusic.musicvae.utils as utils
import symusic.musicvae.models.base as base
from symusic.musicvae.models.seq_decoder import seq_decoder_factory

# noinspection PyUnresolvedReferences
from common import ckpt_out_path

class FakeState:
    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


def assert_seq2seq_ckpt(model):
    utils.save_ckpt(ckpt_out_path, 12, model, FakeState(), FakeState())

    ckpt = torch.load(ckpt_out_path)
    m = base.Seq2Seq.load_from_ckpt(ckpt["model_kwargs"], torch.device("cpu"))
    global_step = utils.load_ckpt(ckpt_out_path, m, FakeState(), FakeState(), torch.device("cpu"))

    assert global_step == 12

    for val1, val2 in zip(model.state_dict().values(), m.state_dict().values()):
        assert torch.allclose(val1, val2)

    for kwargs1, kwargs2 in zip(ckpt["model_kwargs"]["kwargs"], m.create_ckpt()["kwargs"]):
        assert kwargs1 == kwargs2


def test_greedy_seq2seq_ckpt():
    embed_size = 12
    input_size = output_size = 91
    z_size = 33
    device = torch.device("cpu")
    encoder = base.Encoder(input_size=input_size, embed_size=embed_size, hidden_size=32, z_size=z_size, n_layers=2)
    params = {
        "output_size": output_size,
        "z_size": z_size,
        "device": device,
        "decoder_args": {
            "key": "another",
            "params": {
                "z_size": z_size,
                "embed_size": embed_size,
                "hidden_size": 64,
                "n_layers": 2,
                "dropout_prob": 0.2
            }
        }
    }
    seq_decoder = seq_decoder_factory.create("greedy", **params)
    model = base.Seq2Seq(encoder, seq_decoder, 90)
    assert_seq2seq_ckpt(model)


def test_sample_seq2seq_ckpt():
    embed_size = 12
    input_size = output_size = 91
    z_size = 33
    device = torch.device("cpu")
    encoder = base.Encoder(input_size=input_size, embed_size=embed_size, hidden_size=32, z_size=z_size, n_layers=2)
    params = {
        "output_size": output_size,
        "z_size": z_size,
        "device": device,
        "temperature": 3.0,
        "decoder_args": {
            "key": "another",
            "params": {
                "z_size": z_size,
                "embed_size": embed_size,
                "hidden_size": 64,
                "n_layers": 2,
                "dropout_prob": 0.2
            }
        }
    }
    seq_decoder = seq_decoder_factory.create("sample", **params)
    model = base.Seq2Seq(encoder, seq_decoder, 90)
    assert_seq2seq_ckpt(model)


def test_hier_seq2seq_ckpt():
    embed_size = 12
    input_size = output_size = 91
    z_size = 33
    c_size = 134
    n_subseq = 2
    device = torch.device("cpu")
    encoder = base.Encoder(input_size=input_size, embed_size=embed_size, hidden_size=32, z_size=z_size, n_layers=2)

    params = {
        "output_size": output_size,
        "z_size": z_size,
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
            "key": "greedy",
            "params": {
                "z_size": c_size,
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
    assert_seq2seq_ckpt(model)


if __name__ == "__main__":
    test_greedy_seq2seq_ckpt()
    test_sample_seq2seq_ckpt()
    test_hier_seq2seq_ckpt()
