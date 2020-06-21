import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

import symusic.musicvae.utils as utils
from symusic.musicvae.models.base import Encoder, AnotherDecoder, SimpleSeqDecoder, Seq2Seq
from symusic.musicvae.models.hier import HierarchicalSeqDecoder, Conductor

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
    m = Seq2Seq.load_from_ckpt(ckpt["model_kwargs"], torch.device("cpu"))
    global_step = utils.load_ckpt(ckpt_out_path, m, FakeState(), FakeState(), torch.device("cpu"))

    assert global_step == 12

    for val1, val2 in zip(model.state_dict().values(), m.state_dict().values()):
        assert torch.allclose(val1, val2)

    for kwargs1, kwargs2 in zip(ckpt["model_kwargs"]["kwargs"], m.create_ckpt()["kwargs"]):
        assert kwargs1 == kwargs2


def test_simple_seq2seq_ckpt():
    embed_size = 12
    input_size = output_size = 91
    z_size = 33
    device = torch.device("cpu")
    encoder = Encoder(input_size=input_size, embed_size=embed_size, hidden_size=32, z_size=z_size, n_layers=2)
    decoder = AnotherDecoder(output_size=output_size, embed_size=embed_size, hidden_size=64, z_size=z_size, n_layers=2)
    seq_decoder = SimpleSeqDecoder(decoder, z_size, device)

    model = Seq2Seq(encoder, seq_decoder, 90)
    assert_seq2seq_ckpt(model)


def test_hier_seq2seq_ckpt():
    embed_size = 12
    input_size = output_size = 91
    z_size = 33
    c_size = 134
    n_subseq = 2
    device = torch.device("cpu")
    encoder = Encoder(input_size=input_size, embed_size=embed_size, hidden_size=32, z_size=z_size, n_layers=2)

    conductor = Conductor(23, c_size, 3)
    dec = AnotherDecoder(output_size=output_size, embed_size=embed_size, hidden_size=64, z_size=c_size, n_layers=2,
                         dropout_prob=0.3)
    seq_decoder = HierarchicalSeqDecoder(conductor, dec, n_subseq, z_size, device)

    model = Seq2Seq(encoder, seq_decoder, 90)
    assert_seq2seq_ckpt(model)


if __name__ == "__main__":
    test_simple_seq2seq_ckpt()
    test_hier_seq2seq_ckpt()
