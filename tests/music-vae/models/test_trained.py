import torch
import numpy as np

from music_vae import utils
from music_vae.models.trained import TrainedModel
from music_vae.models.base import SimpleSeqDecoder, AnotherDecoder, Seq2Seq, Encoder
from music_vae.models.hier import HierarchicalSeqDecoder, Conductor
from music_vae.datasets.melody_dataset import MapMelodyToIndex

CKPT_PATH = "ckpt.pth"

Z_SIZE = 33

class FakeState:
    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


def create_checkpoint(model):
    utils.save_ckpt(CKPT_PATH, 12, model, FakeState(), FakeState())


def assert_trained(model):
    batch_size = 4
    seq_len = 32

    create_checkpoint(model)

    device = torch.device("cpu")
    melody_dict = MapMelodyToIndex()
    model = TrainedModel(CKPT_PATH, melody_dict, device)

    # get melody
    melodies = np.random.randint(melody_dict.min_pitch,
                                 melody_dict.max_pitch + 1,
                                 size=(batch_size, seq_len),
                                 dtype=np.int64)

    # test encode
    z, mu, logvar = model.encode(melodies)
    assert z.shape == mu.shape == logvar.shape == torch.Size([batch_size, Z_SIZE])

    # test sample
    outputs = model.sample(n=10, length=32)
    assert outputs.shape == torch.Size([10, 32])

    # test decode
    outputs = model.decode(z, length=32)
    assert outputs.shape == torch.Size([batch_size, 32])

    # test interpolate
    outputs = model.interpolate(melodies[0], melodies[1], num_steps=12, length=32)
    assert outputs.shape == torch.Size([12, 32])


def test_simple_trained():
    embed_size = 12
    input_size = output_size = 91
    device = torch.device("cpu")
    encoder = Encoder(input_size=input_size, embed_size=embed_size, hidden_size=32, z_size=Z_SIZE, n_layers=2)
    decoder = AnotherDecoder(output_size=output_size, embed_size=embed_size, hidden_size=64, z_size=Z_SIZE, n_layers=2)
    seq_decoder = SimpleSeqDecoder(decoder, Z_SIZE, device)
    model = Seq2Seq(encoder, seq_decoder, 90)

    assert_trained(model)


def test_hier_trained():
    embed_size = 12
    input_size = output_size = 91
    c_size = 134
    n_subseq = 2
    device = torch.device("cpu")
    encoder = Encoder(input_size=input_size, embed_size=embed_size, hidden_size=32, z_size=Z_SIZE, n_layers=2)

    conductor = Conductor(23, c_size, 3)
    dec = AnotherDecoder(output_size=output_size, embed_size=embed_size, hidden_size=64, z_size=c_size, n_layers=2,
                         dropout_prob=0.3)
    seq_decoder = HierarchicalSeqDecoder(conductor, dec, n_subseq, Z_SIZE, device)

    model = Seq2Seq(encoder, seq_decoder, 90)
    assert_trained(model)

if __name__ == "__main__":
    test_simple_trained()
    test_hier_trained()