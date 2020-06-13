import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from music_vae.models.base import SimpleSeqDecoder, SimpleDecoder, AnotherDecoder
from music_vae.models.hier import HierarchicalSeqDecoder, Conductor


class FakeDecoder(nn.Module):
    """Fake decoder will return one-hot encoded input and keeps
    track of input"""

    def __init__(self, output_size):
        super().__init__()
        self.hidden_size = 12
        self.n_layers = 2
        self.output_size = output_size
        self.inputs = []
        self.one_hot_m = torch.eye(self.output_size)

        self.embedding = nn.Embedding(output_size, output_size)
        self.embedding.weight.data = torch.eye(output_size)

    def forward(self, input: torch.tensor, state, unused2):
        batch_size = input.shape[0]
        self.inputs.append(input)
        return self.embedding(input), state

    def input(self):
        return torch.stack(self.inputs, dim=1)

    def reset(self):
        self.inputs = []


def assert_input_order(seq_decoder, trg, z):
    if hasattr(seq_decoder, "decoder"):
        decoder = seq_decoder.decoder
    else:
        decoder = seq_decoder.seq_decoder.decoder
    seq_len = trg.shape[1]

    # tf 1.0 no initial input, thus decoder should always see input from target
    decoder.reset()
    outputs = seq_decoder(trg, z, teacher_forcing_ratio=1.)
    assert torch.all(trg == decoder.input())
    assert seq_decoder.sampling_rate == 0.0

    # tf 0.0 no initial input, because no initial input was given, the decoder should only see the first elem of target
    decoder.reset()
    outputs = seq_decoder(trg, z, teacher_forcing_ratio=0.)
    first_input = trg[:, :1].repeat(1, seq_len)
    assert torch.all(first_input == decoder.input())
    assert torch.any(trg != decoder.input())
    assert seq_decoder.sampling_rate == (seq_len - 1) / seq_len

    # ft 1.0 with initial input, initial input should be ignored
    decoder.reset()
    initial_input = trg[:, 0] + 1
    outputs = seq_decoder(trg, z, teacher_forcing_ratio=1., initial_input=initial_input)
    assert torch.all(trg == decoder.input())
    assert seq_decoder.sampling_rate == 0.0

    # ft 0.0 with initial input, initial input should be used instead of target
    decoder.reset()
    initial_input = trg[:, 0] + 1
    outputs = seq_decoder(trg, z, teacher_forcing_ratio=0., initial_input=initial_input)
    assert torch.all(initial_input.unsqueeze(dim=1).repeat(1, seq_len) == decoder.input())
    assert torch.any(trg != decoder.input())
    assert seq_decoder.sampling_rate == 1.0


def test_simple_seq_decoder_input():
    z_size = 2
    batch_size = 1
    seq_len = 10
    output_size = 90

    device = torch.device("cpu")
    decoder = FakeDecoder(output_size)
    seq_decoder = SimpleSeqDecoder(decoder, z_size=z_size, device=device)
    z = torch.randn((batch_size, z_size))

    for _ in range(100):
        trg = torch.randint(0, 50, size=(batch_size, seq_len))
        assert_input_order(seq_decoder, trg, z)


def test_hier_seq_decoder_input():
    z_size = 2
    batch_size = 1
    seq_len = 32
    output_size = 90

    device = torch.device("cpu")
    conductor = Conductor(hidden_size=1, c_size=1, n_layers=1)
    decoder = FakeDecoder(output_size)

    seq_decoder = HierarchicalSeqDecoder(conductor, decoder,
                                         n_subsequences=2,
                                         z_size=z_size, device=device)
    z = torch.randn((batch_size, z_size))

    for _ in range(100):
        trg = torch.randint(0, 50, size=(batch_size, seq_len))
        assert_input_order(seq_decoder, trg, z)


def assert_seq_decoder_output_shape(seq_decoder, z_size):
    batch_size = 16
    seq_len = 32
    input_size = 90
    output_size = seq_decoder.output_size
    z = torch.randn((batch_size, z_size))

    for i in range(150):
        input = Categorical(probs=torch.ones((batch_size, seq_len, input_size))).sample()

        outputs = seq_decoder(input, z, teacher_forcing_ratio=1.0)
        assert outputs.shape == torch.Size([batch_size, seq_len, output_size])

        outputs = seq_decoder(input, z, teacher_forcing_ratio=0.0)
        assert outputs.shape == torch.Size([batch_size, seq_len, output_size])

    # check if checkpoint still works
    device = torch.device("cpu")
    ckpt = seq_decoder.create_ckpt()
    new_seq_decoder = seq_decoder.__class__.load_from_ckpt(ckpt, device, state_dict=seq_decoder.state_dict())
    for val1, val2 in zip(seq_decoder.state_dict().values(), new_seq_decoder.state_dict().values()):
        assert torch.allclose(val1, val2)

    for kwargs1, kwargs2 in zip(ckpt["kwargs"], new_seq_decoder.create_ckpt()["kwargs"]):
        assert kwargs1 == kwargs2


def test_simple_seq_decoder_output_shape():
    z_size = 9
    hidden_size = 10
    embed_size = 12
    output_size = 90
    n_layers = 2

    device = torch.device("cpu")
    dec = SimpleDecoder(output_size=output_size, embed_size=embed_size, hidden_size=hidden_size,
                         n_layers=n_layers, dropout_prob=0.2)

    seq_decoder = SimpleSeqDecoder(dec, z_size, device)
    assert_seq_decoder_output_shape(seq_decoder, z_size)

    dec = AnotherDecoder(output_size=output_size, embed_size=embed_size, hidden_size=hidden_size,
                          z_size=z_size, n_layers=n_layers, dropout_prob=0.3)
    seq_decoder = SimpleSeqDecoder(dec, z_size, device)
    assert_seq_decoder_output_shape(seq_decoder, z_size)


def test_hier_seq_decoder_output_shape():
    z_size = 9
    hidden_size = 10
    embed_size = 12
    output_size = 90
    n_layers = 2
    n_subseq = 2
    c_size = 44

    device = torch.device("cpu")
    conductor = Conductor(23, c_size, n_layers)
    dec = SimpleDecoder(output_size=output_size, embed_size=embed_size, hidden_size=hidden_size,
                        n_layers=n_layers, dropout_prob=0.2)
    seq_decoder = HierarchicalSeqDecoder(conductor, dec, n_subseq, z_size, device)
    assert_seq_decoder_output_shape(seq_decoder, z_size)

    dec = AnotherDecoder(output_size=output_size, embed_size=embed_size, hidden_size=hidden_size,
                         z_size=c_size, n_layers=n_layers, dropout_prob=0.3)
    seq_decoder = HierarchicalSeqDecoder(conductor, dec, n_subseq, z_size, device)
    assert_seq_decoder_output_shape(seq_decoder, z_size)


if __name__ == "__main__":
    test_simple_seq_decoder_input()
    test_simple_seq_decoder_output_shape()

    test_hier_seq_decoder_input()
    test_hier_seq_decoder_output_shape()