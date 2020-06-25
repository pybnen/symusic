import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

import symusic.musicvae.models.base as base
import symusic.musicvae.models.seq_decoder as seq_decoder_module
from symusic.musicvae.models.seq_decoder import seq_decoder_factory


# define a fake decoder and add to decoder factory ------------------------------------------------
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
        # return log of one-hot encoding of input
        return torch.log(self.embedding(input)), state

    def input(self):
        return torch.stack(self.inputs, dim=1)

    def reset(self):
        self.inputs = []


def build_fake_decoder(output_size, **_ignore):
    return FakeDecoder(output_size)


base.decoder_factory.register_builder("fake", build_fake_decoder)


# start with tests --------------------------------------------------------------------------------
def assert_input_order(seq_decoder, trg, z):
    if hasattr(seq_decoder, "decoder"):
        # simple/sample seq decoder
        decoder = seq_decoder.decoder
    else:
        # hier seq decoder
        decoder = seq_decoder.seq_decoder.decoder
    seq_len = trg.shape[1]

    # tf 1.0 no initial input, thus decoder should always see input from target
    decoder.reset()
    outputs = seq_decoder(z, trg=trg, teacher_forcing_ratio=1.)
    assert torch.all(trg == decoder.input())
    assert seq_decoder.sampling_rate == 0.0

    # tf 0.0 no initial input, because no initial input was given, the decoder should only see the first elem of target
    decoder.reset()
    outputs = seq_decoder(z, trg=trg, teacher_forcing_ratio=0.)
    first_input = trg[:, :1].repeat(1, seq_len)
    assert torch.all(first_input == decoder.input())
    assert torch.any(trg != decoder.input())
    assert seq_decoder.sampling_rate == (seq_len - 1) / seq_len

    # ft 1.0 with initial input, initial input should be ignored
    decoder.reset()
    initial_input = trg[:, 0] + 1
    outputs = seq_decoder(z, trg=trg, teacher_forcing_ratio=1., initial_input=initial_input)
    assert torch.all(trg == decoder.input())
    assert seq_decoder.sampling_rate == 0.0

    # ft 0.0 with initial input, initial input should be used instead of target
    decoder.reset()
    initial_input = trg[:, 0] + 1
    outputs = seq_decoder(z, trg=trg, teacher_forcing_ratio=0., initial_input=initial_input)
    assert torch.all(initial_input.unsqueeze(dim=1).repeat(1, seq_len) == decoder.input())
    assert torch.any(trg != decoder.input())
    assert seq_decoder.sampling_rate == 1.0

    # ft 0.0 no target sequence
    decoder.reset()
    initial_input = trg[:, 0] + 1
    outputs = seq_decoder(z, length=seq_len, teacher_forcing_ratio=0., initial_input=initial_input)
    assert torch.all(initial_input.unsqueeze(dim=1).repeat(1, seq_len) == decoder.input())
    assert torch.any(trg != decoder.input())
    assert seq_decoder.sampling_rate == 1.0


def test_greedy_seq_decoder_input():
    z_size = 2
    batch_size = 1
    seq_len = 10
    output_size = 90

    device = torch.device("cpu")
    params = {
        "output_size": output_size,
        "z_size": z_size,
        "device": device,
        "decoder_args": {
            "key": "fake",
            "params": {}
        }
    }
    seq_decoder = seq_decoder_factory.create("greedy", **params)
    assert isinstance(seq_decoder, seq_decoder_module.GreedySeqDecoder)

    z = torch.randn((batch_size, z_size))
    for _ in range(100):
        trg = torch.randint(0, 50, size=(batch_size, seq_len))
        assert_input_order(seq_decoder, trg, z)


def test_sample_seq_decoder_input():
    z_size = 2
    batch_size = 1
    seq_len = 10
    output_size = 90

    device = torch.device("cpu")
    params = {
        "output_size": output_size,
        "z_size": z_size,
        "device": device,
        "temperature": 0.5,
        "decoder_args": {
            "key": "fake",
            "params": {}
        }
    }
    seq_decoder = seq_decoder_factory.create("sample", **params)
    assert isinstance(seq_decoder, seq_decoder_module.SampleSeqDecoder)

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
    params = {
        "output_size": output_size,
        "z_size": z_size,
        "n_subsequences": 2,
        "device": device,
        "conductor_args": {
            "key": "conductor",
            "params": {
                "hidden_size": 1,
                "c_size": 1,
                "n_layers": 1
            }
        },
        "seq_decoder_args": {
            "key": "greedy",
            "params": {
                "z_size": 1,
                "device": device,
                "decoder_args": {
                    "key": "fake",
                    "params": {}
                }
            }
        },
    }
    seq_decoder = seq_decoder_factory.create("hier", **params)
    assert isinstance(seq_decoder, seq_decoder_module.HierarchicalSeqDecoder)

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

    for i in range(25):
        input = Categorical(probs=torch.ones((batch_size, seq_len, input_size))).sample()

        outputs = seq_decoder(z, trg=input, teacher_forcing_ratio=1.0)
        assert outputs.shape == torch.Size([batch_size, seq_len, output_size])
        # TODO this is only correct for greedy sampling
        assert torch.all(outputs.argmax(dim=-1) == seq_decoder.output_tokens)

        outputs = seq_decoder(z, trg=input, teacher_forcing_ratio=0.0)
        assert outputs.shape == torch.Size([batch_size, seq_len, output_size])
        # TODO this is only correct for greedy sampling
        assert torch.all(outputs.argmax(dim=-1) == seq_decoder.output_tokens)

        # no target sequence
        outputs = seq_decoder(z, length=128, teacher_forcing_ratio=0.0,
                              initial_input=torch.tensor(0).repeat(batch_size))
        assert outputs.shape == torch.Size([batch_size, 128, output_size])
        # TODO this is only correct for greedy sampling
        assert torch.all(outputs.argmax(dim=-1) == seq_decoder.output_tokens)

    # check if checkpoint still works
    device = torch.device("cpu")
    ckpt = seq_decoder.create_ckpt()
    new_seq_decoder = seq_decoder.__class__.load_from_ckpt(ckpt, device, state_dict=seq_decoder.state_dict())
    for val1, val2 in zip(seq_decoder.state_dict().values(), new_seq_decoder.state_dict().values()):
        assert torch.allclose(val1, val2)

    for kwargs1, kwargs2 in zip(ckpt["kwargs"], new_seq_decoder.create_ckpt()["kwargs"]):
        assert kwargs1 == kwargs2


def test_greedy_seq_decoder_output_shape():
    z_size = 9
    hidden_size = 10
    embed_size = 12
    output_size = 90
    n_layers = 2

    device = torch.device("cpu")
    params = {
        "output_size": output_size,
        "z_size": z_size,
        "device": device,
        "decoder_args": {
            "key": "simple",
            "params": {
                "embed_size": embed_size,
                "hidden_size": hidden_size,
                "n_layers": n_layers,
                "dropout_prob": 0.2
            }
        }
    }
    seq_decoder = seq_decoder_factory.create("greedy", **params)
    assert isinstance(seq_decoder, seq_decoder_module.GreedySeqDecoder)
    assert isinstance(seq_decoder.decoder, base.SimpleDecoder)
    assert_seq_decoder_output_shape(seq_decoder, z_size)

    params["decoder_args"] = {
        "key": "another",
        "params": {
            "embed_size": embed_size,
            "hidden_size": hidden_size,
            "n_layers": n_layers,
            "dropout_prob": 0.3,
            "z_size": z_size
        }
    }
    seq_decoder = seq_decoder_factory.create("greedy", **params)
    assert isinstance(seq_decoder.decoder, base.AnotherDecoder)
    assert_seq_decoder_output_shape(seq_decoder, z_size)


def test_sample_seq_decoder_output_shape():
    z_size = 9
    hidden_size = 10
    embed_size = 12
    output_size = 90
    n_layers = 2

    device = torch.device("cpu")
    params = {
        "output_size": output_size,
        "z_size": z_size,
        "device": device,
        "temperature": 1.e-8,
        "decoder_args": {
            "key": "simple",
            "params": {
                "embed_size": embed_size,
                "hidden_size": hidden_size,
                "n_layers": n_layers,
                "dropout_prob": 0.2
            }
        }
    }
    seq_decoder = seq_decoder_factory.create("sample", **params)
    assert isinstance(seq_decoder, seq_decoder_module.SampleSeqDecoder)
    assert isinstance(seq_decoder.decoder, base.SimpleDecoder)
    assert_seq_decoder_output_shape(seq_decoder, z_size)

    params["decoder_args"] = {
        "key": "another",
        "params": {
            "embed_size": embed_size,
            "hidden_size": hidden_size,
            "n_layers": n_layers,
            "dropout_prob": 0.3,
            "z_size": z_size
        }
    }
    seq_decoder = seq_decoder_factory.create("sample", **params)
    assert isinstance(seq_decoder.decoder, base.AnotherDecoder)
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
                "n_layers": n_layers
            }
        },
        "seq_decoder_args": {
            "key": "greedy",
            "params": {
                "z_size": c_size,
                "decoder_args": {
                   "key": "simple",
                    "params": {
                        "embed_size": embed_size,
                        "hidden_size": hidden_size,
                        "n_layers": n_layers,
                        "dropout_prob": 0.2
                    }
                }
            }
        },
    }

    seq_decoder = seq_decoder_factory.create("hier", **params)
    assert isinstance(seq_decoder, seq_decoder_module.HierarchicalSeqDecoder)
    assert isinstance(seq_decoder.seq_decoder, seq_decoder_module.GreedySeqDecoder)
    assert isinstance(seq_decoder.seq_decoder.decoder, base.SimpleDecoder)
    assert_seq_decoder_output_shape(seq_decoder, z_size)

    params["seq_decoder_args"]["params"]["decoder_args"] = {
        "key": "another",
        "params": {
            "embed_size": embed_size,
            "hidden_size": hidden_size,
            "n_layers": n_layers,
            "dropout_prob": 0.3,
            "z_size": c_size
        }
    }

    seq_decoder = seq_decoder_factory.create("hier", **params)
    assert isinstance(seq_decoder, seq_decoder_module.HierarchicalSeqDecoder)
    assert isinstance(seq_decoder.seq_decoder, seq_decoder_module.GreedySeqDecoder)
    assert isinstance(seq_decoder.seq_decoder.decoder, base.AnotherDecoder)
    assert_seq_decoder_output_shape(seq_decoder, z_size)


if __name__ == "__main__":
    test_greedy_seq_decoder_input()
    test_greedy_seq_decoder_output_shape()

    test_sample_seq_decoder_input()
    test_sample_seq_decoder_output_shape()

    test_hier_seq_decoder_input()
    test_hier_seq_decoder_output_shape()


