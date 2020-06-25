import torch
from torch.distributions.one_hot_categorical import Categorical

import symusic.musicvae.models.base as base
from symusic.musicvae.models.seq_decoder import seq_decoder_factory


def assert_seq2seq(encoder, seq_decoder, sos_token):
    batch_size = 20
    seq_len = 32

    model = base.Seq2Seq(encoder, seq_decoder, sos_token)

    input_size = model.input_size
    output_size = model.output_size

    for i in range(100):
        input = Categorical(probs=torch.ones((batch_size, seq_len, input_size - 1))).sample()

        outputs, _, _, _ = model(input, teacher_forcing_ratio=1.0)
        assert outputs.shape == torch.Size([batch_size, seq_len, output_size])

        outputs, _, _, _ = model(input, teacher_forcing_ratio=0.0)
        assert outputs.shape == torch.Size([batch_size, seq_len, output_size])

    # check if checkpoint still works
    device = torch.device("cpu")
    ckpt = model.create_ckpt()
    new_model = base.Seq2Seq.load_from_ckpt(ckpt, device, state_dict=model.state_dict())
    for val1, val2 in zip(model.state_dict().values(), new_model.state_dict().values()):
        assert torch.allclose(val1, val2)

    for kwargs1, kwargs2 in zip(ckpt["kwargs"], new_model.create_ckpt()["kwargs"]):
        assert kwargs1 == kwargs2


def test_greedy_seq2seq():
    embed_size = 12
    input_size = output_size = 91
    z_size = 33
    device = torch.device("cpu")
    enc = base.Encoder(input_size=input_size, embed_size=embed_size, hidden_size=32, z_size=z_size, n_layers=2)

    params = {
        "output_size": output_size,
        "z_size": z_size,
        "device": device,
        "decoder_args": {
            "key": "another",
            "params": {
                "embed_size": embed_size,
                "hidden_size": 64,
                "z_size": z_size,
                "n_layers": 2
            }
        }
    }
    seq_decoder = seq_decoder_factory.create("greedy", **params)
    assert_seq2seq(enc, seq_decoder, sos_token=90)


def test_sample_seq2seq():
    embed_size = 12
    input_size = output_size = 91
    z_size = 33
    device = torch.device("cpu")
    enc = base.Encoder(input_size=input_size, embed_size=embed_size, hidden_size=32, z_size=z_size, n_layers=2)

    params = {
        "output_size": output_size,
        "z_size": z_size,
        "device": device,
        "temperature": 3.0,
        "decoder_args": {
            "key": "another",
            "params": {
                "embed_size": embed_size,
                "hidden_size": 64,
                "z_size": z_size,
                "n_layers": 2
            }
        }
    }
    seq_decoder = seq_decoder_factory.create("sample", **params)
    assert_seq2seq(enc, seq_decoder, sos_token=90)


def test_hier_seq2seq():
    embed_size = 12
    input_size = output_size = 91
    z_size = 33
    c_size = 121
    n_subseq = 2

    device = torch.device("cpu")
    enc = base.Encoder(input_size=input_size, embed_size=embed_size, hidden_size=32, z_size=z_size, n_layers=2)

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
    assert_seq2seq(enc, seq_decoder, sos_token=90)


if __name__ == "__main__":
    test_greedy_seq2seq()
    test_sample_seq2seq()
    test_hier_seq2seq()
