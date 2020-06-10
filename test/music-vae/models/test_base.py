import torch
import numpy as np
from torch.distributions.one_hot_categorical import OneHotCategorical, Categorical
from music_vae.models.base import Encoder, SimpleDecoder, AnotherDecoder, Seq2Seq


def test_encoder():
    batch_size = 8
    input_length = 10
    input_size = 90
    embed_size = 12
    z_size = 4

    enc = Encoder(input_size=input_size, embed_size=embed_size, hidden_size=1048, z_size=z_size,
                  n_layers=2, dropout_prob=0.2)

    input = Categorical(probs=torch.ones((batch_size, input_length, input_size))).sample()
    mu, logvar = enc(input)
    assert mu.shape == torch.Size([batch_size, z_size])
    assert logvar.shape == torch.Size([batch_size, z_size])

    ckpt = enc.create_ckpt()
    kwargs = ckpt["kwargs"]
    assert kwargs["input_size"] == input_size
    assert kwargs["embed_size"] == embed_size
    assert kwargs["hidden_size"] == 1048
    assert kwargs["z_size"] == z_size
    assert kwargs["n_layers"] == 2
    assert kwargs["dropout_prob"] == 0.2



def test_decoder():
    batch_size = 8
    output_size = 90
    hidden_size = 10
    embed_size = 12

    dec = SimpleDecoder(output_size=output_size, embed_size=embed_size, hidden_size=hidden_size,
                        n_layers=2, dropout_prob=0.2)

    input = Categorical(probs=torch.ones((batch_size, output_size))).sample()

    hidden, cell = torch.zeros(2, dec.n_layers, batch_size, hidden_size)
    output, (hidden, cell) = dec(input, (hidden, cell))

    assert output.shape == torch.Size([batch_size, output_size])
    assert hidden.shape == torch.Size([dec.n_layers, batch_size, hidden_size])
    assert cell.shape == torch.Size([dec.n_layers, batch_size, hidden_size])

    top1 = output.detach().argmax(dim=-1)
    input = top1

    output, (hidden, cell) = dec(input, (hidden, cell))

    assert output.shape == torch.Size([batch_size, output_size])
    assert hidden.shape == torch.Size([dec.n_layers, batch_size, hidden_size])
    assert cell.shape == torch.Size([dec.n_layers, batch_size, hidden_size])

    ckpt = dec.create_ckpt()
    kwargs = ckpt["kwargs"]
    assert kwargs["output_size"] == output_size
    assert kwargs["embed_size"] == embed_size
    assert kwargs["hidden_size"] == hidden_size
    assert kwargs["n_layers"] == 2
    assert kwargs["dropout_prob"] == 0.2


def test_another_decoder():
    batch_size = 8
    z_size = 9
    output_size = 90
    hidden_size = 10
    embed_size = 12

    dec = AnotherDecoder(output_size=output_size, embed_size=embed_size, hidden_size=hidden_size,
                         z_size=z_size, n_layers=2, dropout_prob=0.3)

    input = Categorical(probs=torch.ones((batch_size, output_size))).sample()
    z = torch.randn((batch_size, z_size))

    hidden, cell = torch.zeros(2, dec.n_layers, batch_size, hidden_size)
    output, (hidden, cell) = dec(input, (hidden, cell), z)

    assert output.shape == torch.Size([batch_size, output_size])
    assert hidden.shape == torch.Size([dec.n_layers, batch_size, hidden_size])
    assert cell.shape == torch.Size([dec.n_layers, batch_size, hidden_size])

    top1 = output.detach().argmax(dim=-1)
    input = top1

    output, (hidden, cell) = dec(input, (hidden, cell), z)

    assert output.shape == torch.Size([batch_size, output_size])
    assert hidden.shape == torch.Size([dec.n_layers, batch_size, hidden_size])
    assert cell.shape == torch.Size([dec.n_layers, batch_size, hidden_size])

    ckpt = dec.create_ckpt()
    kwargs = ckpt["kwargs"]
    assert kwargs["output_size"] == output_size
    assert kwargs["embed_size"] == embed_size
    assert kwargs["hidden_size"] == hidden_size
    assert kwargs["z_size"] == z_size
    assert kwargs["n_layers"] == 2
    assert kwargs["dropout_prob"] == 0.3


def test_seq2seq():
    batch_size = 3
    embed_size = 12
    input_size = output_size = 90
    seq_len = 4
    z_size = 33

    enc = Encoder(input_size=input_size, embed_size=embed_size, hidden_size=32, z_size=z_size, n_layers=2)
    dec = AnotherDecoder(output_size=output_size, embed_size=embed_size, hidden_size=64, z_size=z_size, n_layers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2Seq(enc, dec, device)

    input = Categorical(probs=torch.ones((batch_size, seq_len, input_size))).sample()

    outputs, _, _, _ = model(input, input, teacher_forcing_ratio=1.0)
    assert outputs.shape == torch.Size([batch_size, seq_len, output_size])

    outputs, _, _, _ = model(input, input, teacher_forcing_ratio=0.0)
    assert outputs.shape == torch.Size([batch_size, seq_len, output_size])

    ckpt = model.create_ckpt()
    new_model = Seq2Seq.load_from_ckpt(ckpt, model.state_dict(), device)
    for val1, val2 in zip(model.state_dict().values(), new_model.state_dict().values()):
        assert torch.allclose(val1, val2)


def test_seq2seq_load_ckpt():
    ckpt_path = "./ckpt/ckpt.pth"
    device = torch.device("cpu")

    ckpt = torch.load(ckpt_path, map_location=device)
    model = Seq2Seq.load_from_ckpt(ckpt["model_kwargs"], ckpt["model_state_dict"], device)


if __name__ == "__main__":
    test_encoder()
    test_decoder()
    test_another_decoder()
    test_seq2seq()
    test_seq2seq_load_ckpt()
