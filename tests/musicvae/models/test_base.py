import torch
import numpy as np
from torch.distributions.one_hot_categorical import Categorical

from symusic.musicvae.models.base import Encoder, SimpleDecoder, AnotherDecoder
import symusic.musicvae.models.base as base


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


def test_simple_decoder():
    batch_size = 8
    output_size = 90
    hidden_size = 10
    embed_size = 12

    dec = base.decoder_factory.create("simple",
                                      **dict(output_size=output_size, n_layers=2, dropout_prob=0.2,
                                             hidden_size=hidden_size,
                                             embed_size=embed_size))
    assert isinstance(dec, base.SimpleDecoder)

    input = Categorical(probs=torch.ones((batch_size, output_size))).sample()

    hidden, cell = torch.zeros(2, dec.n_layers, batch_size, hidden_size)
    output, (hidden, cell) = dec(input, (hidden, cell), None)

    assert output.shape == torch.Size([batch_size, output_size])
    assert hidden.shape == torch.Size([dec.n_layers, batch_size, hidden_size])
    assert cell.shape == torch.Size([dec.n_layers, batch_size, hidden_size])

    top1 = output.detach().argmax(dim=-1)
    input = top1

    output, (hidden, cell) = dec(input, (hidden, cell), None)

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

    dec = base.decoder_factory.create("another",
                                      **dict(output_size=output_size, n_layers=2, dropout_prob=0.3,
                                             hidden_size=hidden_size, z_size=z_size,
                                             embed_size=embed_size))
    assert isinstance(dec, base.AnotherDecoder)

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


def test_conductor():
    batch_size = 8
    c_size = 90
    hidden_size = 10
    n_layers = 3

    conductor = base.decoder_factory.create("conductor",
                                            **dict(hidden_size=hidden_size, c_size=c_size, n_layers=n_layers))
    assert isinstance(conductor, base.Conductor)

    input = torch.ones((batch_size, 1))

    hidden, cell = torch.zeros(2, conductor.n_layers, batch_size, hidden_size)
    output, (hidden, cell) = conductor(input, (hidden, cell))

    assert output.shape == torch.Size([batch_size, c_size])
    assert hidden.shape == torch.Size([conductor.n_layers, batch_size, hidden_size])
    assert cell.shape == torch.Size([conductor.n_layers, batch_size, hidden_size])

    ckpt = conductor.create_ckpt()
    kwargs = ckpt["kwargs"]
    assert kwargs["hidden_size"] == hidden_size
    assert kwargs["c_size"] == c_size
    assert kwargs["n_layers"] == n_layers


if __name__ == "__main__":
    test_encoder()
    test_simple_decoder()
    test_another_decoder()
    test_conductor()
