import torch

from symusic.musicvae.models.hier import Conductor


def test_conductor():
    batch_size = 8
    c_size = 90
    hidden_size = 10
    n_layers = 3

    conductor = Conductor(hidden_size, c_size, n_layers)

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
    test_conductor()
