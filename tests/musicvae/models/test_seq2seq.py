import torch
from torch.distributions.one_hot_categorical import Categorical

from symusic.musicvae.models.base import Encoder, AnotherDecoder, SimpleSeqDecoder, Seq2Seq
from symusic.musicvae.models.hier import HierarchicalSeqDecoder, Conductor


def assert_seq2seq(encoder, seq_decoder, sos_token):
    batch_size = 20
    seq_len = 32

    model = Seq2Seq(encoder, seq_decoder, sos_token)

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
    new_model = Seq2Seq.load_from_ckpt(ckpt, device, state_dict=model.state_dict())
    for val1, val2 in zip(model.state_dict().values(), new_model.state_dict().values()):
        assert torch.allclose(val1, val2)

    for kwargs1, kwargs2 in zip(ckpt["kwargs"], new_model.create_ckpt()["kwargs"]):
        assert kwargs1 == kwargs2


def test_simple_seq2seq():
    embed_size = 12
    input_size = output_size = 91
    z_size = 33
    device = torch.device("cpu")
    enc = Encoder(input_size=input_size, embed_size=embed_size, hidden_size=32, z_size=z_size, n_layers=2)
    dec = AnotherDecoder(output_size=output_size, embed_size=embed_size, hidden_size=64, z_size=z_size, n_layers=2)
    seq_decoder = SimpleSeqDecoder(dec, z_size, device)

    assert_seq2seq(enc, seq_decoder, sos_token=90)


def test_hier_seq2seq():
    embed_size = 12
    input_size = output_size = 91
    z_size = 33
    c_size = 121
    n_subseq = 2

    device = torch.device("cpu")
    enc = Encoder(input_size=input_size, embed_size=embed_size, hidden_size=32, z_size=z_size, n_layers=2)

    conductor = Conductor(23, c_size, 3)
    dec = AnotherDecoder(output_size=output_size, embed_size=embed_size, hidden_size=64, z_size=c_size, n_layers=2, dropout_prob=0.3)
    seq_decoder = HierarchicalSeqDecoder(conductor, dec, n_subseq, z_size, device)

    assert_seq2seq(enc, seq_decoder, sos_token=90)


if __name__ == "__main__":
    test_simple_seq2seq()
    test_hier_seq2seq()
