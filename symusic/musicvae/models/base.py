import torch
import torch.nn as nn
import torch.nn.functional as F

import symusic.musicvae.utils as utils
import symusic.musicvae.object_factory as object_factory

# encoder definitions ------------------------------------------------------------------------------
class Encoder(nn.Module):

    def __init__(self, input_size, embed_size, hidden_size, z_size, n_layers, dropout_prob=.0):
        super().__init__()

        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.z_size = z_size
        self.n_layers = n_layers
        self.dropout_prob = dropout_prob

        self.embedding = nn.Embedding(input_size, embed_size)
        self.dropout = nn.Dropout(dropout_prob)

        self.rnn = nn.LSTM(input_size=embed_size, hidden_size=hidden_size,
                           num_layers=n_layers, bidirectional=True, dropout=dropout_prob)

        # multiply hidden size by 2 because of bidirectional
        in_features = hidden_size * 2
        self.fc_mu = nn.Linear(in_features=in_features, out_features=z_size)
        self.fc_logvar = nn.Linear(in_features=in_features, out_features=z_size)

    def forward(self, input: torch.Tensor):
        # input [batch size, input len]

        input = input.permute((1, 0))
        embedded = self.dropout(self.embedding(input))

        # embedded [input len, batch size, embed size]

        output, (hidden, cell) = self.rnn(embedded)
        output = output[-1]

        # output [batch size, hidden size * n_directions]

        mu, logvar = self.fc_mu(output), self.fc_logvar(output)

        # mu [batch size, z_size]
        # logvar [batch size, z_size]

        return mu, logvar

    def create_ckpt(self):
        ckpt = {"clazz": ".".join([self.__module__, self.__class__.__name__]),
                "kwargs": dict(input_size=self.input_size,
                               embed_size=self.embed_size,
                               hidden_size=self.hidden_size,
                               z_size=self.z_size,
                               n_layers=self.n_layers,
                               dropout_prob=self.dropout_prob)}
        return ckpt


# decoder definitions -----------------------------------------------------------------------------
class SimpleDecoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, n_layers, dropout_prob=.0):
        super().__init__()

        self.output_size = output_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_prob = dropout_prob

        self.embedding = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout_prob)

        self.rnn = nn.LSTM(input_size=embed_size, hidden_size=hidden_size,
                           num_layers=n_layers, dropout=dropout_prob)

        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, input: torch.Tensor, state: torch.Tensor, _: torch.Tensor):
        hidden, cell = state

        # input [batch size]
        # hidden [n layers, batch size, hidden size]
        # cell [n layers, batch size, hidden size]

        embedded = self.dropout(self.embedding(input))

        # embedded [batch size, embed size]

        output, (hidden, cell) = self.rnn(embedded.unsqueeze(dim=0), (hidden, cell))

        # output [1, batch_size, hidden size]

        output = self.fc_out(output.squeeze(dim=0))

        # output [batch size, output size]

        return output, (hidden, cell)

    def create_ckpt(self):
        ckpt = {"clazz": ".".join([self.__module__, self.__class__.__name__]),
                "kwargs": dict(output_size=self.output_size,
                               embed_size=self.embed_size,
                               hidden_size=self.hidden_size,
                               n_layers=self.n_layers,
                               dropout_prob=self.dropout_prob)}
        return ckpt


class AnotherDecoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, z_size, n_layers, dropout_prob=.0):
        super().__init__()

        self.output_size = output_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.z_size = z_size
        self.n_layers = n_layers
        self.dropout_prob = dropout_prob

        self.embedding = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout_prob)

        self.rnn = nn.LSTM(input_size=embed_size + z_size, hidden_size=hidden_size,
                           num_layers=n_layers, dropout=dropout_prob)

        self.fc_out = nn.Linear(embed_size + z_size + hidden_size, output_size)

    def forward(self, input: torch.Tensor, state: torch.Tensor, z: torch.Tensor):
        hidden, cell = state

        # input [batch size]
        # hidden [n layers, batch size, hidden size]
        # cell [n layers, batch size, hidden size]
        # z [batch size, z size]

        embedded = self.dropout(self.embedding(input))

        # embedded [batch size, embed size]

        input_concat = torch.cat((embedded, z.detach()), dim=1)

        # input_concat [batch_size, output size + z size]

        output, (hidden, cell) = self.rnn(input_concat.unsqueeze(dim=0), (hidden, cell))

        # output [1, batch size, hidden size]

        output = torch.cat((embedded, output.squeeze(dim=0), z.detach()), dim=1)

        # output [batch size, output_size + hidden_size + z size]

        output = self.fc_out(output)

        # output [batch_size, output size]

        return output, (hidden, cell)

    def create_ckpt(self):
        ckpt = {"clazz": ".".join([self.__module__, self.__class__.__name__]),
                "kwargs": dict(output_size=self.output_size,
                               embed_size=self.embed_size,
                               hidden_size=self.hidden_size,
                               n_layers=self.n_layers,
                               z_size=self.z_size,
                               dropout_prob=self.dropout_prob)}
        return ckpt


class Conductor(nn.Module):
    def __init__(self, hidden_size, c_size, n_layers):
        super().__init__()

        self.hidden_size = hidden_size
        self.c_size = c_size
        self.n_layers = n_layers

        self.rnn = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=n_layers)

        self.fc_out = nn.Linear(hidden_size, c_size)

    def forward(self, input: torch.Tensor, state: torch.Tensor):
        hidden, cell = state

        # input [batch size, 1]
        # hidden [n layers, batch size, hidden size]
        # cell [n layers, batch size, hidden size]

        output, (hidden, cell) = self.rnn(input.unsqueeze(dim=0), (hidden, cell))

        # output [1, batch_size, hidden size]

        output = self.fc_out(output.squeeze(dim=0))

        # output [batch size, c size]

        return output, (hidden, cell)

    def create_ckpt(self):
        ckpt = {"clazz": ".".join([self.__module__, self.__class__.__name__]),
                "kwargs": dict(hidden_size=self.hidden_size,
                               c_size=self.c_size,
                               n_layers=self.n_layers)}
        return ckpt


# decoder builder definitions ---------------------------------------------------------------------
def build_simple_decoder(output_size, embed_size, hidden_size, n_layers, dropout_prob=.0, **_ignore):
    return SimpleDecoder(output_size, embed_size, hidden_size, n_layers, dropout_prob)


def build_another_decoder(output_size, embed_size, hidden_size, z_size, n_layers, dropout_prob=.0, **_ignore):
    return AnotherDecoder(output_size, embed_size, hidden_size, z_size, n_layers, dropout_prob)


def build_conductor(hidden_size, c_size, n_layers, **_ignore):
    return Conductor(hidden_size, c_size, n_layers)


decoder_factory = object_factory.ObjectFactory()
decoder_factory.register_builder("simple", build_simple_decoder)
decoder_factory.register_builder("another", build_another_decoder)
decoder_factory.register_builder("conductor", build_conductor)


# seq2seq definitions -----------------------------------------------------------------------------
class Seq2Seq(nn.Module):
    def __init__(self, encoder, seq_decoder, sos_token):
        super().__init__()

        self.encoder = encoder
        self.seq_decoder = seq_decoder
        self.sos_token = sos_token
        self.sampling_rate = 0.0

        self.input_size = encoder.input_size
        self.output_size = seq_decoder.output_size

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, src, teacher_forcing_ratio=0.5):
        # symusic = [batch size, symusic len]

        # encode source input sequence
        mu, logvar = self.encoder(src)
        z = self.reparameterize(mu, logvar)

        # add <sos> token at the start of sequence and trim last input
        #  to get decoder input
        trg = F.pad(src, pad=[1, 0, 0, 0], value=self.sos_token)[:, :-1]

        # trg = [batch size, symusic len]

        outputs = self.seq_decoder(z, trg=trg, teacher_forcing_ratio=teacher_forcing_ratio)

        self.sampling_rate = self.seq_decoder.sampling_rate
        return outputs, mu, logvar, z

    def create_ckpt(self):
        ckpt = {
            "kwargs": dict(sos_token=self.sos_token),
            "encoder": self.encoder.create_ckpt(),
            "seq_decoder": self.seq_decoder.create_ckpt()
        }
        return ckpt

    @classmethod
    def load_from_ckpt(cls, ckpt, device, state_dict=None):
        enc = utils.load_class_by_name(ckpt["encoder"]["clazz"], **ckpt["encoder"]["kwargs"])
        seq_decoder = utils.get_class_by_name(ckpt["seq_decoder"]["clazz"]).load_from_ckpt(ckpt["seq_decoder"], device)

        model = cls(enc, seq_decoder, **ckpt["kwargs"])
        if state_dict is not None:
            model.load_state_dict(state_dict)

        return model
