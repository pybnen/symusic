import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from music_vae import utils


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


class SimpleSeqDecoder(nn.Module):

    def __init__(self, decoder, z_size, device):
        super().__init__()

        self.decoder = decoder
        self.device = device
        self.z_size = z_size
        self.output_size = decoder.output_size
        self.sampling_rate = 0.0

        n_states = decoder.hidden_size * decoder.n_layers * 2
        self.fc_state = nn.Sequential(
            nn.Linear(z_size, n_states),
            nn.Tanh()
        )

    def cell_state_from_context(self, z):
        batch_size = z.shape[0]
        states = self.fc_state(z)
        hidden, cell = states.view(batch_size, self.decoder.n_layers, 2, -1).permute(2, 1, 0, 3).contiguous()
        return hidden, cell

    def forward(self, trg, z, teacher_forcing_ratio=0.5, initial_input=None):
        # trg [batch size, trg len]
        # z [batch size, z size]

        batch_size, trg_len = trg.shape
        output_size = self.output_size

        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, output_size).to(self.device)

        # z is used to init hidden states of decoder
        hidden, cell = self.cell_state_from_context(z)

        top1 = initial_input
        sampling_cnt = 0.
        for t in range(0, trg_len):
            assert top1 is not None or t == 0

            # get input for decoder
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            if teacher_force or top1 is None:
                # use target token as input
                input = trg[:, t]
            else:
                # use predicted token
                input = top1
                sampling_cnt += 1

            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, (hidden, cell) = self.decoder(input, (hidden, cell), z.detach())

            # place predictions in a tensor holding predictions for each token
            outputs[:, t] = output

            # get the highest predicted token from our predictions
            top1 = output.detach().argmax(1)

        self.sampling_rate = sampling_cnt / trg_len
        return outputs

    def create_ckpt(self):
        ckpt = {"clazz": ".".join([self.__module__, self.__class__.__name__]),
                "decoder": self.decoder.create_ckpt(),
                "kwargs": dict(z_size=self.z_size)}
        return ckpt

    @classmethod
    def load_from_ckpt(cls, ckpt, device, state_dict=None):
        dec = utils.load_class_by_name(ckpt["decoder"]["clazz"], **ckpt["decoder"]["kwargs"])

        seq_decoder = cls(decoder=dec, device=device, **ckpt["kwargs"])
        if state_dict is not None:
            seq_decoder.load_state_dict(state_dict)

        return seq_decoder


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
        # src = [batch size, src len]

        # encode source input sequence
        mu, logvar = self.encoder(src)
        z = self.reparameterize(mu, logvar)

        # add <sos> token at the start of sequence and trim last input
        #  to get decoder input
        trg = F.pad(src, pad=[1, 0, 0, 0], value=self.sos_token)[:, :-1]

        # trg = [batch size, src len]

        outputs = self.seq_decoder(trg, z, teacher_forcing_ratio=teacher_forcing_ratio)

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
