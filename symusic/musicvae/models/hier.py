import torch
import torch.nn as nn

import symusic.musicvae.models.base as base
import symusic.musicvae.utils as utils


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


class HierarchicalSeqDecoder(nn.Module):

    def __init__(self, conductor, decoder, n_subsequences, z_size, device):
        super().__init__()

        self.conductor = conductor
        self.seq_decoder = base.SimpleSeqDecoder(decoder, z_size=conductor.c_size, device=device)
        self.n_subsequences = n_subsequences
        self.z_size = z_size
        self.device = device
        self.output_size = decoder.output_size

        self.sampling_rate = 0.0
        self.output_tokens = None

        n_states = conductor.hidden_size * conductor.n_layers * 2
        self.fc_state = nn.Sequential(
            nn.Linear(z_size, n_states),
            nn.Tanh()
        )

    def cell_state_from_context(self, z):
        batch_size = z.shape[0]
        states = self.fc_state(z)
        hidden, cell = states.view(batch_size, self.conductor.n_layers, 2, -1).permute(2, 1, 0, 3).contiguous()
        return hidden, cell

    def forward(self, z, length=None, trg=None, teacher_forcing_ratio=0.5, initial_input=None):
        assert trg is not None or (initial_input is not None and length is not None)
        # z [batch size, z size]
        # trg [batch size, trg len]

        batch_size = z.shape[0]
        trg_len = trg.shape[1] if trg is not None else length
        if trg is None:
            assert teacher_forcing_ratio == 0.0

        output_size = self.output_size

        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, output_size).to(self.device)

        # contains output tokens instead of logits
        self.output_tokens = torch.zeros(batch_size, trg_len, dtype=torch.int64).to(self.device)

        # z is used to init hidden states of decoder
        hidden, cell = self.cell_state_from_context(z)

        if trg is None:
            assert length % self.n_subsequences == 0
            subsequences = None
            subsequence_length = int(length / self.n_subsequences)
        else:
            subsequences = trg.view(batch_size, self.n_subsequences, -1)
            subsequence_length = subsequences.shape[-1]

        start_idx = 0
        total_sampling_rate = 0.0
        for t in range(self.n_subsequences):
            # get conductor embedding for subsequence (use 1 as input)
            input = torch.ones((batch_size, 1)).to(self.device)
            c, (hidden, cell) = self.conductor(input, (hidden, cell))

            # use sequence decoder to get output for subsequence, using current conductor embedding
            if subsequences is not None:
                subseq = subsequences[:, t]
            else:
                subseq = None

            decoder_outputs = self.seq_decoder(c, length=subsequence_length,
                                               trg=subseq,
                                               teacher_forcing_ratio=teacher_forcing_ratio,
                                               initial_input=initial_input)

            total_sampling_rate += self.seq_decoder.sampling_rate
            outputs[:, start_idx:start_idx+subsequence_length] = decoder_outputs
            self.output_tokens[:, start_idx:start_idx+subsequence_length] = self.seq_decoder.output_tokens
            start_idx += subsequence_length
            initial_input = self.seq_decoder.output_tokens[:, -1].detach()

        self.sampling_rate = total_sampling_rate / self.n_subsequences
        return outputs

    def create_ckpt(self):
        ckpt = {"clazz": ".".join([self.__module__, self.__class__.__name__]),
                "conductor": self.conductor.create_ckpt(),
                "decoder": self.seq_decoder.decoder.create_ckpt(),
                "kwargs": dict(n_subsequences=self.n_subsequences,
                               z_size=self.z_size)}
        return ckpt

    @classmethod
    def load_from_ckpt(cls, ckpt, device, state_dict=None):
        cond = utils.load_class_by_name(ckpt["conductor"]["clazz"], **ckpt["conductor"]["kwargs"])
        dec = utils.load_class_by_name(ckpt["decoder"]["clazz"], **ckpt["decoder"]["kwargs"])

        seq_decoder = cls(conductor=cond, decoder=dec, device=device, **ckpt["kwargs"])
        if state_dict is not None:
            seq_decoder.load_state_dict(state_dict)

        return seq_decoder