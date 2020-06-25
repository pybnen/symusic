import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import symusic.musicvae.utils as utils
import symusic.musicvae.models.base as base
import symusic.musicvae.object_factory as object_factory


class GreedySeqDecoder(nn.Module):

    def __init__(self, decoder, z_size, device):
        super().__init__()

        self.decoder = decoder
        self.device = device
        self.z_size = z_size
        self.output_size = decoder.output_size

        self.sampling_rate = 0.0
        self.output_tokens = None

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

    def get_next_input(self, output):
        """output logits from network"""
        return output.detach().argmax(1)

    def forward(self, z, length=None, trg=None, teacher_forcing_ratio=0.5, initial_input=None):
        assert trg is not None or (initial_input is not None and length is not None)
        # z [batch size, z size]
        # trg None or[batch size, trg len]

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

        next_input = initial_input
        sampling_cnt = 0.
        for t in range(0, trg_len):
            assert next_input is not None or t == 0

            # get input for decoder
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            if teacher_force or next_input is None:
                # use target token as input
                input = trg[:, t]
            else:
                # use predicted token
                input = next_input
                sampling_cnt += 1

            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, (hidden, cell) = self.decoder(input, (hidden, cell), z.detach())

            # place predictions in a tensor holding predictions for each token
            outputs[:, t] = output

            # get the highest predicted token from our predictions
            next_input = self.get_next_input(output.detach())
            self.output_tokens[:, t] = next_input.detach()

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


class SampleSeqDecoder(GreedySeqDecoder):

    def __init__(self, decoder, z_size, device, temperature=1.0):
        super().__init__(decoder, z_size, device)
        self.temperature = temperature

    def get_next_input(self, output):
        logits = output / self.temperature

        sampler = Categorical(logits=logits)
        return sampler.sample()


class HierarchicalSeqDecoder(nn.Module):

    def __init__(self, conductor, seq_decoder, n_subsequences, z_size, device):
        super().__init__()

        self.conductor = conductor
        self.seq_decoder = seq_decoder
        self.n_subsequences = n_subsequences
        self.z_size = z_size
        self.device = device
        self.output_size = seq_decoder.output_size

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
                "seq_decoder": self.seq_decoder.create_ckpt(),
                "kwargs": dict(n_subsequences=self.n_subsequences,
                               z_size=self.z_size)}
        return ckpt

    @classmethod
    def load_from_ckpt(cls, ckpt, device, state_dict=None):
        cond = utils.load_class_by_name(ckpt["conductor"]["clazz"], **ckpt["conductor"]["kwargs"])
        #dec = utils.load_class_by_name(ckpt["decoder"]["clazz"], **ckpt["decoder"]["kwargs"])
        bottom_seq_decoder = utils.get_class_by_name(ckpt["seq_decoder"]["clazz"]).load_from_ckpt(ckpt["seq_decoder"], device)

        seq_decoder = cls(conductor=cond, seq_decoder=bottom_seq_decoder, device=device, **ckpt["kwargs"])
        if state_dict is not None:
            seq_decoder.load_state_dict(state_dict)

        return seq_decoder


# seq decoder builder definitions -----------------------------------------------------------------
def build_greedy_seq_decoder(output_size, decoder_args, z_size, device, **_ignore):
    params = {**decoder_args["params"], "output_size": output_size}
    decoder = base.decoder_factory.create(decoder_args["key"], **params)
    return GreedySeqDecoder(decoder, z_size, device)


def build_sample_seq_decoder(output_size, decoder_args, z_size, device, temperature, **_ignore):
    params = {**decoder_args["params"], "output_size": output_size}
    decoder = base.decoder_factory.create(decoder_args["key"], **params)
    return SampleSeqDecoder(decoder, z_size, device, temperature)


def build_hier_seq_decoder(output_size, conductor_args, seq_decoder_args, n_subsequences, z_size, device, **_ignore):
    conductor = base.decoder_factory.create(conductor_args["key"], **conductor_args["params"])

    params = {**seq_decoder_args["params"], "output_size": output_size, "device": device}
    seq_decoder = seq_decoder_factory.create(seq_decoder_args["key"], **params)
    return HierarchicalSeqDecoder(conductor, seq_decoder, n_subsequences, z_size, device)


seq_decoder_factory = object_factory.ObjectFactory()
seq_decoder_factory.register_builder("greedy", build_greedy_seq_decoder)
seq_decoder_factory.register_builder("sample", build_sample_seq_decoder)
seq_decoder_factory.register_builder("hier", build_hier_seq_decoder)