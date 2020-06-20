import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

from music_vae.models.base import Seq2Seq


class TrainedModel:

    def __init__(self, ckpt_path, melody_dict, device):
        with open(ckpt_path, 'rb') as file:
            ckpt = torch.load(file, map_location=device)
            model_ckpt = ckpt["model_kwargs"]
            state_dict = ckpt["model_state_dict"]

        self.model = Seq2Seq.load_from_ckpt(model_ckpt, device, state_dict=state_dict).to(device)
        self.melody_dict = melody_dict
        self.device = device
        # self.model.decoder.set_sampling_probability(1.0)
        self.z_size = self.model.encoder.z_size
        self.output_size = self.model.output_size

    def melodies_to_tensors(self, melodies):
        #  use melody dict to translate melodies to tensors
        melody_tensors = torch.from_numpy(self.melody_dict(melodies))
        return melody_tensors

    def tensors_to_melodies(self, melody_tensors):
        # use melody dict to translate tensors to melodies
        melodies = self.melody_dict.sequence_to_melody(melody_tensors)
        return melodies

    def encode(self, melodies):
        """Encode melodies to latent space

        melodies: np.array of integer type encoded with special events, shape (n_melodies, melody_length)

        Returns torch.tensor, shape (n_melodies, z_dim)"""
        tensor_melodies = self.melodies_to_tensors(melodies)
        return self.encode_tensors(tensor_melodies)

    def encode_tensors(self, tensor_melodies):
        self.model.eval()
        with torch.no_grad():
            tensor_melodies = tensor_melodies.to(self.device)
            mu, logvar = self.model.encoder(tensor_melodies)
            z = self.model.reparameterize(mu, logvar)
            return z, mu, logvar

    def sample(self, n, length, temperature=1.0, same_z=False):
        if same_z:
            z = torch.randn((1, self.z_size)).repeat(n, 1)
        else:
            z = torch.randn((n, self.z_size))
        return self.decode(z, length, temperature=temperature)

    def decode(self, z, length, temperature=1.0):
        """Decode latent space to melodies

        z: torch.tensor, shape (n_melodies, z_size)

        Returns np.array of integer, shape (n_melodies, length)"""
        output_tokens, outputs = self.decode_to_tensors(z, length, temperature=temperature)
        return self.tensors_to_melodies(output_tokens), outputs.detach()

    def decode_to_tensors(self, z, length, temperature=1.0):
        # self.model.decoder.set_temperature(temperature)
        self.model.eval()
        with torch.no_grad():
            batch_size = z.shape[0]
            initial_input = torch.tensor(self.model.sos_token).repeat(batch_size).to(self.device)
            z = z.to(self.device)
            trg = None
            outputs = self.model.seq_decoder(z, length=length, trg=None,
                                             teacher_forcing_ratio=0.0, initial_input=initial_input)
            return self.model.seq_decoder.output_tokens.detach(), outputs.detach()

    def interpolate(self, start_melody, end_melody, num_steps, length, temperature=1.0):
        def _slerp(p0, p1, t):
            """Spherical linear interpolation."""
            omega = np.arccos(np.dot(np.squeeze(p0 / np.linalg.norm(p0)),
                                     np.squeeze(p1 / np.linalg.norm(p1))))
            so = np.sin(omega)
            return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1

        melodies = np.array([start_melody, end_melody])
        z, _, _ = self.encode(melodies)
        z = z.detach().cpu()
        interpolated_z = torch.stack([_slerp(z[0], z[1], t) for t in np.linspace(0, 1, num_steps)])
        return self.decode(interpolated_z, length, temperature=temperature)


class FakeModel:
    def __init__(self, ckpt_path, z_dim):
        self.z_dim = z_dim
        self.model = None  # load from checkpoint path

    def sample(self, n, length, temperature=1.0, same_z=False):
        if same_z:
            z = torch.randn((1, self.z_dim)).repeat(n, 1)
        else:
            z = torch.randn((n, self.z_dim))
        return self.decode(z, length, temperature=temperature)

    def encode(self, melodies):
        """Encode melodies to latent space

        melodies: np.array of integer type encoded with special events, shape (n_melodies, melody_length)

        Returns torch.tensor, shape (n_melodies, z_dim)"""
        return torch.ones((melodies.shape[0], self.z_dim))  # (torch.randn((melodies.shape[0], self.z_dim))

    def decode(self, z, length, temperature=1.0):
        """Decode latent space to melodies

        z: torch.tensor, shape (n_melodies, z_dim)

        Returns np.array of integer, shape (n_melodies, length)"""
        return np.random.randint(-2, 128, size=(z.shape[0], length))

    def interpolate(self, start_melody, end_melody, num_steps, length, temperature=1.0):
        def _slerp(p0, p1, t):
            """Spherical linear interpolation."""
            omega = np.arccos(np.dot(np.squeeze(p0 / np.linalg.norm(p0)),
                                     np.squeeze(p1 / np.linalg.norm(p1))))
            so = np.sin(omega)
            return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1

        melodies = np.array([start_melody, end_melody])
        z = self.encode(melodies).detach().cpu().numpy()
        z = np.array([_slerp(z[0], z[1], t) for t in np.linspace(0, 1, num_steps)])
        return self.decode(z, length, temperature=temperature)

