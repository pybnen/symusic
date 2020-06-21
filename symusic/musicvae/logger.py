import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pypianoroll as pp

import torch

import symusic.musicvae.utils as utils
import symusic.music.melody_lib as melody_lib

matplotlib.use('Agg')

MIN_PITCH = 21
MAX_PITCH = 108
NUM_SPECIAL_EVENTS = 2


def decode_event(event):
    if event < NUM_SPECIAL_EVENTS:
        # event is a special event
        return event - NUM_SPECIAL_EVENTS
    event = event - NUM_SPECIAL_EVENTS + MIN_PITCH
    assert MIN_PITCH <= event <= MAX_PITCH
    return event


def decode_melody(melody):
    return np.array([decode_event(event) for event in melody])


class Logger:
    def __init__(self, writer, ckpt_dir, melody_dict):
        self.writer = writer
        self.ckpt_dir = ckpt_dir
        self.melody_dict = melody_dict
        self.metrics = None
        self.metrics_tags = {
            "loss": "{}.loss",
            "r_loss": "{}.losses/r_loss",
            "kl_loss": "{}.losses/kl_loss",
            "kl_div": "{}.losses/kl_div",
            "acc": "accuracy/{}",
            "sampling": "sampling/{}",
            "beta": "{}.losses/beta"
        }
        self.metrics_cnt = 0
        self.histograms = None
        self.histograms_tags = {
            "mu": "{}.encoder/mean",
            "logvar": "{}.encoder/logvar",
            "z": "{}.encoder/z",
        }
        self.reset()

    def reset(self):
        self.reset_metrics()
        self.reset_histograms()

    def reset_metrics(self):
        self.metrics_cnt = 0
        self.metrics = dict(loss=0.0, r_loss=0.0, kl_loss=0.0, kl_div=0.0, acc=0.0, sampling=0.0, beta=0.0)

    def reset_histograms(self):
        self.histograms = dict(mu=[], logvar=[], z=[])

    def add_step(self, loss, r_loss, kl_loss, kl_div, acc, sampling_rate, beta, mu, logvar, z):
        self.metrics["loss"] += loss
        self.metrics["r_loss"] += r_loss
        self.metrics["kl_loss"] += kl_loss
        self.metrics["kl_div"] += kl_div
        self.metrics["acc"] += acc
        self.metrics["sampling"] += sampling_rate
        self.metrics["beta"] += beta
        self.histograms["mu"].append(mu)
        self.histograms["logvar"].append(logvar)
        self.histograms["z"].append(z)
        self.metrics_cnt += 1

    def log_metrics(self, label, global_step, reset=True):
        for key, value in self.metrics.items():
            tag = self.metrics_tags[key].format(label)
            self.writer.add_scalar(tag, value / self.metrics_cnt, global_step)

        if reset:
            self.reset_metrics()

    def print_metrics(self, global_step, duration, eval=False, new_best=False):
        stats = {}
        for key, value in self.metrics.items():
            stats[key] = value / self.metrics_cnt
        if not eval:
            print("{:d} | {:.6f} | {:.6f} | {:.6f} | {:.4f} | {:.4f} sec.".format(
                global_step, stats["loss"], stats["r_loss"], stats["kl_loss"], stats["acc"], duration))
        else:
            print('====> evaluation loss={:.6f}, r_loss={:.6f}, kl_loss={:.6f}, acc={:.4f} ({:.4f} sec.){}'.format(
                stats["loss"], stats["r_loss"], stats["kl_loss"], stats["acc"], duration,
                " *new best*" if new_best else ""))

    def log_histograms(self, label, global_step, reset=True):
        for key, value in self.histograms.items():
            tag = self.histograms_tags[key].format(label)
            self.writer.add_histogram(tag, torch.cat(value), global_step)

        if reset:
            self.reset_histograms()

    def log_grad_norm(self, label, global_step, model=None, norm=None):
        if model is not None:
            self.writer.add_scalar(f"{label}.grad_norm/encoder", utils.get_grad_norm(model.encoder.parameters()),
                                   global_step)
            self.writer.add_scalar(f"{label}.grad_norm/decoder", utils.get_grad_norm(model.seq_decoder.parameters()),
                                   global_step)
        else:
            self.writer.add_scalar(f"{label}.grad_norm/global", norm, global_step)

    def log_reconstruction(self, label, output, trg, global_step, max_results=3):
        pred_sequence = output[:max_results].argmax(dim=-1)
        target_sequence = trg[:max_results].cpu()

        n_results = pred_sequence.size(0)
        fig = plt.figure(figsize=(15, 8))
        for i in range(n_results):
            ax = fig.add_subplot(2, n_results, i + 1)
            pp.plot_pianoroll(ax, melody_lib.melody_to_pianoroll(self.melody_dict.sequence_to_melody(target_sequence[i])))
            plt.title('Original melody')

            ax = fig.add_subplot(2, n_results, n_results + i + 1)
            pp.plot_pianoroll(ax, melody_lib.melody_to_pianoroll(self.melody_dict.sequence_to_melody(pred_sequence[i])))
            plt.title('Reconstruction')

        self.writer.add_figure(f"{label}.recon", fig, global_step)
        plt.close(fig)

    def save_ckpt(self, model, opt, scheduler, global_step, new_best):
        ckpt_path = self.ckpt_dir / ("ckpt_best.pth" if new_best else "ckpt_current.pth")
        utils.save_ckpt(ckpt_path, global_step, model, opt, scheduler)

