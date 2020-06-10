from pathlib import Path
import time
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils.clip_grad
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sacred import Experiment

from music_vae.datasets.melody_dataset import FixedLengthMelodyDataset, MelodyEncode
from music_vae.models.base import Seq2Seq, AnotherDecoder, Encoder
import music_vae.utils as utils
from music_vae.logger import Logger

ex = Experiment('train_music_vae')

logger = None


def criterion(output, target, mu, logvar, free_bits, beta):
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = torch.mean(torch.max(kl_div - (free_bits * np.log(2.0)), torch.zeros_like(kl_div)))

    _, seq_length, output_size = output.shape

    r_loss = F.cross_entropy(output.view(-1, output_size), target.view(-1), reduction="mean")

    loss = seq_length * r_loss + beta * kl_loss
    return loss, r_loss.item(), kl_loss.item(), torch.mean(kl_div.detach()).item()


def evaluate(model, dl_eval, device, global_step, current_best, beta, free_bits):
    model.eval()
    start_time = time.time()
    logger.reset()

    with torch.no_grad():
        for src in dl_eval:
            src = src.to(device)

            output, mu, logvar, z = model(src, src, teacher_forcing_ratio=0.0)
            loss, r_loss, kl_loss, kl_div = criterion(output, src, mu, logvar, free_bits=free_bits, beta=beta)
            # noinspection PyUnresolvedReferences
            acc = torch.mean((output.detach().argmax(dim=-1) == src.detach()).float()).item()

            logger.add_step(loss.detach().item(), r_loss, kl_loss, kl_div, acc, model.sampling_rate, beta,
                            mu.detach().cpu(), logvar.detach().cpu(), z.detach().cpu())

    loss = logger.metrics["loss"] / logger.metrics_cnt
    new_best = loss < current_best

    logger.print_metrics(global_step, time.time() - start_time, eval=True, new_best=new_best)
    logger.log_metrics("eval", global_step)
    logger.log_histograms("eval", global_step)
    logger.log_reconstruction("eval", output.detach().cpu(), src.detach().cpu(), global_step)

    return loss, new_best


def train(model, dl_train, opt, lr_scheduler, device, beta_settings, sampling_settings, free_bits,
          dl_eval=None, grad_clip=1.0, step=1, num_steps=40,
          evaluate_interval=1000, advanced_logging_interval=200, print_metrics_interval=200):

    sampling_fn = utils.get_sampling_fn(**sampling_settings)
    beta_fn = utils.get_beta_fn(**beta_settings)

    logger.reset()
    start_time = time.time()
    model.train()
    current_best = float("inf")

    try:
        while True:
            for src in dl_train:
                src = src.to(device)

                sampling_rate = sampling_fn(step)
                output, mu, logvar, z = model(src, src, teacher_forcing_ratio=1.0 - sampling_rate)
                beta = beta_fn(step)
                loss, r_loss, kl_loss, kl_div = criterion(output, src, mu, logvar, free_bits=free_bits, beta=beta)

                # noinspection PyUnresolvedReferences
                acc = torch.mean((output.detach().argmax(dim=-1) == src.detach()).float()).item()

                logger.add_step(loss.detach().item(), r_loss, kl_loss, kl_div, acc, model.sampling_rate, beta,
                                mu.detach().cpu(), logvar.detach().cpu(), z.detach().cpu())

                opt.zero_grad()
                loss.backward()

                # gradient clipping (and logging)
                logger.log_grad_norm("train", step, model=model)
                total_norm = nn.utils.clip_grad.clip_grad_norm_(model.parameters(), grad_clip)
                logger.log_grad_norm("train", step, norm=total_norm)

                opt.step()

                # log statistics
                if step % print_metrics_interval == 0:
                    end_time = time.time()
                    logger.print_metrics(step, end_time - start_time)
                    start_time = end_time
                logger.log_metrics("train", step)

                if step % advanced_logging_interval == 0:
                    logger.log_histograms("train", step)
                    logger.log_reconstruction("train", output.detach().cpu(), src.detach().cpu(), step)

                if dl_eval is not None and step % evaluate_interval == 0:
                    loss, new_best = evaluate(model, dl_eval, device, step,
                                              current_best, beta_settings["end_beta"], free_bits)
                    if new_best:
                        current_best = loss
                    lr_scheduler.step(loss)
                    logger.save_ckpt(model, opt, lr_scheduler, step, new_best)

                    logger.reset()
                    start_time = time.time()
                    model.train()

                if step >= num_steps:
                    raise StopIteration

                step += 1
    except StopIteration:
        print("Done")


@ex.capture
def run(_run, num_steps, batch_size, num_workers, z_size, beta_settings, sampling_settings, free_bits,
        encoder_params, decoder_params, learning_rate, melody_dir,
        evaluate_interval=1000, advanced_logging_interval=200, print_metrics_interval=200, ckpt_path=None):
    global logger

    # define logger
    run_dir = utils.get_run_dir(_run)
    writer = SummaryWriter(log_dir=run_dir)
    ckpt_dir = Path(writer.log_dir) / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(writer, ckpt_dir=ckpt_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define model
    enc = Encoder(z_size=z_size, **encoder_params)
    dec = AnotherDecoder(z_size=z_size, **decoder_params)
    model = Seq2Seq(enc, dec, device).to(device)

    # define optimizer
    opt = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    # define dataset
    ds = FixedLengthMelodyDataset(melody_dir=melody_dir)
    dl_train = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=True)

    # define scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.01, patience=5, verbose=True)

    step = 1
    if ckpt_path is not None:
        step = utils.load_ckpt(ckpt_path, model, opt, lr_scheduler, device) + 1

    # start train loop
    train(model, dl_train, opt, lr_scheduler, device, beta_settings, sampling_settings, free_bits,
          step=step, num_steps=num_steps, dl_eval=dl_train,
          evaluate_interval=evaluate_interval,
          advanced_logging_interval=advanced_logging_interval,
          print_metrics_interval=print_metrics_interval)


@ex.config
def config():
    num_steps = 20_000

    batch_size = 16
    num_workers = 0

    ckpt_path: None

    evaluate_interval = 1000
    advanced_logging_interval = 200
    print_metrics_interval = 200

    melody_dir = r"C:\Users\yggdrasil\Studium Informatik\12Semester\Project\data\lmd_full_melody_128\0\0\\"

    learning_rate = 1e-3

    z_size = 32

    sampling_settings = {
        "schedule": "inverse_sigmoid",
        "rate": 1000,
    }

    beta_settings = {
        "start_beta": 0.0,
        "end_beta": 0.2,
        "duration": 20_000
    }

    free_bits = 0

    encoder_params = {
        "input_size": 90,
        "embed_size": 12,
        "hidden_size": 16,
        "n_layers": 2
    }

    decoder_params = {
        "output_size": 90,
        "embed_size": 12,
        "hidden_size": 64,
        "n_layers": 2
    }


@ex.automain
def main():
    run()


