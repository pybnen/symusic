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

import symusic.musicvae.datasets.melody_dataset as melody_dataset
import symusic.musicvae.models.base as base
import symusic.musicvae.models.hier as hier
import symusic.musicvae.utils as utils
from symusic.musicvae.logger import Logger

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
            # forward step ------------------------------------------------------------------------
            src = src.to(device)
            output, mu, logvar, z = model(src, teacher_forcing_ratio=0.0)

            # calculate loss/acc ------------------------------------------------------------------
            loss, r_loss, kl_loss, kl_div = criterion(output, src, mu, logvar, free_bits=free_bits, beta=beta)
            # noinspection PyUnresolvedReferences
            acc = torch.mean((output.detach().argmax(dim=-1) == src.detach()).float()).item()

            # add statistics to logger  -----------------------------------------------------------
            logger.add_step(loss.detach().item(), r_loss, kl_loss, kl_div, acc, model.sampling_rate, beta,
                            mu.detach().cpu(), logvar.detach().cpu(), z.detach().cpu())

    # log statistics ------------------------------------------------------------------------------
    loss = logger.metrics["loss"] / logger.metrics_cnt
    new_best = loss < current_best
    logger.print_metrics(global_step, time.time() - start_time, eval=True, new_best=new_best)
    logger.log_metrics("eval", global_step)
    logger.log_histograms("eval", global_step)
    logger.log_reconstruction("eval", output.detach().cpu(), src.detach().cpu(), global_step)

    return loss, new_best


def train(model, dl_train, opt, lr_scheduler, device, beta_settings, sampling_settings, free_bits,
          dl_eval=None, grad_clip=1.0, step=1, num_steps=40,
          evaluate_interval=1000, advanced_interval=200, print_interval=200):

    # get parameter adjustment functions ----------------------------------------------------------
    sampling_fn = utils.get_sampling_fn(**sampling_settings)
    beta_fn = utils.get_beta_fn(**beta_settings)

    # init training ------------------------------------------------------------------------------
    logger.reset()
    start_time = time.time()
    model.train()
    current_best = float("inf")
    epoch = 1

    try:
        while True:
            for src in dl_train:
                # forward step --------------------------------------------------------------------
                src = src.to(device)
                sampling_rate = sampling_fn(step)
                output, mu, logvar, z = model(src, teacher_forcing_ratio=1.0 - sampling_rate)

                # calculate loss/acc --------------------------------------------------------------
                beta = beta_fn(step)
                loss, r_loss, kl_loss, kl_div = criterion(output, src, mu, logvar, free_bits=free_bits, beta=beta)

                # noinspection PyUnresolvedReferences
                acc = torch.mean((output.detach().argmax(dim=-1) == src.detach()).float()).item()

                # add statistics to logger  -------------------------------------------------------
                logger.add_step(loss.detach().item(), r_loss, kl_loss, kl_div, acc, model.sampling_rate, beta,
                                mu.detach().cpu(), logvar.detach().cpu(), z.detach().cpu())

                # backward, grad clipping and optimizer step --------------------------------------
                opt.zero_grad()
                loss.backward()

                logger.log_grad_norm("train", step, model=model)
                total_norm = nn.utils.clip_grad.clip_grad_norm_(model.parameters(), grad_clip)
                logger.log_grad_norm("train", step, norm=total_norm)
                opt.step()

                # log statistics ------------------------------------------------------------------
                if step % print_interval == 0:
                    end_time = time.time()
                    logger.print_metrics(step, end_time - start_time)
                    start_time = end_time
                logger.log_metrics("train", step)

                if step % advanced_interval == 0:
                    logger.log_histograms("train", step)
                    logger.log_reconstruction("train", output.detach().cpu(), src.detach().cpu(), step)

                # evaluate model ------------------------------------------------------------------
                if dl_eval is not None and step % evaluate_interval == 0:
                    loss, new_best = evaluate(model, dl_eval, device, step,
                                              current_best, beta_settings["end_beta"], free_bits)
                    if new_best:
                        current_best = loss

                    # update lr scheduler ---------------------------------------------------------
                    lr_scheduler.step(loss)

                    # save model and reset for training -------------------------------------------
                    logger.save_ckpt(model, opt, lr_scheduler, step, new_best)
                    logger.reset()
                    start_time = time.time()
                    model.train()

                if step >= num_steps:
                    raise StopIteration

                step += 1

            print(f"--- Step {step}: Iterated through all midi files for the {epoch} time.")
            epoch += 1

    except StopIteration:
        print("Done")


@ex.capture
def run(_run, num_steps, batch_size, num_workers, z_size, beta_settings, sampling_settings, free_bits,
        encoder_params, decoder_params, learning_rate, train_dir, eval_dir, slice_bar,
        lr_scheduler_factor, lr_scheduler_patience, use_hier,
        conductor_params=None, c_size=None, n_subsequences=None,
        evaluate_interval=1000, advanced_interval=200, print_interval=200, ckpt_path=None):
    global logger

    # define dataset ------------------------------------------------------------------------------
    enc_mel_to_idx = melody_dataset.MapMelodyToIndex(has_sos_token=False)
    dec_mel_to_idx = melody_dataset.MapMelodyToIndex(has_sos_token=True)

    ds_train = melody_dataset.MelodyDataset(midi_dir=train_dir, slice_bars=slice_bar, transforms=enc_mel_to_idx,
                                            train=True)
    dl_train = DataLoader(ds_train, batch_size=batch_size, num_workers=num_workers, drop_last=True)

    ds_eval = melody_dataset.MelodyDataset(midi_dir=eval_dir, slice_bars=slice_bar, transforms=enc_mel_to_idx,
                                           train=False)
    dl_eval = DataLoader(ds_eval, batch_size=batch_size, num_workers=num_workers, drop_last=False)
    print(f"Train/Eval files: {len(ds_train.midi_files)} / {len(ds_eval.midi_files)}")

    # define logger -------------------------------------------------------------------------------
    run_dir = utils.get_run_dir(_run)
    writer = SummaryWriter(log_dir=run_dir)
    ckpt_dir = Path(writer.log_dir) / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(writer, ckpt_dir=ckpt_dir, melody_dict=dec_mel_to_idx)

    # define model --------------------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    enc = base.Encoder(input_size=enc_mel_to_idx.dict_size(), z_size=z_size, **encoder_params)

    if not use_hier:
        # flat model ------------------------------------------------------------------------------
        dec = base.AnotherDecoder(output_size=dec_mel_to_idx.dict_size(), z_size=z_size, **decoder_params)
        seq_decoder = base.SimpleSeqDecoder(dec, z_size=z_size, device=device)
    else:
        # hier model ------------------------------------------------------------------------------
        assert conductor_params is not None and c_size is not None and n_subsequences is not None

        conductor = hier.Conductor(c_size=c_size, **conductor_params)
        dec = base.AnotherDecoder(output_size=dec_mel_to_idx.dict_size(), z_size=c_size, **decoder_params)
        seq_decoder = hier.HierarchicalSeqDecoder(conductor, dec, n_subsequences=n_subsequences, z_size=z_size,
                                                  device=device)

    model = base.Seq2Seq(enc, seq_decoder, sos_token=dec_mel_to_idx.get_sos_token()).to(device)

    # define optimizer ----------------------------------------------------------------------------
    opt = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    # define scheduler ----------------------------------------------------------------------------
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
                                                        factor=lr_scheduler_factor,
                                                        patience=lr_scheduler_patience,
                                                        verbose=True)

    # load checkpoint, if given -------------------------------------------------------------------
    step = 1
    if ckpt_path is not None:
        step = utils.load_ckpt(ckpt_path, model, opt, lr_scheduler, device) + 1
        print(f"Loaded checkpoint from \"{ckpt_path}\" start from step {step}.")

    # start train loop ----------------------------------------------------------------------------
    train(model, dl_train, opt, lr_scheduler, device, beta_settings, sampling_settings, free_bits,
          step=step, num_steps=num_steps, dl_eval=dl_eval,
          evaluate_interval=evaluate_interval,
          advanced_interval=advanced_interval,
          print_interval=print_interval)


@ex.config
def config():
    num_steps = 20_000

    batch_size = 2
    num_workers = 0

    ckpt_path = None

    evaluate_interval = 1000
    advanced_interval = 200
    print_interval = 200

    # melody_dir = r"C:\Users\yggdrasil\Studium Informatik\12Semester\Project\data\lmd_full_melody_128\0\0\\"
    train_dir = "../data/lmd_full/test"
    eval_dir = "../data/lmd_full/val"

    slice_bar = 8

    learning_rate = 1e-3
    lr_scheduler_factor = 0.5
    lr_scheduler_patience = 3

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
        "embed_size": 12,
        "hidden_size": 16,
        "n_layers": 2
    }

    use_hier = False

    c_size = None

    n_subsequences = None

    conductor_params = None

    decoder_params = {
        "embed_size": 12,
        "hidden_size": 64,
        "n_layers": 2
    }


@ex.automain
def main():
    run()


