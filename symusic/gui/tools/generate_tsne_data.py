from pathlib import Path
import argparse

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

from smg.datasets.melody_dataset import FixedLengthMelodyDataset, MelodyEncode
from smg.ingredients.data import dataset_train_valid_split
from smg.models.music_vae.music_vae import MusicVAE
from smg.models.music_vae.trained_model import TrainedModel
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser()
parser.add_argument("--n_classes", type=int, default=90)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=0)

parser.add_argument("--ckpt_path", type=str,
                    help="Path to checkpoint, if not set environment variable CKPT_PATH is used.")
parser.add_argument("--temperature", type=float, default=1.0,
                    help="Used for sampling next input from current output, lower temperature favours best guess, very low temp would basically result in argmax.")  # noqa

parser.add_argument("--valid_split", type=float, default=0.2,
                    help="Split ratio between train/eval set, set to 0.0 if not split should be made.")  # noqa
parser.add_argument("--use_train", action="store_true", default=False,
                    help="Use train set instead of eval set, applys only if valid split is given, default False")  # noqa
parser.add_argument("dataset_dirname", type=str, help="Directory name containing dataset")

args = parser.parse_args()


# TODO factor into utils
def load_model_from_ckpt(ckpt_path, device):
    with open(ckpt_path, 'rb') as file:
        ckpt = torch.load(file, map_location=device)
        model_ckpt = ckpt['model']
    return MusicVAE.load_from_ckpt(model_ckpt).to(device)


def calc_loss(x_hat, mu, sigma, x_target):
    variance = sigma.pow(2)
    log_variance = variance.log()

    # kl_div per sample
    kl_div = -0.5 * torch.mean(1 + log_variance - mu.pow(2) - variance, dim=1)

    x_target = x_target.argmax(dim=-1)
    r_losses = []
    for i in range(x_hat.size(0)):
        r_loss = F.cross_entropy(x_hat[i], x_target[i], reduction="mean")
        r_losses.append(r_loss)

    acc = torch.mean((x_hat.argmax(dim=-1) == x_target).float(), dim=1)
    return torch.stack(r_losses, dim=0), kl_div, acc


def generate_data(model, data_loader, temperature=1.0):
    r_loss_arr = []
    kl_loss_arr = []
    acc_arr = []

    orig_melody_arr = []
    recon_melody_arr = []
    z_arr = []
    for x in tqdm(data_loader):
        z, mu, sigma = model.encode_tensors(x)
        _, output_logits = model.decode_to_tensors(z, length=x.shape[1], temperature=temperature)

        r_loss, kl_loss, acc = calc_loss(output_logits, mu, sigma, x)

        # save metrics for all samples in lists
        r_loss_arr.extend(r_loss.tolist())
        kl_loss_arr.extend(kl_loss.tolist())
        acc_arr.extend(acc.tolist())

        orig_melody_arr.extend([m for m in model.tensors_to_melodies(x.argmax(dim=-1))])
        recon_melody_arr.extend([m for m in model.tensors_to_melodies(output_logits.argmax(dim=-1))])
        z_arr.extend([sample_z.numpy() for sample_z in z])

    df = pd.DataFrame(dict(orig_melody=orig_melody_arr,
                           recon_melody=recon_melody_arr,
                           z=z_arr,
                           r_loss=r_loss_arr,
                           kl_loss=kl_loss_arr,
                           acc=acc_arr))
    return df


def generate_tsne(df):
    tsne = TSNE(n_components=2, perplexity=70)

    y = tsne.fit_transform(np.array([z for z in df["z"]]))
    df["y_0"] = y[:, 0]
    df["y_1"] = y[:, 1]


def main():
    ckpt_path = args.ckpt_path or os.getenv("CKPT_PATH")
    if not isinstance(ckpt_path, str) or not Path(ckpt_path).is_file():
        print("Path to checkpoint is not a file. '{}'".format(ckpt_path))
        return

    if not Path(args.dataset_dirname).is_dir():
        print("Dataset directory name is no directory. '{}'".format(args.dataset_dirname))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrainedModel(ckpt_path, device)

    dataset = FixedLengthMelodyDataset(melody_dir=args.dataset_dirname,
                                       transforms=MelodyEncode(args.n_classes, num_special_events=0))
    if args.valid_split > 0.0:
        ds_train, dataset = dataset_train_valid_split(dataset, valid_split=args.valid_split)
        if args.use_train:
            dataset = ds_train

    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                             shuffle=False, drop_last=False)
    print("Dataset length {}.".format(len(dataset)))

    print("Generating data...")
    df = generate_data(model, data_loader, temperature=args.temperature)
    df.to_pickle("tsne_data_wo_tsne.pkl")
    print("Done created data with shape {}".format(df.shape))

    print("Generating tsne projection...")
    generate_tsne(df)
    df.to_pickle("tsne_data.pkl")
    print("Done")


if __name__ == "__main__":
    main()
