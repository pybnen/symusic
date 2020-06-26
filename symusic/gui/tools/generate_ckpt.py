from pathlib import Path
import argparse
import os
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

import symusic.musicvae.datasets.melody_dataset as melody_dataset
import symusic.musicvae.models.trained as trained


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=0)

parser.add_argument("--ckpt_path", type=str,
                    help="Path to checkpoint, if not set environment variable CKPT_PATH is used.")
parser.add_argument("dataset_dir", type=str, help="Directory name containing dataset")
parser.add_argument("slice_bars", type=int, help="Melody length (in bars)")
parser.add_argument("outdir", type=str, help="Output directory")

args = parser.parse_args()


def criterion(outputs, target, mu, logvar):
    # kl_div per sample
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    # calculate r_loss per sample
    batch_size = outputs.size(0)
    r_losses = []
    for i in range(batch_size):
        r_loss = F.cross_entropy(outputs[i], target[i], reduction="mean")
        r_losses.append(r_loss)

    acc = torch.mean((outputs.argmax(dim=-1) == target).float(), dim=1)
    return torch.stack(r_losses, dim=0), kl_div, acc


def generate_data(model, data_loader, device):
    r_loss_arr = []
    kl_loss_arr = []
    acc_arr = []

    orig_melody_arr = []
    recon_melody_arr = []
    z_arr = []
    for x in data_loader:
        x = x.to(device)
        z, mu, logvar = model.encode_tensors(x)
        _, outputs = model.decode_to_tensors(z, length=x.shape[1])

        r_loss, kl_loss, acc = criterion(outputs, x, mu, logvar)

        # save metrics for all samples in lists
        r_loss_arr.extend(r_loss.tolist())
        kl_loss_arr.extend(kl_loss.tolist())
        acc_arr.extend(acc.tolist())

        orig_melody_arr.extend([m for m in model.tensors_to_melodies(x)])
        recon_melody_arr.extend([m for m in model.tensors_to_melodies(outputs.argmax(dim=-1))])
        z_arr.extend([sample_z.numpy() for sample_z in z.detach().cpu()])

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
    # check if arguments are valid
    ckpt_path = args.ckpt_path or os.getenv("CKPT_PATH")
    if not isinstance(ckpt_path, str) or not Path(ckpt_path).is_file():
        print("Path to checkpoint is not a file. '{}'".format(ckpt_path))
        return

    if not Path(args.dataset_dir).is_dir():
        print("Dataset directory name is no directory. '{}'".format(args.dataset_dir))
        return

    # generate outdir
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True)

    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    melody_dict = melody_dataset.MapMelodyToIndex()
    model = trained.TrainedModel(ckpt_path, melody_dict, device)

    # get dataset/loader
    dataset = melody_dataset.MelodyDataset(midi_dir=args.dataset_dir,
                                           max_melodies_per_sample=1,
                                           slice_bars=args.slice_bars,
                                           transforms=melody_dict, train=True)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)
    print(f"Dataset files: {len(dataset.midi_files)}")

    print("Generating data...")
    df = generate_data(model, data_loader, device)
    #  save data from reconstruction
    df.to_pickle(str(outdir / "recon_data.pkl"))

    print("Generating tsne projection...")
    generate_tsne(df)
    # save tsne data
    df.to_pickle(str(outdir / "tsne_data.pkl"))

    # save ckpt containing only model information (i.e. no optimizer)
    with open(ckpt_path, 'rb') as file:
        ckpt = torch.load(file, map_location=device)
        torch.save(dict(model_state_dict=ckpt["model_state_dict"],
                        model_kwargs=ckpt["model_kwargs"]), str(outdir / "model.pth"))
    print("Done")


if __name__ == "__main__":
    main()
