import os
from glob import glob
from pathlib import Path
import pandas as pd

import torch

from symusic.musicvae.models.trained import TrainedModel
from symusic.musicvae.datasets.melody_dataset import MapMelodyToIndex


def setup(ckpt_dir):
    global tsne_data, model

    print(f"Load ckpt \"{Path(ckpt_dir).stem}\"")

    ckpt_path = Path(ckpt_dir) / "model.pth"
    tsne_data_path = Path(ckpt_dir) / "tsne_data.pkl"
    assert ckpt_path.is_file() and tsne_data_path.is_file()

    # set tsne data
    tsne_data = pd.read_pickle(str(tsne_data_path))

    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    melody_dict = MapMelodyToIndex()
    model = TrainedModel(str(ckpt_path), melody_dict, device, use_sample_decoder=True)


assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "./assets"))

midi_assets = ["sample1.mid", "sample2.mid", "sample3.mid", "sample4.mid"]
midi_dropdown_options = {midi: os.path.join(assets_dir, "midis", midi) for midi in midi_assets}

audio_filesystem_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "./tmp_audio")) + "/"
audio_download_url = "/audio/"

STEPS_PER_BAR = 16
MELODY_LENGTH = 8 * STEPS_PER_BAR

tsne_data = None
model = None