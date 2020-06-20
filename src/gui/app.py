import dash
import torch
from glob import glob
from pathlib import Path
from music_vae.models.trained import TrainedModel
from music_vae.datasets.melody_dataset import MapMelodyToIndex
from gui.globals import ckpt_path, audio_dir


files = list(glob(audio_dir + "*"))
for f in files:
    try:
        Path(f).unlink()
    except OSError as e:
        print("Error: %s : %s" % (f, e.strerror))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
melody_dict = MapMelodyToIndex()
model = TrainedModel(ckpt_path, melody_dict, device)

app = dash.Dash(__name__)
server = app.server