import os

assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "./assets"))

midi_assets = ["mel_2bar-bass.mid", "mel_2bar-mel.mid", "sample1.mid", "sample2.mid", "sample3.mid", "sample4.mid"]
midi_dropdown_options = {midi: os.path.join(assets_dir, "midis", midi) for midi in midi_assets}

ckpt_path = os.path.join(assets_dir, "ckpts/8bar_flat.pth")

tsne_data_path = os.path.join(assets_dir, "tsne/8bar_tsne_data_2.pkl")

audio_filesystem_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "./audio")) + "/"
audio_download_url = "/audio/"

STEPS_PER_BAR = 16
MELODY_LENGTH = 8 * STEPS_PER_BAR
