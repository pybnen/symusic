
midi_assets = ["mel_2bar-bass.mid", "mel_2bar-mel.mid", "sample1.mid", "sample2.mid", "sample3.mid", "sample4.mid"]
midi_dropdown_options = {midi: "./assets/midis/{}".format(midi) for midi in midi_assets}

ckpt_path = "./assets/ckpts/8bar_flat.pth"

tsne_data_path = "./assets/tsne/8bar_tsne_data_2.pkl"

audio_dir = "./audio/"

z_size = 512

STEPS_PER_BAR = 16
MELODY_LENGTH = 8 * STEPS_PER_BAR
