<img src="./symusic-logo.png" height="75" alt="symusic">

---

**symusic** is as good as any name trying to shorten the term *symbolic music generation*.
In this repository I am trying to reimplement the *MusicVAE* model, introduced in the paper
[A Hierarchical Latent Vector Model for Learning Long-Term Structure in Music](https://arxiv.org/abs/1803.05428),
an implementation of the model can be found in the [magenta](https://github.com/magenta/magenta/tree/master/magenta/models/music_vae)
repository.

## Installation

In order to run the visualisation or try to train a model, first clone the repository:

```bash
git clone https://github.com/pybnen/symusic.git
```

### Use `environment.yml` 

To install the required dependencies create a new conda environment with the provided `environment.yml` file:

```bash
conda env create -f environment.yml
```

This should create a new environment called `symusic`, which can be activated with:

```bash
conda activate symusic
```

### Manual installation

If the `environment.yml` didn't work you can try to create a conda environment manually with the following commands:

```bash
conda create -n symusic python=3.7.6
conda activate symusic
pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install sacred==0.8.1
pip install pretty-midi==0.2.8
pip install tensorboard==1.15.0
pip install matplotlib==3.1.1
pip install pypianoroll==0.5.2
pip install dash==1.12.0
pip install pandas==0.25.3
pip install scikit-learn==0.23.1
pip install midi2audio
```

### Additional dependencies

To be able to convert MIDI files to `.wav` files, [FluidSynth](http://www.fluidsynth.org/) is required.

### Notes/Problems

[midi2audio](https://github.com/bzamecnik/midi2audio) is used to call FluidSynth commands via python,
the package didn't work on my local machine, which I think is a windows only problem,
so I provided a hotfix for this problem: https://github.com/pybnen/midi2audio.


### Run tests

In order to run the provided tests,
the following commands have to be executed at the base directory of the repository:

```bash
# install symusic, to make the package available to the tests
pip install -e .

# install magenta at the state I used during development.
pip install -e git+https://github.com/tensorflow/magenta.git@8e5da380a1cd39d14c5bcbbae0691e7983f833fa#egg=magenta

# install pytest to run the tests
pip install pytest
```

After this you should be able to run the tests:

```bash
# go to tests directory
cd tests

# run tests
pytest
```

**Why need magenta and why this commit?**

This is the commit I used during development, the current master branch of [magenta](https://github.com/magenta/magenta) might not work.
Magenta is only needed for the tests, to see if my melody extraction does the same as theirs.

**Then why not use their melody extraction?**

When I started my implementation, the melody extraction was inside the magenta repository, and I didn't want all the
dependencies of the magenta project.

I have only recently seen that there is another repository [note-seq](https://github.com/magenta/note-seq), which could
have been used instead. Alas, it is to late.


## Train a Model

[Sacred](https://github.com/IDSIA/sacred) is used to manage experiments,
in order to start training execute the following command at the base directory:

```bash
python -m symusic.musicvae.train with /path/to/config -F /path/to/log/runs
```

See [/symusic/musicvae](/symusic/musicvae/README.md) for a list of the configurable parameter.

### Dataset

I used the [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/). You can simply download the files and split
in train/vaild/test sets. The following commands do exactly this, from the base directory execute:

```bash
# change to data directory
cd data

# run script
./download_and_split_dataset.sh
```

This should create a `lmd_full_split` directory with subdirectories `train`, `val` and `test`
containing the MIDI files from the Lakh MIDI Dataset.

## Start the Visualisation

The visualisation is implemented as a [Dash](https://plotly.com/dash/) application. To start the server run:

```bash
python -m symusic.gui.index /path/to/checkpoint/dir
```

### Checkpoint

Download a checkpoint from my train run [here](https://drive.google.com/file/d/1K7Yy8nJifR5DjfQeMbyy0LECf4_B-TsS/view?usp=sharing).

See [/symusic/musicvae](/symusic/gui/README.md) for a short description on how to create a checkpoint for the
visualisation.

