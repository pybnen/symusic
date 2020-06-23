# MusicVAE Visualisation

Start the [Dash](https://plotly.com/dash/)  server:

```bash
python -m symusic.gui.index /path/to/checkpoint/dir
```

## Checkpoint

Download a checkpoint from my train run [here](link_to_checkpoint).

To create your own checkpoint, first you need to train a model (see [/symusic/musicvae](/symusic/musicvae/README.md)).
The resulting checkpoint file contains the learned weights but also information needed to continue training, like
the state of the optimizer.
To get rid of this information and create a t-sne embedding of the test dataset execute the following command:

```bash
python -m symusic.gui.tools.generate_ckpt --ckpt_path=/path/to/checkpoint/dir /path/to/test/dataset melody_length /path/to/output/
```