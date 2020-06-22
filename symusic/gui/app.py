import dash
from glob import glob
from pathlib import Path

from symusic.gui.globals import audio_filesystem_dir

# create dir for temporary audio files
Path(audio_filesystem_dir).mkdir(exist_ok=True)

# delete previously generated audio files
files = list(glob(audio_filesystem_dir + "*"))
for f in files:
   try:
       Path(f).unlink()
   except OSError as e:
       print("Error: %s : %s" % (f, e.strerror))

app = dash.Dash(__name__)
server = app.server