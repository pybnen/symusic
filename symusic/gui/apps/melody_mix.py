import torch
import numpy as np

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate

import symusic.gui.globals as globals
from symusic.gui.globals import midi_dropdown_options, MELODY_LENGTH
from symusic.gui.components import melody_selector, melody_result_view
from symusic.gui.utils import melody_to_graph, midi_to_melody, melody_to_audio
from symusic.gui.app import app

N_SLIDER = 10
Z_SLIDER_MIN = -5.0
Z_SLIDER_MAX = 5.0
Z_SLIDER_PRECISION = 4
Z_DIM_FORMAT = "{:+." + str(Z_SLIDER_PRECISION) + "f}"

# TODO instead of this global value use store,
#  use callback to only change one dimension if MATCH slider was changed (instead of ALL slider)
#  do this by adding store data as state (additional to output) to have curreent state an only update one dim
global_z = None
global_midi_program = 0


def melody_mixer():
    sliders = []
    for i in range(N_SLIDER):
        slider = html.Div(className="row", children=[
            html.Div(className="columns two", children=html.Label(children="Dim {}".format(i))),
            html.Div(className="columns eight", children=dcc.Slider(
                id={"type": "z-dim-slider", "id": i},
                min=Z_SLIDER_MIN,
                max=Z_SLIDER_MAX,
                step=10 ** -Z_SLIDER_PRECISION,
            )),
            html.Div(className="columns two", children=html.Label(
                id={"type": "z-dim-label", "id": i},
                children=Z_DIM_FORMAT.format(0.0)))
        ])
        sliders.append(slider)

    return html.Div(className="melody-mixer melody-container", children=[
        html.H3("Mixer"),
        html.Div(className="row", children=[
            html.Div(className="columns three",
                     children=html.Button(id="init-z-from-melody-btn", children="From melody")),
            html.Div(className="columns three", children=html.Button(id="init-z-random-btn", children="Random")),
            dcc.Input(id="melody-mix-temperature-input", type="number", value=1.0, className="columns three"),
            html.Div(className="columns three", children=html.Button(id="generate-mix-btn", children="Generate")),
        ]),
        html.Hr(),
        html.Div(className="container z-dim-container", children=sliders),
        dcc.Store(id='memory'),
    ])


layout = html.Div(children=[
    html.Div(className="row", children=[
        html.Div(className="columns six", children=melody_selector("melody-mix-1", midi_dropdown_options)),
    ]),
    html.Div(className="row", style={"marginTop": "10px"}, children=[
        html.Div(className="columns six", children=melody_mixer()),
        html.Div(className="columns six", children=melody_result_view("melody-mix-result")),
    ])
])


@app.callback(Output({"type": "z-dim-slider", "id": ALL}, "value"),
              [Input("init-z-random-btn", "n_clicks"),
               Input("init-z-from-melody-btn", "n_clicks")],
              [State({"type": "midi-dropdown", "id": "melody-mix-1"}, "value"),
               State({"type": "track-dropdown", "id": "melody-mix-1"}, "value"),
               State({"type": "start-bar-input", "id": "melody-mix-1"}, "value")])
def melody_mixer_init_z(n_clicks1, n_clicks2, midi_path, track_idx, start_bar):
    global global_z, global_midi_program

    ctx = dash.callback_context

    if not ctx.triggered or ctx.triggered[0]['prop_id'].rsplit('.', 2)[0] == "init-z-random-btn":
        z = np.round(np.random.randn(globals.model.z_size).clip(Z_SLIDER_MIN, Z_SLIDER_MAX), Z_SLIDER_PRECISION).tolist()
        global_midi_program = 0
    else:
        melody, midi_program = midi_to_melody(midi_path, track_idx, start_bar=start_bar)
        if melody is None:
            raise PreventUpdate
        # z, mu, logvar
        _, z, _ = globals.model.encode(melody.reshape(1, -1))
        z = z[0].detach().cpu().tolist()
        global_midi_program = midi_program
    global_z = z
    return z[:N_SLIDER]


@app.callback(Output({"type": "z-dim-label", "id": ALL}, "children"),
              [Input({"type": "z-dim-slider", "id": ALL}, "value")])
def melody_mixer_update_z_labels(values):
    return [Z_DIM_FORMAT.format(value) for value in values]


@app.callback([Output({"type": "melody-result-graph", "id": "melody-mix-result"}, "src"),
               Output({"type": "melody-result-player", "id": "melody-mix-result"}, "src")],
              [Input("generate-mix-btn", "n_clicks")],
              [State({"type": "z-dim-slider", "id": ALL}, "value"),
               State("melody-mix-temperature-input", "value")])
def melody_result_update_view(n_clicks, values, temperature):
    global global_z, global_midi_program

    if n_clicks is None or temperature <= 0:
        raise PreventUpdate

    if N_SLIDER < globals.model.z_size:
        values.extend(global_z[N_SLIDER:])

    z = torch.tensor(values).view(1, -1)
    # z = torch.tensor(global_z).view(1, -1)

    melodies, _ = globals.model.decode(z, length=MELODY_LENGTH, temperature=temperature)
    melody = melodies[0]
    return melody_to_graph(melody), melody_to_audio(melody, midi_program=global_midi_program)
