import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import numpy as np

from gui.globals import midi_dropdown_options
from gui.components import melody_selector
from gui.utils import midi_to_melody, melody_to_graph, melody_to_audio
from gui.app import app, model


def melody_walk_view():
    return html.Div(className="melody-walk melody-container", children=[
        html.H3("Melody Walk"),

        html.Div(className="row", children=[
            html.Div(className="columns two", children=html.Label(children="Interpolation steps:")),
            dcc.Input(id="num-steps-input", type="number", min=3, max=12, value=7, step=1, className="columns two"),
            dcc.Input(id="melody-walk-temperature-input", type="number", value=1.0, className="columns two"),
            html.Div(className="columns five", children=html.Button(id="interpolate-btn", children="Start walking"))
        ]),
        html.Hr(),
        html.Img(id="melody-walk-graph", className="pianoroll-graph",
                 src="./assets/images/pianoroll_placeholder.png"),
        html.Audio(id="melody-walk-player", className="audio-player", controls=True)
    ])


layout = html.Div(children=[
    html.Div(className="row", children=[
        html.Div(className="columns six", children=melody_selector("melody-walk-selector-1", midi_dropdown_options)),
        html.Div(className="columns six", children=melody_selector("melody-walk-selector-2", midi_dropdown_options)),
    ]),
    html.Div(className="row", style={"marginTop": "10px"}, children=[
        html.Div(className="columns twelve", children=melody_walk_view())
    ])
])


@app.callback([Output("melody-walk-graph", "src"), Output("melody-walk-player", "src")],
              [Input("interpolate-btn", "n_clicks")],
              [State("num-steps-input", "value"),
               State("melody-walk-temperature-input", "value"),
               State({"type": "midi-dropdown", "id": "melody-walk-selector-1"}, "value"),
               State({"type": "track-dropdown", "id": "melody-walk-selector-1"}, "value"),
               State({"type": "start-bar-input", "id": "melody-walk-selector-1"}, "value"),
               State({"type": "midi-dropdown", "id": "melody-walk-selector-2"}, "value"),
               State({"type": "track-dropdown", "id": "melody-walk-selector-2"}, "value"),
               State({"type": "start-bar-input", "id": "melody-walk-selector-2"}, "value")])
def melody_walk_interpolate(n_clicks, num_steps, temperature,
                            midi_path1, track_idx1, start_bar1,
                            midi_path2, track_idx2, start_bar2):
    if n_clicks is None or temperature <= 0:
        raise PreventUpdate

    melody1, midi_program1 = midi_to_melody(midi_path1, track_idx1, start_bar=start_bar1)
    melody2, _ = midi_to_melody(midi_path2, track_idx2, start_bar=start_bar2)
    if melody1 is None or melody2 is None or np.all(melody1 == melody2):
        raise PreventUpdate

    num_steps += 2
    interpolated_melodies, _ = model.interpolate(melody1, melody2, num_steps, length=32, temperature=temperature)
    melody = interpolated_melodies.reshape(-1)
    return melody_to_graph(melody), melody_to_audio(melody, midi_program1)
