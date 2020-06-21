import json

import dash
from dash.dependencies import Input, Output, MATCH
from dash.exceptions import PreventUpdate

from symusic.gui.utils import midi_to_track_options, calc_max_start_bar, midi_to_melody, melody_to_graph, melody_to_audio
from symusic.gui.app import app


# TODO add store for selected melody
@app.callback(Output({"type": "track-dropdown", "id": MATCH}, "options"),
              [Input({"type": "midi-dropdown", "id": MATCH}, "value")])
def melody_selector_update_track_options(midi_path):
    return midi_to_track_options(midi_path)


@app.callback(Output({"type": "track-dropdown", "id": MATCH}, "value"),
              [Input({"type": "track-dropdown", "id": MATCH}, "options")])
def melody_selector_init_track_value(options):
    return options[0]["value"]


@app.callback([Output({"type": "start-bar-input", "id": MATCH}, "value"),
               Output({"type": "start-bar-input", "id": MATCH}, "max")],
              [Input({"type": "midi-dropdown", "id": MATCH}, "value"),
               Input({"type": "track-dropdown", "id": MATCH}, "value")])
def melody_selector_update_start_bar_input(midi_path, track_idx):
    max_start_bar = calc_max_start_bar(midi_path, track_idx)
    if max_start_bar == -1:
        raise PreventUpdate

    return 0, max_start_bar


@app.callback(Output({"type": "melody-pianoroll-graph", "id": MATCH}, "src"),
              [Input({"type": "midi-dropdown", "id": MATCH}, "value"),
               Input({"type": "track-dropdown", "id": MATCH}, "value"),
               Input({"type": "start-bar-input", "id": MATCH}, "value"),
               Input({"type": "pianoroll-view-type", "id": MATCH}, "value")])
def melody_selector_update_view(midi_path, track_idx, start_bar, view_type):
    view_selected = view_type == "Selected bars"
    # if start bar changed but view type is "whole melody", no need
    # to update anything
    ctx = dash.callback_context
    triggered_id = json.JSONDecoder().decode(ctx.triggered[0]["prop_id"].rsplit(".", 2)[0])
    if triggered_id["type"] == "start-bar-input" and not view_selected:
        raise PreventUpdate

    melody, _ = midi_to_melody(midi_path, track_idx, start_bar if view_selected else None)
    if melody is None:
        raise PreventUpdate

    return melody_to_graph(melody, start_bar=start_bar if view_selected else 0)


@app.callback(Output({"type": "melody-audio-player", "id": MATCH}, "src"),
              [Input({"type": "midi-dropdown", "id": MATCH}, "value"),
               Input({"type": "track-dropdown", "id": MATCH}, "value"),
               Input({"type": "start-bar-input", "id": MATCH}, "value")])
def melody_selector_update_player(midi_path, track_idx, start_bar):
    melody, midi_program = midi_to_melody(midi_path, track_idx, start_bar)
    if melody is None:
        raise PreventUpdate

    return melody_to_audio(melody, midi_program=midi_program)
