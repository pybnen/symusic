import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

from symusic.gui.globals import tsne_data_path
from symusic.gui.components import melody_result_view
from symusic.gui.utils import melody_to_graph, melody_to_audio
from symusic.gui.app import app

import pandas as pd
import plotly.express as px


tsne_data = pd.read_pickle(tsne_data_path)
color_dropdown_options = ["acc_arr", "kl_loss", "r_loss"]


def melody_space_view():
    return html.Div(className="melody-space melody-container", children=[
        html.H3("Melody Space"),
        html.P("Two dimensional t-sne projection of z encodings of train/eval melodies."),
        html.Div(className="row", children=[
            html.Div(className="columns five", children=dcc.Dropdown(
                id="melody-space-color-dropdown",
                options=[{"label": value, "value": value} for value in color_dropdown_options],
                value=color_dropdown_options[0]
            )),
        ]),
        html.Hr(),
        dcc.Graph(id="melody-space-graph")
    ])


layout = html.Div(children=[
    html.Div(className="row", children=[
        html.Div(className="columns twelve", children=melody_space_view()),
    ]),
    html.Div(className="row", style={"marginTop": "10px"}, children=[
        html.Div(className="columns six", children=melody_result_view("melody-space-recon")),
        html.Div(className="columns six", children=melody_result_view("melody-space-orig"))
    ])
])


@app.callback([Output({"type": "melody-result-graph", "id": "melody-space-recon"}, 'src'),
               Output({"type": "melody-result-player", "id": "melody-space-recon"}, 'src'),
               Output({"type": "melody-result-graph", "id": "melody-space-orig"}, 'src'),
               Output({"type": "melody-result-player", "id": "melody-space-orig"}, 'src'),],
              [Input('melody-space-graph', 'clickData')])
def display_click_data(click_data):
    if click_data is None:
        raise PreventUpdate

    idx = click_data["points"][0]["pointIndex"]
    orig_melody = tsne_data["orig_melody"][idx]
    recon_melody = tsne_data["recon_melody"][idx]

    return melody_to_graph(recon_melody), melody_to_audio(recon_melody),\
           melody_to_graph(orig_melody),melody_to_audio(orig_melody)


@app.callback(Output("melody-space-graph", "figure"),
              [Input("melody-space-color-dropdown", "value")])
def melody_space_update_view(color):
    return px.scatter(x=tsne_data["y_0"], y=tsne_data["y_1"], color=tsne_data[color], height=800)
