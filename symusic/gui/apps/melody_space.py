import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

import symusic.gui.globals as globals
from symusic.gui.components import melody_result_view
from symusic.gui.utils import melody_to_graph, melody_to_audio
from symusic.gui.app import app
import plotly.graph_objects as go

def melody_space_view():
    return html.Div(className="melody-space melody-container", children=[
        html.H3("Melody Space"),
        html.P("Two dimensional t-sne projection of z encodings of melodies from a test set."),
        html.Button(id="hidden_btn", style=dict(display="None")),
        html.Hr(),
        dcc.Graph(id="melody-space-graph", style=dict(width=800))
    ])


layout = html.Div(children=[
    html.Div(className="row", children=[
        html.Div(className="columns twelve", children=melody_space_view()),
    ]),
    html.Div(className="row", style={"marginTop": "10px"}, children=[
        html.Div(className="columns six", children=melody_result_view("melody-space-recon", h3="Reconstruction")),
        html.Div(className="columns six", children=melody_result_view("melody-space-orig", h3="Original"))
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
    orig_melody = globals.tsne_data["orig_melody"][idx]
    recon_melody = globals.tsne_data["recon_melody"][idx]

    return melody_to_graph(recon_melody), melody_to_audio(recon_melody),\
           melody_to_graph(orig_melody),melody_to_audio(orig_melody)


@app.callback(Output("melody-space-graph", "figure"),
              [Input("hidden_btn", "n_clicks")])
def melody_space_update_view(_):
    fig = go.Figure(data=go.Scatter(x=globals.tsne_data["y_0"],
                                     y=globals.tsne_data["y_1"],
                                     mode='markers',
                                     marker=dict(size=3)))
    fig.update_layout(autosize=False, width=800, height=800,  margin=dict(
        l=0,
        r=0,
        b=0,
        t=0,
        pad=4
    ))
    return fig

