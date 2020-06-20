import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from flask import send_from_directory

from gui.globals import audio_dir
from gui.app import app, server
from gui.apps import melody_mix, melody_walk, melody_space
from gui import callbacks

base_layout = html.Div(id="page-container", className="container", children=[
    # dcc.Location(id="url", refresh=False),
    html.H2("MusicVAE", className="page-title"),
    dcc.Tabs(id='header-tabs', value="mix", children=[
        dcc.Tab(label='Melody Mix', value='mix'),
        dcc.Tab(label='Melody Walk', value='walk'),
        dcc.Tab(label='Melody Space', value='space'),
    ]),
    html.Div(id="page-content")
])

app.layout = base_layout
page_layouts = {
    "mix": melody_mix.layout,
    "walk": melody_walk.layout,
    "space": melody_space.layout,
}

app.validation_layout = html.Div([
    base_layout,
    *page_layouts.values()
])


@app.callback(Output('page-content', 'children'),
              [Input('header-tabs', 'value')])
def render_content(tab):
    return page_layouts[tab]

# @app.callback(Output('page-content', 'children'),
#               [Input("url", "pathname")])
# def display_page(pathname):
#     if pathname is None:
#         return None
#     pathname.strip("/")
#     if pathname in page_layouts:
#         return page_layouts[pathname]
#
#     return "404"


@server.route("/audio/<path:path>")
def download(path):
    """Serve a file from the upload directory."""
    return send_from_directory(audio_dir, path)


if __name__ == "__main__":
    app.run_server(debug=True, dev_tools_hot_reload=False)
