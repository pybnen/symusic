import dash_core_components as dcc
import dash_html_components as html


def melody_selector(id, midi_dropdown_options):
    return html.Div(className="melody-selector melody-container", children=[
        html.H3("Melody Selector"),

        html.Div(className="row", children=[
            html.Div(className="columns five", children=dcc.Dropdown(
                id={"type": "midi-dropdown", "id": id},
                options=[{"label": key, "value": value} for key, value in midi_dropdown_options.items()],
                value=list(midi_dropdown_options.values())[0]
            )),
            html.Div(className="columns five", children=dcc.Dropdown(id={"type": "track-dropdown", "id": id})),
            dcc.Input(id={"type": "start-bar-input", "id": id}, type="number", min=0, value=0, step=1,
                      className="columns two")
        ]),
        html.Hr(),
        dcc.RadioItems(
            id={"type": "pianoroll-view-type", "id": id},
            options=[{'label': i, 'value': i} for i in ['Whole melody', 'Selected bars']],
            value='Selected bars',
            labelStyle={'display': 'inline-block'}
        ),
        html.Img(id={"type": "melody-pianoroll-graph", "id": id}, className="pianoroll-graph",
                 src="./assets/images/pianoroll_placeholder.png"),
        html.Audio(id={"type": "melody-audio-player", "id": id}, className="audio-player", controls=True)
    ])


def melody_result_view(id, show_original=False):
    original_view = []
    class_name = "melody-result melody-container"
    placeolder_url = "./assets/images/pianoroll_placeholder.png"
    if show_original:
        class_name += " show-original"
        placeolder_url = "./assets/images/pianoroll_placeholder_flat.png"
        original_view = [
            html.Label("Original:", className="melody-result-label"),
            html.Img(id={"type": "melody-original-graph", "id": id}, className="pianoroll-graph", src=placeolder_url),
            html.Audio(id={"type": "melody-original-player", "id": id}, className="audio-player", controls=True)
        ]

    return html.Div(className=class_name, children=[
        html.H3("Melody Result"),
        html.Hr(),
        html.Label("Reconstruction:", className="melody-result-label"),
        html.Img(id={"type": "melody-result-graph", "id": id}, className="pianoroll-graph", src=placeolder_url),
        html.Audio(id={"type": "melody-result-player", "id": id}, className="audio-player", controls=True),
        *original_view
    ])