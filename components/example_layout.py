import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from components.utils import *
from components.base_component import BaseComponent

import numpy as np
import matplotlib.pyplot as plt

header_text = '''
## CSE 256: Explanation of NLP Classifier

This is written in Markdown format using `Markdown(...)` (or `dcc.Markdown(...)`).
'''


# Create multiple bars
def bar_graph_figure_from_values(values):
    return {
            'data': [
                {'x': np.arange(len(vals)), 'y': vals, 'type': 'bar', 'name': f'Range{i}'} for i, vals in enumerate(values)
            ],
            'layout': {
                'title': 'Dash Data Visualization'
            },
    }

class ExampleLayout(BaseComponent):

    def render(self, props=None):
        """
        Return the layout here
        """

        # example matplotlib fig
        plt.figure(figsize=(3,3))
        plt.plot(np.arange(10))
        fig = plt.gcf()

        return Container([
            Markdown(header_text),
            html.H3("Below are example of grid system, one row = 12 columms"),
            Row([
                MultiColumn(4, html.Div("4-column", style={'background-color': 'cyan'})),
                MultiColumn(8, html.Div("8-column", style={'background-color': 'cyan'})),
            ]),
            Row([
                MultiColumn(4, html.Div("4-column", style={'background-color': 'yellow'})),
                MultiColumn(4, html.Div("4-column", style={'background-color': 'yellow'})),
                MultiColumn(4, html.Div("4-column", style={'background-color': 'yellow'})),
            ]),
            Row([
                html.H4("Callback example, type textfield and it will map to the label below:")
            ]),
            Row([
                MultiColumn(3, [
                    Row([
                        TextField(id='tf1', value="What is that!", placeholder='Type something here'),
                    ]),
                    Row(html.Div("Sample callback", id='my-out-label'))
                ]),
            ]),
            Row([
                html.Div(None, id="text-splitted")
            ]),
            Row(html.H4("Callbacks can also map to plotly figure like below:")),
            Row([
                MultiColumn(3, [
                    Row(Slider('slider-1', min=0, max=20, step=1, value=5, label='Add more blue bars')),
                    Row(Slider('slider-2', min=0, max=20, step=1, value=5, label='Add more orange bars')),
                ]),
                MultiColumn(9, [
                    dcc.Graph(
                        id='example-graph',
                        figure=bar_graph_figure_from_values([np.arange(5), np.arange(5)])),
                ])
            ]),
            Row(html.H5("Below are converted from Matplotlib to plot.ly, but unlike creating graph from plotly directly which is responsive, we need to specify size in pixels before hand.")),
            Row([
                MultiColumn(6, MatplotlibFigure(id='mpl1', fig=fig, size_in_pixels=(300, 200))),
                MultiColumn(6, MatplotlibFigure(id='mpl2', fig=fig, size_in_pixels=(300, 200))),
            ]),

        ])

    def register_callbacks(self, app):
        """
        Register all callback here
        """
        @app.callback(
            [
                Output(component_id='my-out-label', component_property='children'),
                Output(component_id='text-splitted', component_property='children'),
            ],
            [
                Input(component_id='tf1', component_property='value'),
            ],
        )
        def update_text_from_textfield(input_value):            
            texts = html.Div([html.A([
                    w,
                ], className="ui basic label", id="label") for w in input_value.split()])
            return (f"You typed: {input_value}.",  texts)


        @app.callback(
            Output(component_id='example-graph', component_property='figure'),
            [Input(component_id='slider-1', component_property='value'),
            Input(component_id='slider-2', component_property='value')],
        )
        def update_figure_from_slider1(slider_1, slider_2):
            return bar_graph_figure_from_values([np.arange(slider_1), np.arange(slider_2)])