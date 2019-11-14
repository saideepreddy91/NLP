import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State

from components.utils import *
from components.base_component import BaseComponent
from components.UserReviewComponent import display_mode_dropdown_options, FeatureDisplayMode

from analysis import model_analysis_news, global_vars
from analysis.global_vars import UI_STYLES, news_model
from analysis.misc import rgba


# input_initial_value = "Designer and Fulbright fellow Stanislas Chaillou has created a project at Harvard utilizing machine learning to explore the future of generative design, bias and architectural style. While studying AI and its potential integration into architectural practice, Chaillou built an entire generation methodology using Generative Adversarial Neural Networks (GANs). Chaillou's project investigates the future of AI through architectural style learning, and his work illustrates the profound impact of style on the composition of floor plans."
input_initial_value = None #'Test my input'

class NewsClassificationComponent(BaseComponent):
    """
    Page for News Classification
    """

    def render(self, props=None):
        stores = html.Div([
            dcc.Store(id='news-processed-text-input-data', storage_type='memory'),
            dcc.Store(id='news-prediction-output-data', storage_type='memory'),
            dcc.Store(id='target-category-data', storage_type='memory'),
        ])
        return Container([
            Grid([
                html.H2('Part 2: News Classification'),
                Row([
                    MultiColumn(2, html.Div([
                        html.H3('Input:'),
                        html.Button('Random', id='news-random-input-button', className='ui primary button random-input-button'),
                    ])),
                    MultiColumn(8, html.Div(
                        [
                            dcc.Textarea(
                                id='news-text-input',
                                className='autosize-textarea',
                                placeholder='Enter news content to predict category',
                                value=input_initial_value,
                                rows=3,
                            ),
                        ],
                        className='ui form',
                        style={
                            'padding': '8px',
                        },
                    )),
                    MultiColumn(6, html.Div(id='news-prediction-output-top')),
                ]),
                Row(MultiColumn(16, html.Div([
                    Grid([
                        MultiColumn(8, [
                            html.H3('Top three categories'),
                            html.Div(id='news-top-three-categories-div'),
                            html.Div(
                                dcc.Graph(
                                    id='news-prediction-output-pie-chart',
                                    config={
                                        'displayModeBar': False,
                                    })
                            ),
                        ]),
                        MultiColumn(8, [
                            html.H3('Feature Analysis', id='news-feature-analysis-header'),
                            html.Div(id='news-text-input-feature-highlight'),
                            html.Div(id='news-feature-strengths-bar-graph'),
                            dcc.Dropdown(
                                id='news-feature-display-mode',
                                options=[{'label': lb, 'value': value} for lb, value in display_mode_dropdown_options],
                                value=FeatureDisplayMode.prediction_contribution.value,
                                searchable=False,
                                clearable=False,
                                placeholder='Select Feature Display Type',
                            )
                        ]),
                    ], className='ui internally celled stackable grid'),
                ], className='ui piled segment'
                )), id='news-prediction-output-interactive-figures-segment', style={
                    'opacity': 0,
                }),
            ]),
            stores,
        ])

    def register_callbacks(self, app):
        @app.callback(
            Output('news-processed-text-input-data', 'data'),
            [Input('news-text-input', 'value')],
        )
        def on_news_raw_text_input(text_input):
            if text_input is None:
                text_input = ''
            processed_input = model_analysis_news.preprocess(text_input)
            return processed_input

        @app.callback(
            Output('target-category-data', 'data'),
            [
                Input('news-prediction-output-pie-chart', 'hoverData'),
                Input('news-prediction-output-data', 'data') # for default selection (top category)
            ,]
        )
        def on_hover_pie_chart_update_target_category_data(hoverData, prediction_data):
            if prediction_data is None:
                return None
            if hoverData is None:
                return prediction_data['top_categories_with_probs'][0][0]
            return hoverData['points'][0]['label']

        @app.callback(
            [
                Output('news-prediction-output-data', 'data'),
                Output('news-prediction-output-pie-chart', 'hoverData'),
            ],
            [
                Input('news-processed-text-input-data', 'data')
            ],
        )
        def on_news_text_input_changed_update_prediction(text_input):
            assert text_input is not None, 'Must not be None'
            if text_input == '':
                return None, None
            result = model_analysis_news.make_prediction(text_input)
            return result, None
            
        @app.callback(
            Output('news-prediction-output-top', 'children'),
            [Input('news-prediction-output-data', 'data')],
        )
        def update_top_prediction_output_div(result):
            if result is None: return None
            top_categories_with_probs = result['top_categories_with_probs']
            top_category, top_prob = top_categories_with_probs[0]
            div = model_analysis_news.make_top_prediction_result_div(top_category, top_prob)
            return div

        @app.callback(
            [
                Output('news-top-three-categories-div', 'children'),
                Output('news-prediction-output-pie-chart', 'figure'),
                Output('news-prediction-output-interactive-figures-segment', 'style'),
            ],
            [
                Input('news-prediction-output-data', 'data')
            ],
        )
        def on_prediction_output_update_pie_figure_and_top_three_categories(result):
            if result is None: return (None, {}, {'display': 'none'})
                
            top_categories_with_probs = result['top_categories_with_probs']

            pie_figure = model_analysis_news.make_prediction_probability_pie_chart(top_categories_with_probs)
            top_three_categories_div = model_analysis_news.make_top_three_predicted_categories(top_categories_with_probs)

            div_piled_section_style = {
                'display': 'block',
            }
            return top_three_categories_div, pie_figure, div_piled_section_style

        @app.callback(
            [
                Output('news-text-input-feature-highlight', 'children'),
                Output('news-feature-strengths-bar-graph', 'children'),
                Output('news-feature-analysis-header', 'children'),
            ],
            [
                Input('target-category-data', 'data'),
                Input('news-processed-text-input-data', 'data'),
                Input('news-feature-display-mode', 'value')
            ],
        )
        def on_news_text_input_changed_update_feature_highlight(target_category, text_input, display_mode):
            assert text_input is not None, 'Must not be None.'

            if target_category is None or text_input == '': return (None, None, 'Feature Analysis')
            news_highlighted_div, bar_figure = model_analysis_news.make_news_feature_highlights_bar_graph_div(
                text_input, 
                target_category, 
                FeatureDisplayMode(display_mode),
            )
            return (news_highlighted_div, dcc.Graph(figure=bar_figure, config={'displayModeBar': False}), f'Feature Analysis - {target_category}')

        @app.callback(
            Output('news-text-input', 'value'),
            [
                Input('news-random-input-button', 'n_clicks'),
            ],
        )
        def on_click_random_input_button(n_clicks):
            if n_clicks == 0: return ''
            return model_analysis_news.get_random_sample()