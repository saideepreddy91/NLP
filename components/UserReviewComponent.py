import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from components.utils import *
from components.base_component import BaseComponent
from analysis import model_analysis_user_review
from analysis.global_vars import user_review_model
from analysis.global_vars import UI_STYLES
from analysis.model_analysis_user_review import FeatureDisplayMode, map_to_new_low_and_high
from analysis.misc import rgba

import re
import numpy as np 

display_mode_dropdown_options = [
    ('Prediction Contribution', FeatureDisplayMode.prediction_contribution.value), 
    ('Feature Weight', FeatureDisplayMode.feature_weight.value),
    ('TF-IDF', FeatureDisplayMode.raw_feature_tfidf.value)
]

input_initial_value = "Came in for some good after a long day at work. Some of the food I wanted wasn't ready, and I understand that, but the employee Bianca refused to tell"
#'Insert your favorite review here!'

class UserReviewComponent(BaseComponent):
    """
    Part 1: User Review
    """
    def render(self, props=None):
        stores = html.Div([
            dcc.Store(id='text_input', storage_type='memory'),
            dcc.Store(id='sp_data', storage_type='memory'),
        ])
        return Container([
            Grid([
                html.H2('Part 1: User Review'),
                Row([
                    MultiColumn(2, [
                        html.H3('Input:'),
                        html.Button('Random', id='user-review-random-input-button', className='ui primary button random-input-button'),
                    ]),
                    MultiColumn(8, 
                        # TextField(id='input-text', value=input_initial_value, placeholder='Type review here')
                        html.Div(
                            dcc.Textarea(
                                id='input-text',
                                className='autosize-textarea',
                                placeholder='Enter user review to predict sentiment',
                                value=input_initial_value,
                                rows=3,
                            ), className='ui form', style={'padding': '8px'})
                        ),
                    MultiColumn(6, html.Div(id='prediction-output')),
                ]),
                Row([
                    MultiColumn(2, dcc.Markdown('### Tokens:')),
                    MultiColumn(14, html.Div(None, id="input-text-splitted")),
                ]),
                html.Div(className="ui divider"),
                Row([
                    MultiColumn(2, Row([
                        dcc.Markdown('### Features:'),
                    ])),
                    MultiColumn(14, Row([
                        Row(html.Div([
                            MultiColumn(5, dcc.Dropdown(
                                id='weight-display-mode',
                                options=[{'label': lb, 'value': value} for lb, value in display_mode_dropdown_options],
                                value=FeatureDisplayMode.prediction_contribution.value,
                                searchable=False,
                                clearable=False,
                                placeholder='Select Feature Display Type',
                            ))
                        ], style={'margin-bottom': '10px'})),
                        html.Div(None, id="sorted-features"),
                    ])),
                ]),
                Row([
                    MultiColumn(16, dcc.Graph(
                        id='coef-weight-graph',
                        config={ 
                            'displayModeBar': False,
                            # 'staticPlot': True,
                    })),
                ]),
                Row([
                    MultiColumn(16, [
                        html.Div(html.P('Click on one of the feature bars to see statistics', style={
                            'text-align': 'center',
                            'font-style': 'italic',
                            'text-color': 'silver',
                            'font-size': '18px',
                        }), className='ui compact info message center aligned', style={
                            'display':'table',
                            'margin': '0 auto',
                        })
                    ]),
                ], id='feature-in-context-hint'),
                Row([
                    MultiColumn(16, html.Div(id='feature-in-context-explaination-div')),
                ], id = 'feature-in-context-row', style= {'display': 'none'}),
                html.Div(className="ui divider"),
                Row([
                    MultiColumn(2, Row([
                        dcc.Markdown('### Sentiment Prediction:'),
                        html.H4('Show Top-K', id='top-k-slider-label'),
                        dcc.Slider(
                            id='sp-top-k-slider',
                            min=1,
                            max=20,
                            step=1,
                            value=3,
                            marks={
                                1: {'label': '1'},
                                20: {'label': '20'}
                            }
                        )
                    ])),
                    MultiColumn(14, Row([
                        dcc.Graph(
                            id='sentiment-prediction-graph',
                            config={
                                'displayModeBar': False,
                            }
                        ),
                    ])),
                ]),
                html.Div(className="ui divider"),
                Row([MultiColumn(16, html.H3("Information Value (IV) from the training set"))] +
                    [MultiColumn(8, dcc.Graph(figure=fig, config={'displayModeBar': False}))
                    for fig in model_analysis_user_review.get_information_values_for_top_positive_and_negative_features()]
                ),
            ]),
            stores,
        ])

    def register_callbacks(self, app):
    
        @app.callback(
            [
                Output('feature-in-context-explaination-div', 'children'),
                Output('feature-in-context-row', 'style'), # show row
                Output('feature-in-context-hint', 'style'), # hide hint
            ],
            [
                Input('coef-weight-graph', 'clickData'),
            ]
        )
        def on_detected_feature_click(clickData):
            feature = clickData['points'][0]['y']
            value = clickData['points'][0]['x']

            figure, metadata = model_analysis_user_review.part1_create_feature_in_context(feature, 3)

            positive_samples = metadata['positive_samples']
            negative_samples = metadata['negative_samples']
            positive_samples_pred_probs = metadata['positive_samples_pred_probs']
            negative_samples_pred_probs = metadata['negative_samples_pred_probs']

            pos_color = rgba(*UI_STYLES.POSITIVE_COLOR, 0.7)
            neg_color = rgba(*UI_STYLES.NEGATIVE_COLOR, 0.7)
            feature_color = pos_color if value > 0 else neg_color

            def get_formatted_list_from_samples_and_probs(samples, probs, highlight_color):
                def get_formatted_list_item(sentence):
                    # Here we split sentence by 'feature', then join it back with Span of 'feature', to show highlighting effects.
                    non_match_components = re.compile(f'\\b{feature}\\b').split(sentence.lower()) # lowercase here to make things simpler                
                    inserted_span_feature_in_between_result = ['"']
                    for comp in non_match_components:
                        inserted_span_feature_in_between_result.append(comp)
                        inserted_span_feature_in_between_result.append(html.Span(feature, className='feature-tag-highlighted', style={'border-radius': 6, 'background-color': highlight_color}))
                    inserted_span_feature_in_between_result.pop() # remove last one
                    inserted_span_feature_in_between_result.append('"')
                    return inserted_span_feature_in_between_result

                return html.Ul([
                    html.Li(get_formatted_list_item(sentence))
                    for sentence, prob in zip(samples, probs)
                ])

            sentence_samples_div_children = []
            if len(positive_samples) > 0:
                sentence_samples_div_children.append(html.H5('Positive Samples:'))
                sentence_samples_div_children.append(get_formatted_list_from_samples_and_probs(positive_samples, positive_samples_pred_probs, feature_color))

            if len(negative_samples) > 0:
                sentence_samples_div_children.append(html.H5('Negative Samples:'))
                sentence_samples_div_children.append(get_formatted_list_from_samples_and_probs(negative_samples, negative_samples_pred_probs, feature_color))

            explaination_div = Grid([
                MultiColumn(8, html.Div([
                        html.H4([
                            "Statistics of ",
                            html.Span(feature, style={
                                'border-radius': 8,
                                'background-color': feature_color,
                            }),
                        ], className='ui header'),
                        dcc.Markdown(metadata['md_explaination']),
                        html.Div(sentence_samples_div_children, id='sentence-samples-div'),
                    ], 
                    className='ui piled compact segment')),
                MultiColumn(8, [
                    dcc.Graph(
                        figure=figure,
                        id='feature-in-context-pie-graph',
                        config={
                            'displayModeBar': False,
                        })
                ])
            ])
            return explaination_div, { 'display': 'block' }, { 'display': 'none' }
            

        ########################################
        # MEMORY DATA 
        ########################################
        @app.callback(
            Output('text_input', 'data'),
            [Input('input-text', component_property='value')])
        def on_raw_input_text_update(raw_input_text):
            # existing_text = data.get('preprocessed_input', '')
            new_text = model_analysis_user_review.preprocess(raw_input_text)
            return new_text

        @app.callback(
            Output(component_id='input-text-splitted', component_property='children'),
            [
                Input(component_id='input-text', component_property='value'),
            ],
        )
        def on_enter_input_text(input_text):
            if input_text == '':
                return html.Div('Empty input text.')

            input_text = model_analysis_user_review.preprocess(input_text)
            feature_names_set = user_review_model.feature_names_set

            splitted_text_tags = []
            tokens = input_text.split()
            for w in (tokens):
                is_unseen = w not in feature_names_set
                new_tag = None
                if is_unseen:
                    html_data = {'data-tooltip': f"'{w}' is not in token list", 'data-position': "top center"}
                    new_tag = html.Div(
                        children=html.Div(w, className='ui basic unseen label'),
                        style={'display': 'inline-block'},
                        **html_data,
                    )
                else:
                    new_tag = html.Div(w, className='ui basic label')

                splitted_text_tags.append(new_tag)

            return html.Div(splitted_text_tags)

        @app.callback(Output('top-k-slider-label', 'children'), [Input('sp-top-k-slider', 'value')])
        def update_top_k_slider_label(top_k_value):
            return f'Show Top-{top_k_value} features'

        @app.callback(
            Output('sentiment-prediction-graph', 'figure'),
            [
                Input('sp_data', 'data'),
                Input('sp-top-k-slider', 'value')
            ]
        )
        def on_sp_data_update(sp_data, top_k_value):
            if sp_data is None:
                return {}
            return model_analysis_user_review.part1_create_sentiment_prediction_figure(sp_data, top_k=top_k_value)

        @app.callback(
            [
                Output('coef-weight-graph', 'figure'),
                Output('sorted-features', 'children'),
                Output('prediction-output', 'children'),
                Output('sp_data', 'data'),
            ],
            [
                Input('text_input', 'data'),
                Input('weight-display-mode', 'value')
            ]
        )
        def on_enter_input_text_show_weights(text_input, display_mode):
            preprocessed_input = text_input

            try:
                display_mode = FeatureDisplayMode(display_mode)
                result = model_analysis_user_review.part1_analyze_coefficients(preprocessed_input, display_mode=display_mode)
            except Exception as error:
                print("Error:", error, "with args:", error.args)
                return {}, html.Div(error.args[0]), None, None

            figure_fc = result['figure_feature_contribution']
            sp_data = result['sp_data']

            features = result['human_sorted_features']
            values = result['human_sorted_values']
            relative_feature_strengths = result['relative_feature_strengths']

            # DETECTED FEATURES
            feature_tags = []
            for i in range(len(features)):
                f = features[i]
                v = values[i]
                feature_strength = relative_feature_strengths[i]

                className = f'ui {UI_STYLES.POSITIVE_COLOR_CLASSNAME} label' if v > 0 else f'ui {UI_STYLES.NEGATIVE_COLOR_CLASSNAME} label'
                html_data = {'data-tooltip': f"{np.round(v, 2)}", 'data-position': "top center"}

                opacity = np.round(feature_strength, 2)
                font_size = np.round(map_to_new_low_and_high(feature_strength, 0, 1, 12, 20), 1)
                tag = html.Div(f, 
                    className=className, 
                    style={
                    'opacity': opacity,
                    'fontSize': font_size,
                    },
                )
                wrapper_div = html.Div(tag, style={
                    'id': f'detected-feature-{i}',
                    'display': 'inline-block', 
                    'margin-right': '1.7px',
                    }, 
                    **html_data)
                feature_tags.append(wrapper_div)

            detected_feature_tags_div = html.Div(feature_tags)
            
            pred_x = result['pred_x']
            prob_x = result['prob_x']


            # USER SENTIMENT ICON
            sentiment_icon_class = None
            sentiment_label = None
            if prob_x[1] >= 0.7:
                sentiment_icon_class = 'smile' 
                sentiment_label = 'POSITIVE'
            elif prob_x[1] >= 0.4:
                sentiment_icon_class = 'meh'
                sentiment_label = 'MEH'
            else:
                sentiment_icon_class = 'frown'
                sentiment_label = 'NEGATIVE'

            user_sentiment_icon = html.I(className=f'{sentiment_icon_class} outline icon')

            # PREDICTION OUTPUT BADGE
            color_class = UI_STYLES.POSITIVE_COLOR_CLASSNAME if pred_x else UI_STYLES.NEGATIVE_COLOR_CLASSNAME
            prediction_div_classname = f'ui {color_class} statistic'
            prediction_output_div = html.Div([
                html.Div([
                    f'{np.round(max(prob_x) * 100, 1)}',
                    html.I(className='mini percent icon'),
                    ], className='value'),
                html.Div([
                    sentiment_label,
                    user_sentiment_icon,
                    ], className='label'),
            ], 
            className=prediction_div_classname,
            style={
                'display': 'block',
            })
            return figure_fc, detected_feature_tags_div, prediction_output_div, sp_data
        @app.callback(
            Output('input-text', 'value'),
            [
                Input('user-review-random-input-button', 'n_clicks'),
            ],
        )
        def on_click_random_input_button(n_clicks):
            if n_clicks == 0: return ''
            return model_analysis_user_review.get_random_sample()