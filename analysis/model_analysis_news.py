import plotly.graph_objs as go
import dash_html_components as html

import numpy as np
import re 

from analysis.misc import rgba, get_relative_strengths, hex_string_to_rgb
from analysis.global_vars import news_model as model
from analysis.global_vars import UI_STYLES
from analysis.model_analysis_user_review import FeatureDisplayMode

def preprocess(raw_input_text):
    text = model.fv_text_preprocessor(raw_input_text)
    text = ' '.join(model.fv_text_tokenize(text))
    return text

def make_prediction(sentence):
    """Predict (already-preprocessed) news"""
    assert sentence is not None and sentence != '', "Invalid sentence"

    fv = model.fv
    clf = model.clf

    x = fv.transform([sentence])

    prob_x = clf.predict_proba(x)[0]

    probs_sorted = prob_x.argsort()[::-1]
    
    # (category, prob), sorted from max to min prob
    top_categories_with_probs = [(clf.classes_[i], prob_x[i]) for i in probs_sorted]

    return dict(
        top_categories_with_probs=top_categories_with_probs
    )

def make_prediction_probability_pie_chart(top_categories_with_probs):
    categories = [cat for cat, _ in top_categories_with_probs]
    category_probs = [prob for _, prob in top_categories_with_probs]

    pie_trace = go.Pie(
        labels=categories,
        values=category_probs,
        hoverinfo='label+percent',
        textinfo='none',
        textfont=dict(size=14),
        marker=dict(
            colors=[model.category_to_colors[cat] for cat in categories],
            # colorscale='haline',
        ),
    )

    pie_layout = go.Layout(
        title=f"Probabilities by categories",
    )
    
    pie_figure = go.Figure(data=[pie_trace], layout=pie_layout)
    return pie_figure

def make_top_three_predicted_categories(top_categories_with_probs):
    TOP_K = 3
    top_k_categories = top_categories_with_probs[:TOP_K]
    # scales = get_relative_strengths([prob for _, prob in top_k_categories], 1, 1.2)
    return html.Div([
        html.Div(cat, className='ui news-top-category label', style={
            'background-color': model.category_to_colors[cat],
            # 'transform': f'scale({scales[i]})'
        }, **{'data-tooltip': f"{np.round(prob*100, 1)}%", 'data-position': "top center", 'data-inverted': ''})
        for i, (cat, prob) in enumerate(top_k_categories)
    ])
    
def get_random_sample():
    n_test_data = len(model.news_data.test_data)
    ind = np.random.randint(n_test_data)
    return model.news_data.test_data[ind]

def make_news_feature_highlights_bar_graph_div(
    text_input, 
    target_category, 
    display_mode=FeatureDisplayMode.prediction_contribution, 
    highlight_top_k_features=10
    ):

    clf = model.clf
    fv = model.fv

    category_index = list(clf.classes_).index(target_category)
    category_coefficients = clf.coef_[category_index]

    x = fv.transform([text_input]).toarray().flatten()
    coef_feature_products = category_coefficients * x

    nonzero_inds = x.nonzero()[0]

    nonzero_strength_values = None
    figure_title = display_mode.title
    if display_mode == FeatureDisplayMode.prediction_contribution:
        nonzero_strength_values = coef_feature_products[nonzero_inds]
    elif display_mode == FeatureDisplayMode.feature_weight:
        nonzero_strength_values = category_coefficients[nonzero_inds]
    elif display_mode == FeatureDisplayMode.raw_feature_tfidf:
        nonzero_strength_values = x[nonzero_inds]
    else:
        raise ValueError("Invalid `display_mode` type.")

    argsort_nonzero_strengh_values = nonzero_strength_values.argsort()[::-1]

    # sort top to bottom values, limit to top K
    nonzero_strength_values = nonzero_strength_values[argsort_nonzero_strengh_values][:highlight_top_k_features]
    nonzero_inds = nonzero_inds[argsort_nonzero_strengh_values][:highlight_top_k_features]

    # feature names for these inds
    nonzero_features = [model.feature_names[i] for i in nonzero_inds]

    all_feature_strings = '|'.join(nonzero_features)
    regex_pattern = f"\\b{all_feature_strings}\\b"
    regex_compiled = re.compile(regex_pattern)
    matches = regex_compiled.findall(text_input) # m - 1
    neighbors = regex_compiled.split(text_input) # m

    highlight_color = model.category_to_colors[target_category]
    highlight_color_rgb = hex_string_to_rgb(highlight_color)
    div_children = [neighbors[0]]
    for i in range(len(matches)):
        span = html.Span(matches[i], 
            className='news-feature-tag-highlighted', 
            style={'background-color': rgba(*highlight_color_rgb, 0.8)}
        )
        div_children.append(span)
        div_children.append(neighbors[i+1])


    feature_strengths_trace = go.Bar(
        x = nonzero_features,
        y = nonzero_strength_values,
        name = target_category,
        marker = {
            'color': rgba(*highlight_color_rgb, 0.7),
            'opacity': 0.7,
            'line': {
                'color': highlight_color,
                'width': 2,
            }
        })

    bar_graph_feature_contribution = {
        'data': [
            feature_strengths_trace
        ],
        'layout': go.Layout(
            title=figure_title,
            yaxis=dict(
                automargin=True,
                fixedrange=True,
            ),
            xaxis=dict(
                automargin=True,
                fixedrange=True,
            ),
        ),
    }
    
    return html.Div(div_children, id='news-feature-tag-sentence-wrapper-div'), bar_graph_feature_contribution
    

def make_top_prediction_result_div(top_category, top_prob):
    prediction_output_div = html.Div([
        html.Div([
            top_category,
        ], className='value', style={'color': model.category_to_colors[top_category]}),
        html.Div([
            f'{np.round(top_prob * 100, 1)}%',
        ], className='label', style={
            'font-size': '1.4em',
        }),
    ], 
    style={
        'display': 'block',
    }, 
    className='ui statistic')
    return prediction_output_div
