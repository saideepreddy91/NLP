import plotly.graph_objs as go
import numpy as np

from analysis.misc import rgba
from analysis.global_vars import user_review_model
from analysis.global_vars import UI_STYLES


from analysis.misc import map_to_new_low_and_high, get_relative_strengths

from enum import Enum

class FeatureDisplayMode(Enum):
    prediction_contribution = 'prediction_contribution'
    feature_weight = 'feature_weight'
    raw_feature_tfidf = 'tfidf'

    @property
    def title(self):
        figure_title = None
        display_mode = self
        if display_mode == FeatureDisplayMode.prediction_contribution:
            figure_title = 'Prediction Contribution'
        elif display_mode == FeatureDisplayMode.feature_weight:
            figure_title = 'Feature Weight'
        elif display_mode == FeatureDisplayMode.raw_feature_tfidf:
            figure_title = 'TF-IDF'
        else:
            raise ValueError("Invalid `display_mode` type.")
        return figure_title

def preprocess(raw_input_text):
    text = user_review_model.fv_text_preprocessor(raw_input_text)
    text = ' '.join(user_review_model.fv_text_tokenize(text))
    return text

def sort_features_human_friendly_order(tokens, features):    
    """
    Sort ngram features in order of input tokens.
    """
    preferred_ordered_features = []

    # Short features last
    features = sorted(features, key=len, reverse=True)
    
    for token in tokens:
        # Iterate from last (shortest features first), and remove in-place*
        for feature in reversed(features):
            # Only add those that begins with current token
            if feature.startswith(token):
                preferred_ordered_features.append(feature)
                features.remove(feature)
    return preferred_ordered_features

def get_random_sample():
    n_dev_total = len(user_review_model.sentiment.dev_data)
    return user_review_model.sentiment.dev_data[np.random.randint(n_dev_total)]

def part1_analyze_coefficients(sentence, display_mode):
    """Analyze (already-preprocessed) review sentence"""

    assert isinstance(display_mode, FeatureDisplayMode), "`display_mode` must be `FeatureDisplayMode`."

    fv = user_review_model.fv
    clf = user_review_model.clf
    clf_coefficients = user_review_model.clf_coefficients
    feature_names = user_review_model.feature_names
    # feature_names_set = user_review_model.feature_names_set

    x = fv.transform([sentence]).toarray().flatten()

    prob_x = clf.predict_proba([x])[0]
    pred_x = int(prob_x[1] > 0.5)

    coef_feature_products = clf_coefficients * x

    nonzero_inds = x.nonzero()[0]

    if len(nonzero_inds) == 0:
        raise ValueError('No features detected.')

    figure_title = display_mode.title
    if display_mode == FeatureDisplayMode.prediction_contribution:
        nonzero_strength_values = coef_feature_products[nonzero_inds]
    elif display_mode == FeatureDisplayMode.feature_weight:
        nonzero_strength_values = clf_coefficients[nonzero_inds]
    elif display_mode == FeatureDisplayMode.raw_feature_tfidf:
        nonzero_strength_values = x[nonzero_inds]
    else:
        raise ValueError("Invalid `display_mode` type.")

    detected_features = [feature_names[ind] for ind in nonzero_inds]

    ##################################
    # Show in feature extraction list
    ##################################

    tokenize = fv.build_tokenizer()
    tokens = tokenize(sentence)
    human_sorted_features = sort_features_human_friendly_order(tokens, detected_features)

    feature_to_ind = fv.vocabulary_
    ind_to_feature_contribution = {ind: contrib for ind, contrib in zip(nonzero_inds, nonzero_strength_values)}
    human_sorted_values = [ind_to_feature_contribution[feature_to_ind[f]] for f in human_sorted_features]


    ########################################
    # Show in feature contribution bar graph
    ########################################

    sorted_feature_values = sorted(zip(detected_features, nonzero_strength_values), key=lambda tup: tup[1]) # sort by values

    negative_feature_list = []
    negative_feature_values = []
    positive_feature_list = []
    positive_feature_values = []


    # Separate negative and positive
    min_val = np.inf
    max_val = -np.inf
    for f, val in sorted_feature_values:
        if val < 0:
            negative_feature_list.append(f)
            negative_feature_values.append(val)
        else:
            positive_feature_list.append(f)
            positive_feature_values.append(val)

        # Also get max/min values for later use
        abs_val = abs(val)
        if abs_val < min_val:
            min_val = abs_val
        if abs_val > max_val:
            max_val = abs_val

    positive_bars = go.Bar(
        y = positive_feature_list,
        x = positive_feature_values,
        name = 'Positive',
        orientation = 'h',
        marker = {
            'color': rgba(*UI_STYLES.POSITIVE_COLOR, 0.7),
            'opacity': 0.7,
            'line': {
                'color': rgba(*UI_STYLES.POSITIVE_COLOR),
                'width': 2,
            }
        },
    )

    negative_bars = go.Bar(
        y = negative_feature_list,
        x = negative_feature_values,
        name = 'Negative',
        orientation = 'h',
        marker = {
            'color': rgba(*UI_STYLES.NEGATIVE_COLOR, 0.7),
            'line': {
                'color': rgba(*UI_STYLES.NEGATIVE_COLOR),
                'width': 2,
            }
        }
    )
        
    figure_feature_contribution = {
        'data': [
            negative_bars,
            positive_bars,
        ],
        'layout': go.Layout(
            title=figure_title,
            yaxis=dict(
                autorange="reversed", 
                automargin=True,
            ),
            xaxis=dict(
                automargin=True,
            ),
        ),
    }

    # Will used to later map in html UI e.g., opacity of elements based on strength
    relative_feature_strengths = get_relative_strengths(np.abs(human_sorted_values), 0.15, 1.0)
    data_for_sp = {
        'positive_features': list(zip(positive_feature_list, positive_feature_values)),
        'negative_features': list(zip(negative_feature_list, negative_feature_values)),
        'min_val': min_val,
        'max_val': max_val,
    }


    return {
        'figure_feature_contribution': figure_feature_contribution,
        'sp_data': data_for_sp,
        'human_sorted_features': human_sorted_features,
        'human_sorted_values': human_sorted_values,
        'relative_feature_strengths': relative_feature_strengths,
        'pred_x': pred_x,
        'prob_x': prob_x,
    }


def part1_create_sentiment_prediction_figure(sp_data, top_k=10):
    ########################################
    # Sentiment Prediction (sp_) Stacked Bar graph
    ########################################

    positive_features = sp_data['positive_features']
    negative_features = sp_data['negative_features']
    min_val = sp_data['min_val']
    max_val = sp_data['max_val']

    if len(positive_features) + len(negative_features) == 0:
        return {}

    clf_intercept = user_review_model.clf_intercept

    sp_figure_data = []

    base_strength = 0.3

    TOP_K_FEATURES = top_k

    top_k_positives = list(reversed(positive_features[-TOP_K_FEATURES:]))
    rest_positives = positive_features[:-TOP_K_FEATURES]
    total_rest_positive_value = sum([v for _, v in rest_positives])

    top_k_negatives = list(negative_features[:TOP_K_FEATURES])
    rest_negatives = negative_features[TOP_K_FEATURES:]
    total_rest_negative_value = abs(sum([v for _, v in rest_negatives]))

    def __create_bar(name, value, show_text, x, marker_color, line_color): 
        return go.Bar(
            x = [x],
            y = [value],
            text = name if show_text else None,
            name = name,
            textposition='auto',
            marker= {
                'color': marker_color,
                'line': {
                    'color': line_color,
                    'width': 1,
                },
            }
        )

    def create_positive_bar(name, value, opacity, show_text):
        return __create_bar(name, value, show_text, 'POSITIVE', rgba(*UI_STYLES.POSITIVE_COLOR, opacity), rgba(*UI_STYLES.POSITIVE_COLOR))
        
    def create_negative_bar(name, value, opacity, show_text):
        return __create_bar(name, value, show_text, 'NEGATIVE', rgba(*UI_STYLES.NEGATIVE_COLOR, opacity), rgba(*UI_STYLES.NEGATIVE_COLOR))

    ##################
    # POSITIVE STACKS
    ##################
    for i, (f,v) in enumerate(top_k_positives):
        relative_strength = np.round(map_to_new_low_and_high(v, min_val, max_val, base_strength, 1), 1)
        sp_figure_data.append(create_positive_bar(f, v, relative_strength, show_text=(i < 3)))

    if len(rest_positives) > 0:
        sp_figure_data.append(create_positive_bar(f'{len(rest_positives)} others', total_rest_positive_value, 0.1, show_text=True))
    
    ##################
    # NEGATIVE STACKS
    ##################
    for i, (f,v) in enumerate(top_k_negatives):
        v = abs(v)
        relative_strength = np.round(map_to_new_low_and_high(v, min_val, max_val, base_strength, 1), 1)
        sp_figure_data.append(create_negative_bar(f, v, relative_strength, show_text=(i < 3)))

    if len(rest_negatives) > 0:
        sp_figure_data.append(create_negative_bar(f'{len(rest_negatives)} others', total_rest_negative_value, 0.1, show_text=True))


    ##################
    # INTERCEPT
    ##################
    sp_intercept_bar = None
    if clf_intercept > 0:
        opacity = np.round(map_to_new_low_and_high(clf_intercept, min_val, max_val, base_strength, 1), 1)
        sp_intercept_bar = create_positive_bar('INTERCEPT', clf_intercept, opacity, True)
    else:
        opacity = np.round(map_to_new_low_and_high(abs(clf_intercept), min_val, max_val, base_strength, 1), 1)
        sp_intercept_bar = create_negative_bar('INTERCEPT', abs(clf_intercept), opacity, True)

    sp_figure_data.append(sp_intercept_bar)

    sp_stacked_bars_layout = go.Layout(
        title='Positiveness vs Negativeness',
        barmode='stack'
    )

    figure_sp_stacked_bars = go.Figure(data=sp_figure_data, layout=sp_stacked_bars_layout)
    return figure_sp_stacked_bars

def part1_create_feature_in_context(feature, show_k_samples):
    fv = user_review_model.fv
    sentiment = user_review_model.sentiment
    trainX = user_review_model.trainX
    pred_probs = user_review_model.train_pred_probs

    feature_ind = fv.vocabulary_[feature]
    found_in_training_inds = trainX[:, feature_ind].nonzero()[0]

    found_preds = sentiment.trainy[found_in_training_inds]
    positive_inds = found_in_training_inds[np.where(found_preds == 1)][:show_k_samples]
    negative_inds = found_in_training_inds[np.where(found_preds == 0)][:show_k_samples]

    num_training_samples = trainX.shape[0]
    num_appears_in_train_set = len(found_in_training_inds)
    num_not_appear_in_train_set = num_training_samples - num_appears_in_train_set

    num_positives = sum(sentiment.trainy[found_in_training_inds])
    num_negatives = num_appears_in_train_set - num_positives


    pie_trace = go.Pie(
        labels=['Positive context', 'Negative context', 'Not appear'], 
        values=[num_positives, num_negatives, num_not_appear_in_train_set],
        hoverinfo='label+percent', 
        textinfo='value',
        textfont=dict(size=16),
        marker=dict(
            colors=[rgba(*UI_STYLES.POSITIVE_COLOR), rgba(*UI_STYLES.NEGATIVE_COLOR), rgba(217, 217, 217, 0.5)], 
        ),
    )

    pie_layout = go.Layout(
        title=f"'{feature}' in training data",
    )
    
    pie_figure = go.Figure(data=[pie_trace], layout=pie_layout)

    appearance_percent_value = np.round(100*num_appears_in_train_set/num_training_samples, 2)
    appearance_percent_text = f' ({appearance_percent_value}%)' if appearance_percent_value != 0 else ''

    positive_num_text = f'**{num_positives}**' if num_positives > num_negatives else num_positives
    negative_num_text = f'**{num_negatives}**' if num_negatives > num_positives else num_negatives

    positive_negative_comparison_text = None
    if num_positives == 0:
        positive_negative_comparison_text = f"'{feature}' appears **only in negative context!**"
    elif num_negatives == 0:
        positive_negative_comparison_text = f"'{feature}' appears **only in positive context!**"
    else:
        pos_neg_ratio = num_positives / num_negatives
        neg_pos_ratio = 1. / pos_neg_ratio
        if pos_neg_ratio  >= 2:
            positive_negative_comparison_text = f"'{feature}' appears **{int(pos_neg_ratio)} times in positive context** compared to negative context!"
        elif neg_pos_ratio >= 2:
            positive_negative_comparison_text = f"'{feature}' appears **{int(neg_pos_ratio)} times in negative context** compared to positive context!"
        if positive_negative_comparison_text is not None:
            positive_negative_comparison_text = f"""
>
> {positive_negative_comparison_text}
>
"""     
        else:
            positive_negative_comparison_text = ''

    md_explaination = f"""
'{feature}' appears in {num_appears_in_train_set} training samples, from a total of {num_training_samples} samples{appearance_percent_text}.

* {positive_num_text} of them are positive ({np.round(100*num_positives/num_appears_in_train_set, 2)}% of total appearances).
* {negative_num_text} of them are negative ({np.round(100*num_negatives/num_appears_in_train_set, 2)}% of total appearances).

{positive_negative_comparison_text}
    """
    return pie_figure, dict(
        md_explaination=md_explaination,
        positive_samples=[sentiment.train_data[ind] for ind in positive_inds],
        negative_samples=[sentiment.train_data[ind] for ind in negative_inds],
        positive_samples_pred_probs = pred_probs[positive_inds],
        negative_samples_pred_probs = pred_probs[negative_inds],
    )
    

def get_information_values_for_top_positive_and_negative_features():

    top_negatives = [
        ('the worst', 0.555712),
        ('horrible', 0.195638),
        ('worst', 0.184547),
        ('not', 0.183855),
        ('terrible', 0.100446),
        ('rude', 0.091172),
        ('bad', 0.054171),
        ('asked', 0.034427),
        ('disappointed', 0.030780),
        ('slow', 0.030436),
    ]

    top_positives = [
        ('great', 0.217689),
        ('amazing', 0.148440),
        ('delicious', 0.123200),
        ('awesome', 0.097469),
        ('love', 0.094778),
        ('excellent', 0.092318),
        ('love this', 0.063266),
        ('favorite', 0.056707),
        ('perfect', 0.041055),
        ('fantastic', 0.036014),
    ]

    top_positive_iv_bars = go.Bar(
        y = [iv for _, iv in top_positives],
        x = [feature for feature, _ in top_positives],
        name = 'Most informative features for positive',
        marker = {
            'color': rgba(*UI_STYLES.POSITIVE_COLOR, 0.7),
            'opacity': 0.7,
            'line': {
                'color': rgba(*UI_STYLES.POSITIVE_COLOR),
                'width': 2,
            }
        },
    )
    top_negative_iv_bars = go.Bar(
        y = [iv for _, iv in top_negatives],
        x = [feature for feature, _ in top_negatives],
        name = 'Most informative features for negative labeling',
        marker = {
            'color': rgba(*UI_STYLES.NEGATIVE_COLOR, 0.7),
            'opacity': 0.7,
            'line': {
                'color': rgba(*UI_STYLES.NEGATIVE_COLOR),
                'width': 2,
            }
        },
    )
    top_positive_layout = go.Layout(
        title="Most informative features (IV) for positive labeling",
        yaxis=dict(
            title='IV',
            automargin=True,
            fixedrange=True,
        ),
        xaxis=dict(
            automargin=True,
            fixedrange=True,
        )
    )
    top_negative_layout = go.Layout(
        title="Most informative features (IV) for negative labeling",
        yaxis=dict(
            title='IV',
            automargin=True,
            fixedrange=True,
        ),
        xaxis=dict(
            automargin=True,
            fixedrange=True,
        )
    )
    return go.Figure([top_positive_iv_bars], top_positive_layout), go.Figure([top_negative_iv_bars], top_negative_layout)



