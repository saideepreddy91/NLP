
import os
from analysis.misc import renamed_load, read_sentiment, read_news_data, rgba

from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
nltk.download('punkt')

class UI_STYLES:
    POSITIVE_COLOR_CLASSNAME = 'blue'
    NEGATIVE_COLOR_CLASSNAME = 'red'
    POSITIVE_COLOR = (0, 116, 217)
    NEGATIVE_COLOR = (176, 48, 96)

    TEN_COLOR_PALETTE_FOR_GRAPH = [
        '#3D6496',
        '#E49344',
        '#D0605D',
        '#84B6B2',
        '#6A9F58',
        '#E7CB60',
        '#A87C9F',
        '#F2A2A9',
        '#977662',
        '#B8B0AC',
    ]

class UserReviewGlobalModel:
    fv = None
    clf = None

    sentiment = None
    trainX = None
    train_pred_probs = None

    fv_text_preprocessor = None
    fv_text_tokenize = None

    feature_names = None
    feature_names_set = None
    clf_coefficients = None
    clf_intercept = None


class NewsClassificationGlobalModel:
    fv = None
    clf = None

    feature_names_set = None

    news = None
    category_to_colors = None

user_review_model = UserReviewGlobalModel()
news_model = NewsClassificationGlobalModel()

def load_pickle(name):
    with open(name, 'rb') as fin:
        data = renamed_load(fin)
        print("Loaded: ", name)
        return data


def initialize_global_vars_for_user_review_section():
    root_dir = os.getcwd()

    fv = load_pickle(os.path.join(root_dir, 'assets/model_user_review/fv.pkl'))
    clf = load_pickle(os.path.join(root_dir, 'assets/model_user_review/clf.pkl'))

    fv_text_preprocessor = fv.build_preprocessor()
    fv_text_tokenize = fv.build_tokenizer()

    feature_names = fv.get_feature_names()
    feature_names_set = set(feature_names)
    clf_coefficients = clf.coef_[0] # 1d array
    clf_intercept = clf.intercept_[0] # scalar

    sentiment = read_sentiment(os.path.join(root_dir, 'assets/model_user_review/sentiment.tar.gz'))
    trainX = fv.transform(sentiment.train_data)
    train_pred_probs = clf.predict_proba(trainX)


    user_review_model.fv = fv
    user_review_model.clf = clf
    user_review_model.fv_text_preprocessor = fv_text_preprocessor
    user_review_model.fv_text_tokenize = fv_text_tokenize
    user_review_model.feature_names = feature_names
    user_review_model.feature_names_set = feature_names_set
    user_review_model.clf_coefficients = clf_coefficients
    user_review_model.clf_intercept = clf_intercept
    user_review_model.sentiment = sentiment
    user_review_model.trainX = trainX
    user_review_model.train_pred_probs = train_pred_probs

def initialize_global_vars_for_news_section():
    root_dir = os.getcwd()

    fv = load_pickle(os.path.join(root_dir, 'assets/model_news/fv_news.pkl'))
    clf = load_pickle(os.path.join(root_dir, 'assets/model_news/clf_news.pkl'))
    news_data = read_news_data(os.path.join(root_dir, 'assets/model_news/news_dataset.tar.gz'))
    fv_text_preprocessor = fv.build_preprocessor()
    fv_text_tokenize = fv.build_tokenizer()
    feature_names = fv.get_feature_names()
    feature_names_set = set(feature_names)

    category_to_colors = {c: color for c, color in zip(clf.classes_, UI_STYLES.TEN_COLOR_PALETTE_FOR_GRAPH)}
    
    news_model.fv = fv
    news_model.clf = clf
    news_model.feature_names = feature_names
    news_model.feature_names_set = feature_names_set
    news_model.news_data = news_data
    news_model.fv_text_preprocessor = fv_text_preprocessor
    news_model.fv_text_tokenize = fv_text_tokenize
    news_model.category_to_colors = category_to_colors
