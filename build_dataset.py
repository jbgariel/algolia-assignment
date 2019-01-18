# coding=utf-8
# Some function are based on Anokas's kaggle kernel:
# https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-lb
''' Build data set and features for learning to rank experiments '''

import argparse
import nltk
import numpy as np
import pandas as pd
import requests

from collections import Counter
from Levenshtein import distance
from nltk.corpus import stopwords
from sklearn.datasets import dump_svmlight_file
from sklearn.preprocessing import LabelEncoder

from notebooks.utils import import_data

nltk.download('stopwords')

DATA_DIR = 'data'
HN_API_URL = 'https://hacker-news.firebaseio.com/v0/item/{}.json'
STOPWORDS = set(stopwords.words("english"))


# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely
# rare words smaller
def get_weight(count, eps=5000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)


def enrich_data(id):
    r = requests.get(HN_API_URL.format(id))
    if r.status_code == 200:
        value = r.json()
        print(str(value).encode('utf-8'))
    else:
        print('error with {}'.format(id))
    return [value.get('title', value.get('text', None)),
            value.get('score', None), value.get('descendants', None),
            value.get('type', None), value.get('by', None)]


def filter_data(data):
    data['nb_clicks'] = data['clicks'].apply(
        lambda x: len(x) if isinstance(x, list) else 0)
    data = data[data['nb_clicks'] > 0]

    data['nb_hits_displayed'] = data['hits'].apply(
        lambda x: len(x) if isinstance(x, list) else 0)
    data = data[data['nb_hits_displayed'] > 1]
    return data


def tfidf_word_match_share(string_1, string_2, weights):
    q1words = {}
    q2words = {}
    for word in str(string_1).lower().split():
        if word not in STOPWORDS:
            q1words[word] = 1
    for word in str(string_2).lower().split():
        if word not in STOPWORDS:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R


def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['query']).lower().split():
        if word not in STOPWORDS:
            q1words[word] = 1
    for word in str(row['title']).lower().split():
        if word not in STOPWORDS:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--call_hn',
        help=('Call HN API to get data.'),
        default=False,
        type=bool)
    known_args, _ = parser.parse_known_args()

    # Import data
    data = import_data(DATA_DIR).head(10000)
    print('Imported {} lines'.format(len(data.index)))

    # Filter data
    data = filter_data(data)
    print('Filter data: {} lines'.format(len(data.index)))

    # Expand dataframe, one line = one hit
    lens = [len(item) for item in data['hits']]
    df_expand = pd.DataFrame({'query': np.repeat(data['query'].values, lens),
                              'query_id': np.repeat(data['query_id'].values, lens),
                              'clicks': np.repeat(data['clicks'].values, lens),
                              'timestamp': np.repeat(data['timestamp'].values, lens),
                              'hit': np.hstack(data['hits'])
                              })

    # Determine where are the clicks
    df_expand['target'] = 0
    for idx, row in df_expand.iterrows():
        if any(click['object_id'] == row['hit'] for click in row['clicks']):
            df_expand.loc[idx, 'target'] = 1
    print('{}/{} clicks in dataframe'.format(sum(df_expand.target), len(df_expand.index)))

    # Enrich with HN API
    if known_args.call_hn:
        df_expand[['title', 'score', 'descendants', 'type', 'by']] = df_expand.apply(
            lambda row: pd.Series(enrich_data(row['hit'])), axis=1)
    else:
        hn_data = pd.read_csv('hn_data_1801.tsv', sep='\t')
        hn_data = hn_data[np.isfinite(hn_data['hit'])]
        hn_data['hit'] = hn_data['hit'].astype(int)
        hn_data['title'] = hn_data['title'].astype(str)
        df_expand['hit'] = df_expand['hit'].astype(int)
        df_expand = df_expand.merge(hn_data, on='hit')
        df_expand[['title', 'type', 'by']] = df_expand[['title', 'type', 'by']].fillna('')
        df_expand['score'] = df_expand['score'].fillna(0)

    '''
    FEATURES BUILDING
    '''

    # Computing TF-IDF weights
    word_list = pd.Series(df_expand['title'].tolist() + df_expand['query'].tolist()).astype(str)
    words = (" ".join(word_list)).lower().split()
    counts = Counter(words)
    weights = {word: get_weight(count) for word, count in counts.items()}

    # Applying features functions on dataframe
    df_expand['f1_levenshtein_distance'] = df_expand[['title', 'query']].apply(
        lambda x: distance(*x)/100, axis=1)
    df_expand['f2_log1_score'] = np.log1p(df_expand['score'])
    df_expand['f3_word_match'] = df_expand.apply(word_match_share,
                                                 axis=1, raw=True)
    df_expand['f4_tfidf_word_match'] = df_expand[['title', 'query']].apply(
        lambda x: tfidf_word_match_share(*x, weights=weights), axis=1)

    # Transform query_id to incremental id (needed for LibSVM format)
    df_expand['qid'] = LabelEncoder().fit_transform(df_expand['query_id'])

    # Order dataframe by qid
    df_expand.sort_values(by=['qid'], inplace=True, ascending=False)

    # Split in train, test, valid, not using train_test_split for avoiding shuffling
    first_cut = int(len(df_expand)*0.80)
    second_cut = first_cut + int(len(df_expand)*0.10)
    train = df_expand.head(first_cut)
    test = df_expand.iloc[first_cut:second_cut, :]
    valid = df_expand.tail(second_cut)

    # Save to LibSVM format
    features = ['f1_levenshtein_distance', 'f2_log1_score',
                'f3_word_match', 'f4_tfidf_word_match']

    X_train = train[features]
    y_train = train.target
    query_train = train.qid
    dump_svmlight_file(X_train, y_train, 'hn.train', zero_based=False,
                       query_id=query_train, multilabel=False)

    X_test = test[features]
    y_test = test.target
    query_test = test.qid
    dump_svmlight_file(X_test, y_test, 'hn.test', zero_based=False,
                       query_id=query_test, multilabel=False)

    X_valid = valid[features]
    y_valid = valid.target
    query_valid = valid.qid
    dump_svmlight_file(X_valid, y_valid, 'hn.vali', zero_based=False,
                       query_id=query_valid, multilabel=False)


if __name__ == "__main__":
    main()
