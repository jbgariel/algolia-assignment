# coding=utf-8
# Some function are based on Anokas's kaggle kernel:
# https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-lb
''' Build data set and features for learning to rank experiments '''

import argparse
import nltk
import numpy as np
import os
import pandas as pd

from collections import Counter
from Levenshtein import distance
from nltk.corpus import stopwords
from sklearn.datasets import dump_svmlight_file
from sklearn.preprocessing import LabelEncoder

from notebooks.utils import import_data

nltk.download('stopwords', quiet=True)

DATA_DIR = 'data'
HN_DATA = 'hn_data.tsv'
STOPWORDS = set(stopwords.words("english"))


# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely
# rare words smaller
def get_weight(count, eps=1000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)


# Get additional data from HN
def enrich_data(df_expand):
    hn_data = pd.read_csv(HN_DATA, sep='\t', low_memory=False)
    hn_data = hn_data[np.isfinite(hn_data['hit'])]
    hn_data['hit'] = hn_data['hit'].astype(int)
    hn_data['title'] = hn_data['title'].astype(str)
    df_expand['hit'] = df_expand['hit'].astype(int)
    df_expand = df_expand.merge(hn_data, on='hit')
    df_expand[['title', 'type', 'by']] = df_expand[['title', 'type', 'by']].fillna('')
    df_expand['score'] = df_expand['score'].fillna(0)
    return df_expand


# Remove queries without click and only one hit
def filter_data(data):
    data['nb_clicks'] = data['clicks'].apply(
        lambda x: len(x) if isinstance(x, list) else 0)
    data = data[data['nb_clicks'] > 0]

    data['nb_hits_displayed'] = data['hits'].apply(
        lambda x: len(x) if isinstance(x, list) else 0)
    data = data[data['nb_hits_displayed'] > 1]
    return data


# Custom word match with TfIDF from
# https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-lb
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


# Custom word match from
# https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-lb
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
    # Get flags
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--light',
        help=('Compute less data for quicker iterations.'),
        action='store_true')
    known_args, _ = parser.parse_known_args()

    # Import data
    print('> Loading data...')
    data = import_data(DATA_DIR)
    if known_args.light:
        data = data.head(1000)
    assert len(data.index) > 0, 'Please add unzip data in the /data directory'
    print('> Imported {} lines'.format(len(data.index)))

    # Filter data
    data = filter_data(data)
    print('> Filtered data: {} lines'.format(len(data.index)))

    # Expand dataframe, one line = one hit
    lens = [len(item) for item in data['hits']]
    df_expand = pd.DataFrame({'query': np.repeat(data['query'].values, lens),
                              'query_id': np.repeat(data['query_id'].values, lens),
                              'clicks': np.repeat(data['clicks'].values, lens),
                              'timestamp': np.repeat(data['timestamp'].values, lens),
                              'hit': np.hstack(data['hits'])
                              })
    print('> Expanded to: {} lines'.format(len(df_expand.index)))

    # Enrich with HN API
    df_expand = enrich_data(df_expand)

    # Detect where are the clicks
    df_expand['target'] = 0
    for idx, row in df_expand.iterrows():
        if idx % 1000 == 0:
            print('  > {}/{}'.format(idx, len(df_expand.index)),
                  end="\r", flush=True)
        if any(click['object_id'] == row['hit'] for click in row['clicks']):
            df_expand.loc[idx, 'target'] = 1
    print('> {}/{} rows with clicks in dataframe'.format(sum(df_expand.target), len(df_expand.index)))

    '''
    FEATURES ENGINEERING
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

    # Order dataframe by qid (needed for LibSVM format)
    df_expand.sort_values(by=['qid'], inplace=True, ascending=False)

    # Split in train, test, valid, not using train_test_split for avoiding shuffling
    first_cut = int(len(df_expand)*0.80)
    second_cut = first_cut + int(len(df_expand)*0.10)
    train = df_expand.iloc[:first_cut, :]
    test = df_expand.iloc[first_cut:second_cut, :]
    valid = df_expand.iloc[second_cut:, :]
    print('> train: {} rows, test: {} rows, valid: {} rows'.format(len(train), len(test), len(valid)))

    '''
    SAVE TO LIBSVM FORMAT
    '''

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
