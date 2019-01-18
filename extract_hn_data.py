# coding=utf-8
''' Extract HN data to enrich dataframe '''

import pandas as pd
import requests

DATA_DIR = 'data'
HN_API_URL = 'https://hacker-news.firebaseio.com/v0/item/{}.json'


def enrich_data(hit):
    r = requests.get(HN_API_URL.format(hit))
    value = r.json()
    return {
        'hit': hit,
        'title': value.get('title', value.get('text', None)),
        'score': value.get('score', None),
        'descendants': value.get('descendants', None),
        'type': value.get('type', None),
        'by': value.get('by', None)
        }


def main():
    # Import data
    data = pd.read_csv('tmp_data.csv', sep='\t')
    print('Imported {} lines'.format(len(data.index)))
    hits = data['hit'].drop_duplicates().tolist()
    print('Deduplicated to: {}'.format(len(hits)))

    i = 0
    list_ = []
    for hit in hits:
        print('  > {}/{}'.format(i, len(hits)), end="\r", flush=True)
        try:
            dict_ = enrich_data(hit)
            list_.append(dict_)
        except Exception as error:  # pylint: disable=broad-except
            print('Error with line: {}'.format(hit))
        i = i + 1

        if i % 1000 == 0:
            pd.DataFrame(list_).to_csv('hn_data.tsv', index=False,
                                       sep='\t', encoding='utf-8')
    pd.DataFrame(list_).to_csv('hn_data.tsv', index=False,
                               sep='\t', encoding='utf-8')


if __name__ == "__main__":
    main()
