''' Utils functions for algolia assignement'''

import glob
import json
import pandas as pd


def import_data(path_to_dir):
    ''' import and merge all json files in path_to_dir '''
    all_files = glob.glob(path_to_dir + "/*")
    list_ = []
    for file_ in all_files:
        with open(file_, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                try:
                    list_.append(json.loads(line))
                except json.decoder.JSONDecodeError:
                    # print('error with file {} at line {}'.format(file_, index))
                    pass  # skip errors
    df = pd.DataFrame(list_)
    print('Imported {} lines from {} files'.format(len(df.index), len(all_files)))
    return df
