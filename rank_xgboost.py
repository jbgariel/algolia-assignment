# coding=utf-8
# This is an adaptation from Xgboost learning to rank demo
# https://github.com/dmlc/xgboost/tree/master/demo/rank
''' Learning to rank experiment with Xgboost '''

import pandas as pd
import xgboost as xgb

from sklearn.datasets import load_svmlight_file
from xgboost import DMatrix

from notebooks.rank_metrics import mean_reciprocal_rank, \
    mean_average_precision, ndcg_at_k


# 🤷‍♂️ Dont understand why XgBOOST needs that.. Tensorflow doesn't..
def group_qid(list_):
    output = []
    for idx, val in enumerate(list_):
        if idx == 0:
            output = [1]
            cursor = 0
        else:
            if val == list_[idx - 1]:
                output[cursor] = output[cursor] + 1
            else:
                cursor = cursor + 1
                output.append(1)
    return output


# Assert function is working
assert group_qid([1]) == [1]
assert group_qid([1, 1, 1, 2, 2]) == [3, 2]


# Regroup prediction in a format used by metrics
def regroup_results(group_test, pred, y_test):
    clicks_matrix = []
    for mat_len in group_test:
        pred_ = pred[:mat_len]
        y_test_ = y_test[:mat_len]

        _array = [x for _, x in sorted(zip(pred_, y_test_), reverse=True)]
        clicks_matrix.append(_array)

        pred = pred[mat_len:]
        y_test = y_test[mat_len:]

    return clicks_matrix


# Assert function is working
assert_group_test = [3, 2]
assert_pred = [.1, .3, .2, .3, .2]
assert_y_test = [0, 0, 1, 1, 0]
assert regroup_results(assert_group_test,
                       assert_pred, assert_y_test) == [[0, 1, 0], [1, 0]]


def main():
    #  Import training data
    x_train, y_train, qid_train = load_svmlight_file("hn.train", query_id=True)
    x_valid, y_valid, qid_valid = load_svmlight_file("hn.vali", query_id=True)
    x_test, y_test, qid_test = load_svmlight_file("hn.test", query_id=True)

    group_train = group_qid(qid_train)
    group_valid = group_qid(qid_valid)
    group_test = group_qid(qid_test)

    train_dmatrix = DMatrix(x_train, y_train)
    valid_dmatrix = DMatrix(x_valid, y_valid)
    test_dmatrix = DMatrix(x_test)

    train_dmatrix.set_group(group_train)
    valid_dmatrix.set_group(group_valid)
    test_dmatrix.set_group(group_test)

    # Train Xgboost with basic parameters
    params = {'objective': 'rank:pairwise', 'eta': 0.1,
              # 'gamma': 1.0,
              # 'min_child_weight': 0.1,
              'max_depth': 3}
    params['eval_metric'] = ['ndcg@1', 'ndcg@3', 'ndcg@5', 'ndcg@10']
    xgb_model = xgb.train(params, train_dmatrix, num_boost_round=4,
                          evals=[(valid_dmatrix, 'validation')])
    pred = xgb_model.predict(test_dmatrix)

    data_predict = regroup_results(group_test, pred, y_test)

    # Testing random sample
    # Simple debug function that print algolia results and predictions
    def print_random_sample(line):
        prevsum = sum(group_test[:line])
        print('Algolia clicks are: {}'.format(y_test[prevsum:prevsum + group_test[line]]))
        print('Predictions are: {}'.format(pred[prevsum:prevsum + group_test[line]]))
        print('Xgboost clicks are: {}'.format(data_predict[line]))
    print_random_sample(89)

    print('> Mean reciprocal rank is : {}'.format(
        mean_reciprocal_rank(data_predict)))
    print('> Mean average position is : {}'.format(
        mean_average_precision(data_predict)))

    # nDCG
    for i in [1, 3, 5, 10]:
        ndcg_ = []
        for query in data_predict:
            ndcg_.append(ndcg_at_k(query, i))
        print('> nDCG@{} is : {}'.format(i, pd.Series(ndcg_).mean()))


if __name__ == "__main__":
    main()
