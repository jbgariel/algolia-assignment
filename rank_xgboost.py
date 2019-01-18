# coding=utf-8
# This is an adaptation from Xgboost learning to rank demo
# https://github.com/dmlc/xgboost/tree/master/demo/rank
''' Learning to rank experiment with Xgboost '''

import xgboost as xgb

from sklearn.datasets import load_svmlight_file
from xgboost import DMatrix


# 🤷‍♂️ Not understand why XgBOOST needs that.. Tensorflow doesn't..
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

    # Train Xgboost with basic parameters
    params = {'objective': 'rank:pairwise', 'eta': 0.1, 'gamma': 1.0,
              'min_child_weight': 0.1, 'max_depth': 6}
    xgb_model = xgb.train(params, train_dmatrix, num_boost_round=4,
                          evals=[(valid_dmatrix, 'validation')])
    pred = xgb_model.predict(test_dmatrix)

    print(pred)
    print(y_test)


if __name__ == "__main__":
    main()
