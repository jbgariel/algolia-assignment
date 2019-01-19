# coding=utf-8
# This is an adaptation from Tensorflow Ranking learning-to-rank demo
# https://github.com/tensorflow/ranking/blob/master/tensorflow_ranking/examples/tf_ranking_libsvm.ipynb
''' Learning to rank experiment with Tensorflow Ranking '''

import tensorflow as tf
import tensorflow_ranking as tfr

tf.enable_eager_execution()
tf.executing_eagerly()

_TRAIN_DATA_PATH = "hn.train"
_TEST_DATA_PATH = "hn.test"
_LOSS = "pairwise_logistic_loss"
_LIST_SIZE = 10  # We only have the first 10 hits
_NUM_FEATURES = 4
_BATCH_SIZE = 32
_HIDDEN_LAYER_DIMS = ["20", "10"]


def input_fn(path):
    train_dataset = tf.data.Dataset.from_generator(
        tfr.data.libsvm_generator(path, _NUM_FEATURES, _LIST_SIZE),
        output_types=(
            {str(k): tf.float32 for k in range(1, _NUM_FEATURES+1)},
            tf.float32
        ),
        output_shapes=(
            {str(k): tf.TensorShape([_LIST_SIZE, 1])
             for k in range(1, _NUM_FEATURES+1)},
            tf.TensorShape([_LIST_SIZE])
        )
    )

    train_dataset = train_dataset.shuffle(1000).repeat().batch(_BATCH_SIZE)
    return train_dataset.make_one_shot_iterator().get_next()


def example_feature_columns():
    """Returns the example feature columns."""
    feature_names = [
        "%d" % (i + 1) for i in range(0, _NUM_FEATURES)
    ]
    return {
        name: tf.feature_column.numeric_column(
            name, shape=(1,), default_value=0.0) for name in feature_names
    }


def make_score_fn():
    """Returns a scoring function to build `EstimatorSpec`."""

    def _score_fn(context_features, group_features, mode, params, config):
        """Defines the network to score a documents."""
        del params
        del config
        # Define input layer.
        example_input = [
            tf.layers.flatten(group_features[name])
            for name in sorted(example_feature_columns())
        ]
        input_layer = tf.concat(example_input, 1)

        cur_layer = input_layer
        for i, layer_width in enumerate(int(d) for d in _HIDDEN_LAYER_DIMS):
            cur_layer = tf.layers.dense(
              cur_layer,
              units=layer_width,
              activation="tanh")

        logits = tf.layers.dense(cur_layer, units=1)
        return logits

    return _score_fn


def eval_metric_fns():
    """Returns a dict from name to metric functions.

    This can be customized as follows. Care must be taken when handling padded
    lists.

    def _auc(labels, predictions, features):
    is_label_valid = tf_reshape(tf.greater_equal(labels, 0.), [-1, 1])
    clean_labels = tf.boolean_mask(tf.reshape(labels, [-1, 1], is_label_valid)
    clean_pred = tf.boolean_maks(tf.reshape(predictions, [-1, 1], is_label_valid)
    return tf.metrics.auc(clean_labels, tf.sigmoid(clean_pred), ...)
    metric_fns["auc"] = _auc

    Returns:
    A dict mapping from metric name to a metric function with above signature.
    """
    metric_fns = {}
    metric_fns.update({
        "metric/ndcg@%d" % topn: tfr.metrics.make_ranking_metric_fn(
            tfr.metrics.RankingMetricKey.NDCG, topn=topn)
        for topn in [1, 3, 5, 10]
    })

    print(metric_fns)
    metric_fns['metric/mean_reciprocal_rank'] = tfr.metrics.make_ranking_metric_fn(
        tfr.metrics.RankingMetricKey.MRR)
    metric_fns['metric/average_relevance_position'] = tfr.metrics.make_ranking_metric_fn(
        tfr.metrics.RankingMetricKey.ARP)

    return metric_fns


def get_estimator(hparams):
    """Create a ranking estimator.

    Args:
    hparams: (tf.contrib.training.HParams) a hyperparameters object.

    Returns:
    tf.learn `Estimator`.
    """
    def _train_op_fn(loss):
        """Defines train op used in ranking head."""
        return tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=hparams.learning_rate,
            optimizer="Adagrad")

    ranking_head = tfr.head.create_ranking_head(
        loss_fn=tfr.losses.make_loss_fn(_LOSS),
        eval_metric_fns=eval_metric_fns(),
        train_op_fn=_train_op_fn)

    return tf.estimator.Estimator(
        model_fn=tfr.model.make_groupwise_ranking_fn(
            group_score_fn=make_score_fn(),
            group_size=1,
            transform_fn=None,
            ranking_head=ranking_head),
        params=hparams)


hparams = tf.contrib.training.HParams(learning_rate=0.05)
ranker = get_estimator(hparams)

print(ranker.train(input_fn=lambda: input_fn(_TRAIN_DATA_PATH), steps=100))
print(ranker.evaluate(input_fn=lambda: input_fn(_TEST_DATA_PATH), steps=100))
print('Tensorboard logs available at: {}'.format(ranker.model_dir))
