#!/bin/bash

set -u

echo installing tensorflow ranking...

apt-get update && apt-get install bazel
git clone https://github.com/tensorflow/ranking.git
cd ranking

bazel build //tensorflow_ranking/tools/pip_package:build_pip_package
bazel-bin/tensorflow_ranking/tools/pip_package/build_pip_package /tmp/ranking_pip

pip install /tmp/ranking_pip/tensorflow_ranking*.whl

bazel test //tensorflow_ranking/...

python -c "import tensorflow_ranking"
