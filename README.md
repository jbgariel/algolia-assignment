# Algolia Assignement

> This repository provide an end-to-end workflow and steps to analyze, train and serve a machine learning model. 

## Features
* 🔍 **Data Analysis and validation** of HN dataset, running on a local jupiter notebook
* 💫 **Algolia Search performance metrics** on local jupiter notebook
* 🏋️‍♀️ **Dataset preparation and features engineering**
* 🚀 **ML Experiments** for Learning to rank algorithms
* 🐳 **Out of the box**, with dockerization

## The dataset
The dataset was provided by algolia. The data comes from [Hacker News Search](https://hn.algolia.com/).

## Dependencies
Docker:
* Engine version ≥ 17.05
* Compose version ≥ 1.13

## Setup the container

Build and run the container:
```
docker-compose build
docker-compose up -d
```

## 1- 🔍 Analyzing and validating the data

> Jupyter notebook should be activated, go to [hnsearch_analysis](http://localhost:9999/notebooks/hnsearch_analysis.ipynb#)

### Key findings
* There are some missing lines at the end of each file, I did not take them into account
* Looks like there are clicking bots (863 clicks in a few seconds for one query)
* Less than 10% of queries don't return any hit (63043/881052, 7%)

## 2- 💫 Performance metrics of Algolia Search

> Jupyter notebook should be activated, go to [algoliasearch_metrics](http://localhost:9999/notebooks/algoliasearch_metrics.ipynb#)

This notebook propose a serie of performance metrics to evaluate Algolia Search based on the given dataset.

### Key findings
* Mean Reciprocal Rank is **0.53**
* Mean Average Precision is **0.52**
* Average first click position is **4.55** (from previous notebook)

## 3- 👩‍🔬 Learning to Rank experiment

Algolia search engine returns a list of ranked document, 

Schema

**Could we improve Algolia ranking with machine learning technics?**

> ⚠️ Due to time constrains, some shortcuts have been made, like not taking into account filters etc..

### 1- Feature Engineering

Run the following commands (remove `--light` flag to compute the whole dataset):
```
docker exec -it algolia-assignement python build_dataset.py --light
```

The data is filtered on query with clicks and at least 2 hits (there is no ranking if only one hit is returned 😉). Some necessary HN data (post title, author, score etc) was queried with a simple script (`docker exec -it algolia-assignement python extract_hn_data.py`), this script took a few hours to run so the data is directly joined to the git repo.

The proposed features are:
* Keyword match in query
* Keyword match in query ponderated with TF-IDF
* Levenshtein distance
* HN score (with log1p transformation)

Some other features could be used (not implemented):
* Jaccard distance and other distance metrics
* Cosine Similarity Between aggregated Word2Vec of query and title
* Post age (query_timestamp - post_timestamp)

For Learning-to-rank algorithms, the output data must follow [the LIBSVM format](https://sourceforge.net/p/lemur/wiki/RankLib%20File%20Format).

### 2- Learning to Rank Algorithms

Two algorithms have beed tested:
* **XgBoost**
* **Tensorflow Ranking** *(freshly release in december 2018 👌)*

For XgBoost, run the command:
```
docker exec -it algolia-assignement python rank_xgboost.py
```

For Tensorflow Ranking, run the command:
```
docker exec -it algolia-assignement python rank_tensorflow.py
```

Tensorboard logs are saved, to see them:
```
docker exec -it algolia-assignement tensorboard --logdir=<ranker.model_dir output>
```
...and visit [localhost:6006](http://localhost:6006)

### 3- Evaluation metrics and results

* Due to time constrain (and lack of implemented Learning to rank metrics), Xgboost metrics were not computed (there is no standard way of computing metrics)
* The most complicated part is actually building the training data frame in the LibSVM format (I did not know this format before)
* ML algorithm are performing well, the training is super fast even without GPUs
* Tensorflow Ranking increased significally the performance of Algolia Search (see graph below)
* I also found that some Algolia competitors (elasticsearch) are building [ML plugin](https://elasticsearch-learning-to-rank.readthedocs.io/en/latest/index.html) for Learning-to-rank
* Algolia could also leverage other data to improve click rate, like users, ip address, history etc..

![alt text](metrics.png)

## Takeaway
* Algolia search is performing super well 👏 both in time performance and relevance
* Learning-to-rank is not well documented, this is a secret domain (google is not releasing a lot of stuff on that 😉)
* Machine Learning could improve Algolia search relevance and click rate




