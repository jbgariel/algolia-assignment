# algolia-assignement

> This repository provide an end-to-end workflow and steps to analyze, train and serve a machine learning model. 

## Features
* 🔍 **Data Analysis and validation** of HN dataset, running on a local jupiter notebook
* 💫 **Algolia Search performance metrics**
* 🏋️‍♀️ **Model Training** in local
* 🚀 **Serving**, on simple local flask API
* 🐳 **Out of the box**, with dockerization

## The dataset
The dataset was provided by algolia. The data comes from [HN Search](https://hn.algolia.com/)

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

docker build -f build/Dockerfile -t algo_assign .
docker run -it -p 5001:5000 -p 8889:8889 -p 6669:6006 -d -v $(pwd)/notebooks:/usr/src/app/notebooks -v $(pwd)/data:/usr/src/app/data --name algo_assign algo_assign


## 🔍 Analyzing and validating the data

> Jupyter notebook should be activated, go to [hnsearch_analysis](http://localhost:8889/notebooks/hnsearch_analysis.ipynb#)

### Key findings
* There are some missing lines at the end of each file, I did not took them into account
* Looks like there are clicking bots (863 clicks in a few seconds for one query)
* Less than 10% of queries don't return any hit (63043/881052, 7%)

## 💫 Performance metrics of Algolia Search

> Jupyter notebook should be activated, go to [algoliasearch_metrics](http://localhost:8889/notebooks/algoliasearch_metrics.ipynb#)

This notebook propose a serie of performance metrics to evaluate Algolia Search based on the given dataset.

### Key findings
* Mean Reciprocal Rank is **0.53**
* Mean Average Precision is **0.52**









