# SciKit Learn Lab 2

Simple implementation of 11 different Machine Learning algorithms on the wine dataset from https://archive.ics.uci.edu/ml/datasets/Wine

## Running the Code

### Install Packages
Install the following packages to your environment:
- sklearn
- sklearn.model_selection
- sklearn.metrics
- xgboost
- pandas
- argparse

### Run Command
Run the file corresponding to the algorithm of choice:
```
python3 [file_name] [data_path]

#example
python3 models/decision_tree_wine.py ./data/wine/wine.csv
```

### Algorithm Choices and File Names
- Decision Tree - decision_tree_wine.py
- Neural Net - neural_net_wine.py
- Support Vector Machine - svc_wine.py
- Gaussian Naive Bayes - gnb_wine.py
- Logistic Regression - logistic_regression_wine.py
- K-Nearest Neighbors - knn_wine.py
- Bagging - bagging_wine.py
- Random Forest - random_forest_wine.py
- AdaBoost - adaboost_wine.py
- Gradient Boosting - grad_boost_wine.py
- XGBoost - xgboost_wine.py

## Future Work

- Update code such that models are chosen via command line and implemented more eloquently.
- Test algorithms on more complex datasets
- Test with more parameter choices and GPU to find better results for more complex datasets.
