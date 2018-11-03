from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import argparse

print(__doc__)

parser = argparse.ArgumentParser(description='Train classifier.')
parser.add_argument('data_path', action="store")
args = parser.parse_args()

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Abalone Data
# ./data/wine/wine.csv
df = pd.read_csv(args.data_path,
                 header=None,
                 names=['Quality',
                        'Alcohol',
                        'MalicAcid',
                        'Ash',
                        'Alcalinity',
                        'Magnesium',
                        'Phenols',
                        'Flavanoids',
                        'NonflavanoidPhenols',
                        'Proanthocyanins',
                        'ColorIntensity',
                        'Hue',
                        'OD280/OD315',
                        'Proline'])


print(df[0:5])

# Loading the Digits dataset
digits = datasets.load_digits()

# Split into our data and targets
X = df.drop(['Quality'], axis=1).values
y = df['Quality'].values


# Split the dataset in two equal parts into 80:20 ratio for train:test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# This is a key step where you define the parameters and their possible values
# that you would like to check.
tuned_parameters = [{'learning_rate': [.05, .1, .15],
                     'n_estimators': [75, 100, 125],
                     'max_depth': [2, 3, 4],
                     # 'max_features': [None, 'auto', 'sqrt', 'log2'],
                     'min_impurity_decrease': [0., 0.01, 0.1]}]

scores = ['accuracy']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(GradientBoostingClassifier(), tuned_parameters, cv=5,
                       scoring='%s' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print("Detailed confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Accuracy Score: \n")
    print(accuracy_score(y_true, y_pred))

    print()
