# This example has been taken from SciKit documentation and has been
# modifified to suit this assignment. You are free to make changes, but you
# need to perform the task asked in the lab assignment


from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pandas as pd

print(__doc__)

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Abalone Data
df = pd.read_csv('./data/abalone/abalone.csv',
                 header=None,
                 names=['Sex', 'Length', 'Diam', 'Height', 'Whole',
                        'Shucked', 'Viscera', 'Shell', 'Rings'])

# One hot encode 'Sex' column
df = pd.get_dummies(df, columns=['Sex'])

# Convert all values to floats
df = df.astype(float)

print(df[0:5])

# Loading the Digits dataset
digits = datasets.load_digits()

# Split into our data and targets
X = df.drop(['Rings'], axis=1).values
y = df['Rings'].values

print(X[0:5])
print(y[0:5])

print()
print()

# Split the dataset in two equal parts into 80:20 ratio for train:test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print('Train')
print(X_train.shape)
print(y_train.shape)
print('\nTest')
print(X_test.shape)
print(y_test.shape)
print(type(X_train))
print(type(y_train))

# This is a key step where you define the parameters and their possible values
# that you would like to check.
tuned_parameters = [{'kernel': ['rbf', 'linear'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}]

scores = ['accuracy']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=2,
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

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
# n_samples = len(digits.images)
# X = digits.images.reshape((n_samples, -1))
# y = digits.target
#
# # Split the dataset in two equal parts into 80:20 ratio for train:test
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=0)
#
# # This is a key step where you define the parameters and their possible values
# # that you would like to check.
# tuned_parameters = [{'kernel': ['rbf', 'linear'], 'gamma': [1e-3, 1e-4],
#                      'C': [1, 10, 100, 1000]}
#                     ]
#
# # We are going to limit ourselves to accuracy score, other options can be
# # seen here:
# # http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
# # Some other values used are the predcision_macro, recall_macro
# scores = ['accuracy']
#
# for score in scores:
#     print("# Tuning hyper-parameters for %s" % score)
#     print()
#
#     clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
#                        scoring='%s' % score)
#     clf.fit(X_train, y_train)
#
#     print("Best parameters set found on development set:")
#     print()
#     print(clf.best_params_)
#     print()
#     print("Grid scores on development set:")
#     print()
#     means = clf.cv_results_['mean_test_score']
#     stds = clf.cv_results_['std_test_score']
#     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean, std * 2, params))
#     print()
#
#     print("Detailed classification report:")
#     print()
#     print("The model is trained on the full development set.")
#     print("The scores are computed on the full evaluation set.")
#     print()
#     y_true, y_pred = y_test, clf.predict(X_test)
#     print(classification_report(y_true, y_pred))
#     print("Detailed confusion matrix:")
#     print(confusion_matrix(y_true, y_pred))
#     print("Accuracy Score: \n")
#     print(accuracy_score(y_true, y_pred))
#
#     print()
#
# # Note the problem is too easy: the hyperparameter plateau is too flat and the
# # output model is the same for precision and recall with ties in quality.
