# -*- coding: utf-8 -*-
"""Run"""

import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from implementations import *
from classifiers import *
from solver import *


"""
TRAIN DATA PULL
"""

# fetch train data
DATA_TRAIN_PATH = '../data/train.csv'
y, tX, ids, features = load_csv_data(DATA_TRAIN_PATH, sub_sample=False)

# adds offset
X = np.c_[np.ones(len(y)), tX]

"""
FEATURE ENGINEERING
"""

# additive binarization of NaN values
feats_binarization = []
X, features = binarize_undefined(X, features, feats_binarization)

# removing unnecessary features
feats_removal = []
X, features = remove_features(X, features, feats_removal)

# handling NaN values by mean replacement
#TODO: Anthony

# standardization
#TODO: decide whether or not we standardize

"""
FIT AND PREDICT
"""

# choice of classifier
classifier = LeastSquares()

# fitting
#classifier.fit(y, X)

# prediction
#weight = classifier.predict(X)


accuracy = cross_validate(y, X, classifier, 0.8, 10)


"""
TEST DATA PULL
"""

# fetch test data
DATA_TEST_PATH = '../data/test.csv'
y_test, tX_test, ids_test, features_test = load_csv_data(DATA_TEST_PATH)

# adds offset
X_test = np.c_[np.ones(len(y_test)), tX_test]

"""
TEST DATA FEATURE ENGINEERING
to fit dimensionality, apply feature engineering to test data
"""

# additive binarization of NaN values
X_test, features_test = binarize_undefined(X, features_test, feats_binarization)

# removing unnecessary features
X_test, features_test = remove_features(X, features_test, feats_removal)

# handling NaN values by mean replacement (with train data mean)
#TODO: Anthony

# standardization (with train data attributes)
#TODO: decide whether or not we standardize

"""
OUTPUT PREDICTIONS
"""

OUTPUT_PATH = '../results/predictions.csv'
y_pred = predict_labels(weight, tX_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
