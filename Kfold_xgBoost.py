import time

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from util import gini_xgb
from preprocessing import preproc


####################### Data Preprocessing #####################
# Importing the training dataset
dataset_train = pd.read_csv('train.csv')

# Importing the test dataset
dataset_test = pd.read_csv('test.csv')

# preprocessing both sets
X_train, y_train = preproc(dataset_train, mode="train", oneHot=False)
X_test, y_test = preproc(dataset_test, mode="test", oneHot=False)


####################### Training #####################
i = 0
K = 5  # number of folds
kf = KFold(n_splits=K, random_state=42, shuffle=True)
# KFold Cross Validation
results = []
for train_index, test_index in kf.split(X_train):
    train_X, valid_X = X_train[train_index], X_train[test_index]
    train_y, valid_y = y_train[train_index], y_train[test_index]
    weights = np.zeros(len(y_train))
    weights[y_train == 0] = 1
    weights[y_train == 1] = 1
    print(weights, np.mean(weights))
    watchlist = [(xgb.DMatrix(train_X, train_y, weight=weights), 'train'), (xgb.DMatrix(valid_X, valid_y), 'valid')]
    # Setting parameters for XGBoost model
    params = {'eta': 0.03, 'max_depth': 4, 'objective': 'binary:logistic', 'seed': 42, 'silent': True}
    model = xgb.train(params, xgb.DMatrix(train_X, train_y, weight=weights), 1500, watchlist,  maximize=True,
                      verbose_eval=5, feval=gini_xgb, early_stopping_rounds=100)
    resy = pd.DataFrame(model.predict(xgb.DMatrix(X_test)))
    i += 1
    # Saving results for all CV models
    results.append(resy)
    # resy.to_csv(str(i)+'fold.csv')

# Creating the submission file
submission = pd.DataFrame((results[0] + results[1] + results[2] + results[3] + results[4]) / 5)
# submission.to_csv('sumbission_5kfold_xgboost.csv')
