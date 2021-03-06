#import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from util import gini_lgbm
from preprocessing import preproc


####################### Data Preprocessing #####################
# Importing the dataset
dataset_train = pd.read_csv('train.csv')

# Importing the test dataset
dataset_test = pd.read_csv('test.csv')

# Preprocessing both tests
X_train, y_train = preproc(dataset_train, mode='train', oneHot=False)
X_test, y_test = preproc(dataset_test, mode="test", oneHot=False)


####################### Training #####################
# parameters
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'gini'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

i = 0
K = 5
kf = KFold(n_splits=K, random_state=42, shuffle=True)
#training with KFold Cross Validation
results = []
weights = np.zeros(len(y_train))
weights[y_train == 0] = 1
weights[y_train == 1] = 2
#print('Start training...')
for train_index, test_index in kf.split(X_train):
    lgb_train = lgb.Dataset(X_train[train_index], y_train[train_index], weight=weights[train_index])
    lgb_eval = lgb.Dataset(X_train[test_index], y_train[test_index], reference=lgb_train, weight=weights[test_index])
    gbm = lgb.train(params,
        train_set=lgb_train,
        num_boost_round=200,
        valid_sets=lgb_eval,
        early_stopping_rounds=10,
        verbose_eval=5,
        feval=gini_lgbm)
    res = gbm.predict(X_test)
    i+=1
    results.append(res)

def to_csv(y_pred, ids):
    import csv
    with open('sumbission_5Kfold_lightgbm.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(['id', 'target'])
        for i in range(len(y_pred)):
            spamwriter.writerow([ids[i], y_pred[i]])

submission = (results[0] + results[1] + results[2] + results[3] + results[4]) / 5
idx = dataset_test.iloc[:, 0].values
to_csv(submission[:,0],idx)
