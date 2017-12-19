#import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from util import gini_normalized, gini_lgbm
from preprocessing import preproc


####################### Data Preprocessing #####################
# Importing the dataset
dataset_train = pd.read_csv('train.csv')

# Importing the test dataset
dataset_test = pd.read_csv('test.csv')

# Preprocessing both tests
X_train, y_train = preproc(dataset_train, mode='train', oneHot=False)
X_test, y_test = preproc(dataset_test, mode="test", oneHot=False)

# Splitting the dataset into the Training set and Test set
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# create dataset for lightgbm
#lgb_train = lgb.Dataset(X_train, y_train)
#lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

####################### Training #####################
# parameters
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'auc'},
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
#print('Start training...')
for train_index, test_index in kf.split(X_train):
    lgb_train = lgb.Dataset(X_train[train_index], y_train[train_index])
    lgb_eval = lgb.Dataset(X_train[test_index], y_train[test_index], reference=lgb_train)
    gbm = lgb.train(params,
        lgb_train,
        num_boost_round=100,
        valid_sets=lgb_eval,
        early_stopping_rounds=10,
        verbose_eval=5,
        feval = gini_lgbm)
    res = gbm.predict(X_test)
    i+=1
    results.append(res)

submission = pd.DataFrame((results[0] + results[1] + results[2] + results[3] + results[4]) / 5)

####################### Prediction #####################
#print('Start predicting...')
#y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
#y_pred_train = gbm.predict(X_train, num_iteration=gbm.best_iteration)

#print("gini normalized score (train): ")
#gini_score = gini_normalized(y_train, y_pred_train)
#print(gini_score)

#print("gini normalized score (test): ")
#gini_score = gini_normalized(y_test, y_pred)
#print(gini_score)

#np.savetxt("y_test", y_test)
#np.savetxt("y_pred", y_pred)

#np.savetxt("y_train", y_test)
#np.savetxt("y_pred_train", y_pred)

#print("mean de y pred")
#print(np.mean(y_pred))

#parameters.update({
#    "result": {
#        "gini_score": gini_score
#}})

#f = open("results.json", "r")
#results_txt = f.read()
#f.close()
#results = json.loads(results_txt)
# décommenter cette ligne si vous voulez sauvegarder les résultats
# results.append(parameters)
#f = open("results.json", "w")
#f.write(json.dumps(results))
#f.close()
