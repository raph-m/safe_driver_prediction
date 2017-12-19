import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from preprocessing import preproc
from tools import log_loss_lgbm

from datetime import datetime
import time

path_to_data = "/home/raph/Downloads/"

print("loading data")
training = pd.read_csv(path_to_data+"training.csv")
testing = pd.read_csv(path_to_data+"testing.csv")

print("changing dates to time stamps")
training["membership_expire_date"] = training.membership_expire_date.apply(lambda x: datetime.strptime(str(x), "%Y-%m-%d").date() if pd.notnull(x) else x)
training["membership_expire_date"] = training.membership_expire_date.apply(lambda x: time.mktime(x.timetuple()) if pd.notnull(x) else 0.0)

training["transaction_date"] = training.transaction_date.apply(lambda x: datetime.strptime(str(x), "%Y-%m-%d").date() if pd.notnull(x) else x)
training["transaction_date"] = training.transaction_date.apply(lambda x: time.mktime(x.timetuple()) if pd.notnull(x) else 0.0)

training["registration_init_time"] = training.registration_init_time.apply(lambda x: datetime.strptime(str(x), "%Y-%m-%d").date() if pd.notnull(x) else x)
training["registration_init_time"] = training.registration_init_time.apply(lambda x: time.mktime(x.timetuple()) if pd.notnull(x) else 0.0)

testing["membership_expire_date"] = testing.membership_expire_date.apply(lambda x: datetime.strptime(str(x), "%Y-%m-%d").date() if pd.notnull(x) else x)
testing["membership_expire_date"] = testing.membership_expire_date.apply(lambda x: time.mktime(x.timetuple()) if pd.notnull(x) else 0.0)

testing["transaction_date"] = testing.transaction_date.apply(lambda x: datetime.strptime(str(x), "%Y-%m-%d").date() if pd.notnull(x) else x)
testing["transaction_date"] = testing.transaction_date.apply(lambda x: time.mktime(x.timetuple()) if pd.notnull(x) else 0.0)

testing["registration_init_time"] = testing.registration_init_time.apply(lambda x: datetime.strptime(str(x), "%Y-%m-%d").date() if pd.notnull(x) else x)
testing["registration_init_time"] = testing.registration_init_time.apply(lambda x: time.mktime(x.timetuple()) if pd.notnull(x) else 0.0)

print("preprocessing")
X_train, y_train = preproc(training, mode='train', oneHot=False)
X_test, y_test = preproc(testing, mode="test", oneHot=False)

# parameters
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'binary_logloss'},
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

# training with KFold Cross Validation
results = []
weights = np.zeros(len(y_train))
weights[y_train == 0] = 1
weights[y_train == 1] = 2

print('Start training...')
for train_index, test_index in kf.split(X_train):
    lgb_train = lgb.Dataset(X_train[train_index], y_train[train_index], weight=weights[train_index])
    lgb_eval = lgb.Dataset(X_train[test_index], y_train[test_index], reference=lgb_train, weight=weights[test_index])
    gbm = lgb.train(params,
        train_set=lgb_train,
        num_boost_round=200,
        valid_sets=lgb_eval,
        early_stopping_rounds=10,
        verbose_eval=5,
        feval=log_loss_lgbm)
    res = gbm.predict(X_test)
    i += 1
    results.append(res)

submission = pd.DataFrame((results[0] + results[1] + results[2] + results[3] + results[4]) / 5)
submission.to_csv('5Kfold_lgbm.csv')
print("created submission file")
