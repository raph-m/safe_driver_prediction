
# coding: utf-8

# In[1]:


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


# In[2]:


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


# In[15]:


testing = testing.drop(["registration_init_time"], axis=1)
testing = testing.drop(["transaction_date"], axis=1)
testing = testing.drop(["membership_expire_date"], axis=1)
training = training.drop(["registration_init_time"], axis=1)
training = training.drop(["transaction_date"], axis=1)
training = training.drop(["membership_expire_date"], axis=1)


# In[16]:


training.head()


# In[17]:


print("preprocessing")
X_train, y_train = preproc(training, mode='train', oneHot=False)
X_test, y_test = preproc(testing, mode="test", oneHot=False)

# parameters
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

i = 0
K = 5
kf = KFold(n_splits=K, random_state=42, shuffle=True)

# training with KFold Cross Validation
weights = np.zeros(len(y_train))
weights[y_train == 0] = 1
weights[y_train == 1] = 1


# In[ ]:


results = []
from tools import log_loss
print('Start training...')
for train_index, test_index in kf.split(X_train):
    lgb_train = lgb.Dataset(X_train[train_index], y_train[train_index], weight=weights[train_index])
    lgb_eval = lgb.Dataset(X_train[test_index], y_train[test_index], reference=lgb_train)
    gbm = lgb.train(params,
        train_set=lgb_train,
        num_boost_round=200,
        valid_sets=lgb_eval,
        early_stopping_rounds=30,
        verbose_eval=5,
        feval=log_loss_lgbm)
    res = gbm.predict(X_test)
    i += 1
    results.append(res)
    
    print("my log loss train")
    print(log_loss(y_train[train_index], gbm.predict(X_train[train_index])))
    print("my log loss test")
    print(log_loss(y_train[test_index], gbm.predict(X_train[test_index])))


# In[5]:


submission = pd.DataFrame((results[0] + results[1] + results[2] + results[3] + results[4]) / 5)


# In[6]:


submission.columns = ["is_churn"]


# In[7]:


submission.describe()


# In[8]:


submission["msno"] = testing["msno"]


# In[9]:


submission.describe()


# In[10]:


submission.head()


# In[11]:


submission.to_csv('5Kfold_lgbm_drop_dates.csv', header=True, index=False)
print("created submission file")


# In[ ]:


submission.describe()

