import lightgbm as lgb
import pandas as pd
import numpy as np


num_boost_round = 300

train_master = pd.read_csv('train.csv')
test_master = pd.read_csv('test.csv')

# train_master.describe()
np.random.seed(3)
model_scores = {}

# Drop binary columns with almost all zeros.
# Why now? Just follow along for now. We have a lot of experimentation to be done
train = train_master.drop(['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_13_bin'], axis=1)
test = test_master.drop(['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_13_bin'], axis=1)

# Drop calculated features
# But WHY???
# Because we are assuming that tree can generate any complicated function
# of base features and calculated features add no more information
# Is this assumption valid? Results will tell
calc_columns = [s for s in list(train_master.columns.values) if '_calc' in s]
train = train.drop(calc_columns, axis=1)
test = test.drop(calc_columns, axis=1)

# Get categorical columns for encoding later
categorical_columns = [s for s in list(train_master.columns.values) if '_cat' in s]
target_column = 'target'

# Replace missing values with NaN
train = train.replace(-1, np.nan)
test = test.replace(-1, np.nan)

# Initialize DS to store validation fold predictions
y_val_fold = np.empty(len(train))

# Initialize DS to store test predictions with aggregate model and individual models
y_test = np.zeros(len(test))
y_test_model_1 = np.zeros(len(test))
y_test_model_2 = np.zeros(len(test))
y_test_model_3 = np.zeros(len(test))


def encode_cat_features(train_df, test_df, cat_cols, target_col_name, smoothing=1):
    prior = train_df[target_col_name].mean()
    probs_dict = {}
    for c in cat_cols:
        probs = train_df.groupby(c, as_index=False)[target_col_name].mean()
        probs['counts'] = train_df.groupby(c, as_index=False)[target_col_name].count()[[target_col_name]]
        probs['smoothing'] = 1 / (1 + np.exp(-(probs['counts'] - 1) / smoothing))
        probs['enc'] = prior * (1 - probs['smoothing']) + probs['target'] * probs['smoothing']
        probs_dict[c] = probs[[c, 'enc']]
    return probs_dict


# Encode categorical variables using training fold
encoding_dict = encode_cat_features(train, 1, categorical_columns, target_column)

for c, encoding in encoding_dict.items():
    train = pd.merge(train, encoding[[c, 'enc']], how='left', on=c, sort=False, suffixes=('', '_' + c))
    train = train.drop(c, axis=1)
    train = train.rename(columns={'enc': 'enc_' + c})

    test = pd.merge(test, encoding[[c, 'enc']], how='left', on=c, sort=False, suffixes=('', '_' + c))
    test = test.drop(c, axis=1)
    test = test.rename(columns={'enc': 'enc_' + c})

from sklearn.model_selection import train_test_split
X_train, X_eval, y_train, y_eval = train_test_split(train.iloc[:, 2:].values, train.iloc[:, 1].values, test_size=0.2, random_state=0)

X_test = test.iloc[:, 1:].values

# Define parameters of GBM as explained before for 3 trees
params_1 = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'auc',
    'max_depth': 3,
    'learning_rate': 0.05,
    'feature_fraction': 1,
    'bagging_fraction': 1,
    'bagging_freq': 10,
    'verbose': 0,
    'scale_pos_weight': 20  # n'est pris en compte que quand c'est 'binary'
}


# Create appropriate format for training and evaluation data
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)

# Create the 3 classifiers with 1000 rounds and a window of 100 for early stopping
clf_1 = lgb.train(params_1, lgb_train, num_boost_round=num_boost_round,
                  valid_sets=lgb_eval, early_stopping_rounds=100, verbose_eval=50)

from util import gini_normalized
# Predict raw scores for validation ids
y_eval_pred = clf_1.predict(X_eval, raw_score=True)

print("Gini eval mean on all trees: ")
print(gini_normalized(y_eval, y_eval_pred))

y_train_pred = clf_1.predict(X_train, raw_score=True)

print("Gini train mean on all trees: ")
print(gini_normalized(y_train, y_train_pred))

c1 = 0
c2 = 0

temp = y_train_pred < 0.2
indexes_to_retrain = [i for i in range(temp) if y_train_pred[i]]
X_to_retrain = X_train[indexes_to_retrain]
y_to_retrain = y_train[indexes_to_retrain]

clf_2 = lgb.train(params_1, lgb_train, num_boost_round=num_boost_round,
                  valid_sets=lgb_eval, early_stopping_rounds=100, verbose_eval=50)

y_test_pred = (clf_1.predict(X_test, raw_score=True))

clf_1.save_model('clf_1.txt')

print(y_test_pred)
np.savetxt("y_test_pred", y_test_pred)

ids = test.iloc[:, 0].values

from tools import to_csv

# to_csv(y_test_pred, ids)

