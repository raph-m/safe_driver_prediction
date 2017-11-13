import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from util import gini_normalized
from feature_selection_1 import get_cached_features, continuous_values, categorical_features

feature_selection = "none"
number_of_features = 10
alpha = 32
max_depth = 4
n_estimators = 100
loss = "rank:pairwise"
subsample = 0.8
learning_rate = 0.05
colsample_bytree = 0.8
gamma = 9

parameters = {
    "feature_selection": {
        "name": feature_selection,
        "number_of_features": number_of_features
    },
    "classifier": {
        "name": "xgboost",
        "loss":
            {
                "name": loss,
                "alpha": alpha
            },
        "max_depth": max_depth,
        "n_estimators": n_estimators
    }
}

# Part 1 - Data Preprocessing
# Importing the dataset
dataset = pd.read_csv('train.csv')

# feature selection
if feature_selection == "infogain":
    categorical_features = get_cached_features(parameters["feature_selection"])
    continuous_values = []

categorical_features_count = len(categorical_features)
selected_features = categorical_features + continuous_values


X = dataset.iloc[:, selected_features].values
y = dataset.iloc[:, 1].values

column_ranges = []

print("replacing missing values")
t0 = time.time()
print("number of examples: "+str(len(X[:, 0])))
for i in range(len(X[0, :])):
    if i <= categorical_features_count:
        # si c'est une variable de catégories, on prend comme stratégie de remplacer par la
        # valeur la plus fréquente
        (values, counts) = np.unique(X[:, i], return_counts=True)
        counts = [counts[i] if values[i] >= 0 else 0 for i in range(len(values))]
        ind = np.argmax(counts)
        column_ranges.append(max(values))
        replacement_value = values[ind]
    else:
        # sinon on prend simplement la moyenne
        replacement_value = np.mean(X[:, i])

    for j in range(len(X[:, i])):
        if X[j, i] < -0.5:
            X[j, i] = replacement_value
t1 = time.time()
print(t1-t0)
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# specify your configurations as a dict
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

print('Start training...')
# train
t2 = time.time()
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)
t3 = time.time()
print(t3-t2)

print('Save model...')
# save model to file
gbm.save_model('model.txt')

print('Start predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
y_pred_train = gbm.predict(X_train, num_iteration=gbm.best_iteration)

print("gini normalized score (train): ")
gini_score = gini_normalized(y_train, y_pred_train)
print(gini_score)

print("gini normalized score (test): ")
gini_score = gini_normalized(y_test, y_pred)
print(gini_score)

import numpy as np
np.savetxt("y_test", y_test)
np.savetxt("y_pred", y_pred)

np.savetxt("y_train", y_test)
np.savetxt("y_pred_train", y_pred)

print("mean de y pred")
print(np.mean(y_pred))

parameters.update({
    "result": {
        "gini_score": gini_score
}})

f = open("results.json", "r")
results_txt = f.read()
f.close()
results = json.loads(results_txt)
# décommenter cette ligne si vous voulez sauvegarder les résultats
# results.append(parameters)
f = open("results.json", "w")
f.write(json.dumps(results))
f.close()


