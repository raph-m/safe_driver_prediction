import time

"/home/montaud_raphael/safe_driver_prediction/proj2/testing.csv"

from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tools import log_loss, to_csv

path_to_data = "/home/raph/Downloads/"

feature_selection = "none"
number_of_features = 10
alpha = 1.6
max_depth = 6
n_estimators = 100
loss = 'binary:logistic'  # "rank:pairwise"
subsample = 0.8
learning_rate = 0.09
min_child_weight=0.77
colsample_bytree = 0.75
gamma = 10
reg_alpha=8
reg_lambda=1.3
eval_metric='auc'

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
dataset = pd.read_csv('training.csv')
print(dataset.columns)
i = 0
msno_index = 0
is_churn_index = 0

for c in dataset.columns:
    if c=="is_churn":
        print("is_churn")
        is_churn_index = i
    if c=="msno":
        print("msno")
        msno_index = i


mask = []
for i in range(len(dataset.columns)):
    if i not in [msno_index, is_churn_index]:
        mask.append(i)

X = dataset.iloc[:, mask].values
y = dataset.iloc[:, is_churn_index].values

column_ranges = []

t1 = time.time()
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("training classifier")
classifier = XGBClassifier(
    subsample=subsample,
    max_depth=max_depth,
    scale_pos_weight=alpha,
    objective=loss,
    gamma=gamma,
    colsample_bytree=colsample_bytree,
    learning_rate=learning_rate,
    min_child_weight=min_child_weight,
    reg_alpha=reg_alpha,
    reg_lambda=reg_lambda
    #eval_metric=eval_metric
)
classifier.fit(X_train, y_train)
t2 = time.time()
print(t2-t1)

# Predicting the Test set results
y_pred = classifier.predict_proba(X_test)[:, 1]
y_pred_train = classifier.predict_proba(X_train)[:, 1]

print("gini normalized score (train): ")
log_score = log_loss(y_train, y_pred_train)
print(log_score)

print("gini normalized score (test): ")
log_score = log_loss(y_test, y_pred)
print(log_score)

print("mean de y pred")
print(np.mean(y_pred))


evaluation_dataset = pd.read_csv('testing.csv')


X_eval = evaluation_dataset.iloc[:, 2:].values  # à changer !!
y_pred_eval = classifier.predict_proba(X_eval)[:, 1]

msno = evaluation_dataset.iloc[:, 0].values

to_csv(y_pred_eval, msno)

