import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

from util import cross_entropy, gini_normalized, gini
from parameters import parameters, batch_size, epochs, layers, activation_functions, loss, alpha
from feature_selection_1 import get_cached_features, continuous_values

if loss == "cross_entropy":
    loss = cross_entropy

# Part 1 - Data Preprocessing
# Importing the dataset
dataset = pd.read_csv('train.csv')

# feature selection
categorical_features = get_cached_features(parameters["feature_selection"])

categorical_features_count = len(categorical_features)
selected_features = categorical_features# + continuous_values

X = dataset.iloc[:, selected_features].values
y = dataset.iloc[:, 1].values

column_ranges = []

print("replacing missing values")
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

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


for i in range(10):
    y_pred = np.random.rand(len(y_test))
    print("gini normalized score: ")
    gini_score = gini_normalized(y_test, y_pred)
    print(gini_score)

y_pred = np.zeros(len(y_test))
print("gini normalized score: ")
gini_score = gini_normalized(y_test, y_pred)
print(gini_score)

y_pred = np.ones(len(y_test))
print("gini normalized score: ")
gini_score = gini_normalized(y_test, y_pred)
print(gini_score)

print("mean de y pred")
print(np.mean(y_pred))
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix")
print(cm)

parameters.update({
    "result": {
        "tp": int(cm[0, 0]),
        "tn": int(cm[1, 1]),
        "fp": int(cm[1, 0]),
        "fn": int(cm[0, 1]),
        "gini_score": gini_score
}})

