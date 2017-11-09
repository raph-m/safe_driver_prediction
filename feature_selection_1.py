#  feature selection on categorical data using infogain or chisquared:

# Feature selection with the Information Gain measure

import numpy as np
from math import log
import pandas as pd
import json


def entropy(s):
    values = np.unique(s)
    h = 0.0
    for v in values:
        p = np.sum(s == v) / len(s)
        h -= p * log(p, 2)
    return h


def infogain(x, y):
    """
        x: features (data)
        y: output (classes)
    """
    info_gains = np.zeros(x.shape[1])  # features of x

    # calculate entropy of the data *hy* with regards to class y
    hy = entropy(y)

    # calculate the information gain for each column (feature)
    dim = x.shape[1]
    for j in range(dim):
        current_h = 0
        if j % 20 == 0:
            print(str(int(100.0 * j / dim)) + "%")
        column = x[:, j]
        for v in np.unique(column):
            indexes = (column == v)
            current_h += entropy(y[indexes]) * len(y[indexes])
        info_gains[j] = hy - current_h / len(y)
    return info_gains


def select_categorical_features(number_of_features = 10, method = "infogain"):

    dataset = pd.read_csv('train.csv')
    categorical_features = [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 53, 54, 55, 56, 57, 58]

    X = dataset.iloc[:, categorical_features].values

    y = dataset.iloc[:, 1].values

    if method == "infogain":
        gain = infogain(X, y)
    else:
        return
    print("gain")
    print(gain)
    index = np.argsort(gain)[::-1]

    real_indexes = []

    print("index")
    print(index)

    for i in index:
        real_indexes.append(categorical_features[i])

    print("real indexes")
    return real_indexes[:number_of_features]


def get_cached_features(feature_selection, recompute=False):
    f = open("feature_selection_cache.json", "r")
    cache = json.loads(f.read())
    f.close()
    name = feature_selection["name"]
    number_of_features = feature_selection["number_of_features"]
    if not recompute:
        for v in cache:
            if v["name"] == name and v["number_of_features"] == number_of_features:
                print("getting feature selection from cache")
                return v["indexes"]

    print("computing feature selection")
    if name == "infogain":
        indexes = select_categorical_features(number_of_features, "infogain")
    else:
        return

    found = False
    for v in cache:
        if v["name"] == name and v["number_of_features"] == number_of_features:
            v.update({"indexes": indexes})
            found = True
            break
    if not found:
        cache.append({
            "name": name,
            "feature_selection": feature_selection,
            "indexes": indexes
        })
    f = open("feature_selection_cache.json", "w")
    f.write(json.dumps(cache))
    f.close()
    return indexes
