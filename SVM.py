import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm

from util import gini_normalized
from feature_selection_1 import get_cached_features, continuous_values, categorical_features

feature_selection = "none"
number_of_features = 10
n_estimators = 100
#SVC parameters
C=1.0
kernel = 'rbf' #or 'linear', 'poly', 'sigmoid', 'precomputed', custom (callable)
degree = 3 #only if kernel = 'poly'
gamma='auto' #rbf, poly or sigmoid, default 1/n_features
class_weight = {1:3, 0:1} #weight of given labels
cach_size=1000

parameters = {
    "feature_selection": {
        "name": feature_selection,
        "number_of_features": number_of_features
    },
    "classifier": {
        "name": "SVM",
        "kernel":
            {
                "name": kernel,
                "degree": degree,
                "gamma":gamma
            },
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

print("training classifier")
classifier = svm.SVC(
    C=C,
    kernel=kernel,
    degree=degree,
    gamma=gamma,
    class_weight=class_weight,
    cach_size=cach_size
    )

t2 = time.time()
classifier.fit(X_train, y_train)
t3 = time.time()
print(t3-t2)

# Predicting the Test set results
y_pred = classifier.predict_proba(X_test)[:, 1]
y_pred_train = classifier.predict_proba(X_train)[:, 1]

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
y_pred = (y_pred > 0.5)

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
