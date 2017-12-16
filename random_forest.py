import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

from util import gini_normalized, gini
from parameters import parameters, batch_size, epochs, layers, activation_functions, loss, alpha
from feature_selection_1 import get_cached_features, continuous_values
from preprocessing import preproc

# Part 1 - Data Preprocessing
# Importing the dataset
dataset = pd.read_csv('train.csv')

X, y = preproc(dataset, mode='train', oneHot=False)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Now let's make the Classifier!
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
class_weight = {0: 1., 1: alpha}
classifier = RandomForestClassifier(n_estimators=10, criterion = 'gini', random_state = 0, max_features=1, class_weight=class_weight)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

print("gini normalized score: ")
gini_score = gini_normalized(y_test, y_pred)
print(gini_score)

import numpy as np
np.savetxt("y_test", y_test)
np.savetxt("y_pred", y_pred)

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

f = open("results.json", "r")
results_txt = f.read()
f.close()
results = json.loads(results_txt)
# décommenter cette ligne si vous voulez sauvegarder les résultats
# results.append(parameters)
f = open("results.json", "w")
f.write(json.dumps(results))
f.close()
