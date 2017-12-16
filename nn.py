# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

from util import cross_entropy, gini_normalized
from parameters import parameters, batch_size, epochs, layers, activation_functions, loss, alpha
from feature_selection_1 import get_cached_features, continuous_values
from preprocessing import preproc

if loss == "cross_entropy":
    loss = cross_entropy

# Part 1 - Data Preprocessing
# Importing the dataset
dataset = pd.read_csv('train.csv')

# preprocessing
X, y = preproc(dataset, 'train', oneHot=True)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = layers[1], kernel_initializer = 'uniform', activation = activation_functions[0], input_dim =layers[0]))  # input dim =225 normalement

# Adding the second hidden layer
# no need to specify input-size since it is the output size of the previous layer
for i in range(len(layers)-3):
    classifier.add(Dense(units=layers[i+2], kernel_initializer = 'uniform', activation = activation_functions[i+1]))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = loss, metrics = [])

# Fitting the ANN to the Training set
class_weight = {0: 1., 1: alpha}
classifier.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, class_weight=class_weight)

# Part 3 - Making the predictions and evaluating the model

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
