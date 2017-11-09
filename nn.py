# Artificial Neural Network
import tensorflow as tf
# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

alpha = 50


def gini(actual, pred, cmpcol=0, sortcol=1):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)


def my_loss(actual, pred):
    return tf.reduce_mean(-tf.log(pred) * actual * alpha - tf.log(1-pred) * (1 - actual))

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, 2:-1].values

y = dataset.iloc[:, 1].values

print("mean")
print(np.mean(y))

categorical_features = [1, 3, 4, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
#  en réalité c'est: [1, 3, 4, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
#  sauf que scikit n'accepte pas les valeurs négatives apparemment en plus il y a un problème même avec les valeurs qui
#  sont apparemment positives. Je pense qu'il faut les transformer en int

X[:, 1].dtype = np.dtype(np.int32)
number_of_negatives = 0
for i in categorical_features:
    min = 0
    for j in range(len(X[:, i])):
        if X[j, i] < min:
            min = X[j, i]
    print("for feature number "+str(i)+", the minimum was: ")
    print(min)
    for j in range(len(X[:, i])):
        X[j, i] -= min
    for j in range(20):
        continue

# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = categorical_features)
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# specify input size and output size because
classifier.add(Dense(output_dim = 30, init = 'uniform', activation = 'relu', input_dim = 104))  # input dim =225 normalement

# Adding the second hidden layer
# no need to specify input-size since it is the output size of the previous layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = my_loss, metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 1000, epochs = 1)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)


print("gini normalized loss: ")
print(gini_normalized(y_test, y_pred))


y_pred = (y_pred > 0.5)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix")
print(cm)

# conclusion de ce que j'ai fait: j'ai de temps en temps de problèmes pour charger toutes les données
# (j'ai des memory errors) donc le mieux reste d'éliminer certaines variables qui ne sont pas explicatives
# Sinon quand ca marchait j'avais toujours soit que des prédictions fausses soit que des prédictions vraies
# donc soit notre modèle n'arrive pas à saisir la logique du truc et donc il met tout à zéro.
# peut etre aussi que comme il y a plus de zeros que de 1 (3% de 1) le modèle tend à mettre tout le monde
# a zero si on ne sanctionne pas le 1
