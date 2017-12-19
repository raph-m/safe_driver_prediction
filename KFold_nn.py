# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

#import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

from util import cross_entropy, gini_normalized
from parameters import parameters, batch_size, epochs, layers, activation_functions, loss, alpha
from preprocessing import preproc

# Part 1 - Data Preprocessing
# Importing the train dataset
dataset_train = pd.read_csv('train.csv')

# Importing the test dataset
dataset_test = pd.read_csv('test.csv')

# preprocessing train dataset
X_train, y_train, scaler = preproc(dataset_train, 'train', oneHot=True, scale=True)

# preprocessing test dataset
X_test, y_test = preproc(dataset_test, 'test', oneHot=True, scale=True, scaler=scaler)

# Part 2 - Now let's make the ANN!
# Implement KFold cross validation
class_weight = {0: 1., 1: alpha}
K = 5
kf = KFold(n_splits=K, random_state=42, shuffle=True)
#training with KFold Cross Validation
i=0
results = []
for train_index, test_index in kf.split(X_train):
# Initialising the ANN
    classifier = Sequential()

    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units = layers[1], kernel_initializer = 'uniform', activation = activation_functions[0], input_dim =layers[0]))  # input dim =204 normalement

    # Adding the second hidden layer
    # no need to specify input-size since it is the output size of the previous layer
    for i in range(len(layers)-3):
        classifier.add(Dense(units=layers[i+2], kernel_initializer = 'uniform', activation = activation_functions[i+1]))

    # Adding the output layer
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = loss, metrics = [])
    train_x, train_y = X_train[train_index], y_train[train_index]
    eval_x, eval_y = X_train[test_index], y_train[test_index]
    classifier.fit(train_x, train_y, batch_size = batch_size, epochs = epochs, class_weight=class_weight)
    res_eval = classifier.predict(eval_x)
    res = classifier.predict(X_test)
    results.append(res)
    print('gini_eval', i)
    gini_score = gini_normalized(eval_y, res_eval)
    print(gini_score)
    i+=1

def to_csv(y_pred, ids):
    import csv
    with open('sumbission_5Kfold_nn.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(['id', 'target'])
        for i in range(len(y_pred)):
            spamwriter.writerow([ids[i], y_pred[i]])

submission = (results[0] + results[1] + results[2] + results[3] + results[4]) / 5
idx = dataset_test.iloc[:, 0].values
to_csv(submission[:,0],idx)
