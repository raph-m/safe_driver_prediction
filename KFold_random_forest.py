#import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from util import gini_normalized
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


# Now let's make the Classifier!
# Fitting Random Forest Classification to the Training set

class_weight = {0: 1., 1: alpha}
K = 5
kf = KFold(n_splits=K, random_state=42, shuffle=True)
#training with KFold Cross Validation
i=0
results = []
for train_index, test_index in kf.split(X_train):
    train_x, train_y = X_train[train_index], y_train[train_index]
    eval_x, eval_y = X_train[test_index], y_train[test_index]
    classifier = RandomForestClassifier(n_estimators=30, criterion = 'gini', random_state = 1, max_depth=5, max_features='auto', class_weight=class_weight)
    classifier.fit(train_x, train_y)
    res_train = classifier.predict(train_x)
    res_eval = classifier.predict(eval_x)
    res = classifier.predict(X_test)
    results.append(res)
    print('round k=',i)
    print('eval gini score  ', 'train gini score')
    gini_eval = gini_normalized(eval_y, res_eval)
    gini_train = gini_normalized(train_y, res_train)
    print(gini_eval,'  ', gini_train)
    print()
    i+=1


def to_csv(y_pred, ids):
    import csv
    with open('sumbission_5Kfold_random_forest.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(['id', 'target'])
        for i in range(len(y_pred)):
            spamwriter.writerow([ids[i], y_pred[i]])

submission = (results[0] + results[1] + results[2] + results[3] + results[4]) / 5
idx = dataset_test.iloc[:, 0].values
to_csv(submission[:,0],idx)
