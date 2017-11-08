import numpy as np
import csv

# opening and reading train and test dataset
# dataset size: train 595213 rows and 59 columns. first column is diver_id, and second column is target (0 or 1). The other columns are features
# dataset size : test 892817 rows and 58 columns. same structure as train dataset without target column.

def read_dataset(size_training, size_testing):
    train_data = np.loadtxt('train.csv', delimiter = ',', skiprows = 1)
    test_data = np.loadtxt('test.csv', delimiter = ',', skiprows = 1)

    total_training = train_data.shape[0]
    if(size_training > total_training):
        size_training = total_training
    total_testing = test_data.shape[0]
    if(size_testing > total_testing):
        size_testing = total_testing

    rd_train_inst = list(range(total_training))
    np.random.shuffle(rd_train_inst)
    rd_test_inst = list(range(total_testing))
    np.random.shuffle(rd_test_inst)

    return train_data[rd_train_inst[0:size_training],:], test_data[rd_test_inst[0:size_testing],:]
