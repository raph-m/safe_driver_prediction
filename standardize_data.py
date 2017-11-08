import numpy as np
#standardize the non binary datas
def standardize(train_data, test_data):
    train_nonBin_col = [2, 3, 4, 5, 6, 15, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]
    test_nonBin_col =[1, 2, 3, 4, 5, 14, 15, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    for i in train_nonBin_col:
        mini = np.min(train_data[:,i])
        maxi = np.max(train_data[:,i])
        train_data[:,i] = (train_data[:,i] - mini)/(maxi-mini)
    for i in test_nonBin_col:
        print(test_data[:,i],i)
        mini = np.min(test_data[:,i])
        maxi = np.max(test_data[:,i])
        test_data[:,i] = (test_data[:,i] - mini)/(maxi-mini)
    return train_data, test_data
