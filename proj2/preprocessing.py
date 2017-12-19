def preproc(dataset, mode, oneHot):

    is_churn = 0
    msno = 0
    selected_features = []
    for i in range(len(dataset.columns)):
        if dataset.columns[i] == "is_churn":
            is_churn = i
        elif dataset.columns[i] == "msno":
            msno = i
        else:
            selected_features.append(i)

    X = dataset.iloc[:, selected_features].values
    y = dataset.iloc[:, is_churn].values

    return X, y
