import time

import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_selection_1 import get_cached_features


def preproc(dataset, mode, oneHot, scale=False, scaler=None):
    # categorical and binary features
    categorical_features = [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                            33, 53, 54, 55, 56, 57, 58]

    # continuous values
    continuous_values = [2, 4, 15, 16, 20, 21, 22, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                         51, 52]

    feature_selection = "none"
    number_of_features = 10
    alpha = 32
    max_depth = 4
    n_estimators = 100
    loss = "rank:pairwise"
    parameters = {
        "feature_selection": {
            "name": feature_selection,
            "number_of_features": number_of_features
        },
        "classifier": {
            "name": "xgboost",
            "loss":
                {
                    "name": loss,
                    "alpha": alpha
                },
            "max_depth": max_depth,
            "n_estimators": n_estimators
        }
    }

    if feature_selection == "infogain":
        categorical_features = get_cached_features(parameters["feature_selection"])
        continuous_values = []

    categorical_features_count = len(categorical_features)

    if mode == 'train':
        selected_features = categorical_features + continuous_values
    elif mode == 'test':
        selected_features = categorical_features + continuous_values
        selected_features = np.array(selected_features) - 1
    else:
        Warning("Mode must be train or set, otherwise it will lead to local"
                " a variable referenced before assignment error")

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

    if oneHot:
        print("One hot encoding")
        # NB: les nouvelles colonnes sont placées juste devant l'autre colonne
        onehotencoder = OneHotEncoder(categorical_features=range(categorical_features_count))
        X = onehotencoder.fit_transform(X).toarray()

        # pour éviter le piège de la "dummy variable" on va retirer une colonne pour chaque ajout de colonnes
        # typiquement pour les trucs binaires on aurait pas du rajouter de colonnes (on verra ca plus tard)
        # donc la c'est un peu technique, il faudra relire ca en détail

        to_delete = []
        t = 0

        for i in range(categorical_features_count):
            to_delete.append(t)
            t += column_ranges[i]

        mask = []

        for s in range(len(X[0, :])):
            if s not in to_delete:
                mask.append(s)

        X = X[:, mask]

        if scale:
            if scaler == None:
                sc = StandardScaler()
                X = sc.fit_transform(X)
                return X, y, sc
            else:
                X = scaler.transform(X)
    return X, y
