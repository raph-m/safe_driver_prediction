feature_selection = "infogain"
number_of_features = 10
batch_size = 10000
epochs = 10
layers = [177, 30, 10, 3, 1]
activation_functions = ["relu", "relu", "relu", "sigmoid"]
loss = "binary_crossentropy"
alpha = 30

parameters = {
    "feature_selection": {
        "name": feature_selection,
        "number_of_features": number_of_features
    },
    "classifier": {
        "name": "neural_network",
        "batch_size": batch_size,
        "epochs": epochs,
        "layers": layers,
        "activation_functions": activation_functions,
        "loss":
            {
                "name": loss,
                "alpha": alpha
            }
    }
}