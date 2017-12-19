feature_selection = "infogain"
number_of_features = 10
batch_size = 4000
epochs = 2
#layers = [177, 30, 10, 3, 1]
layers = [177, 80, 20, 10, 1]
dropout = [0.35, 0.15, 0.15]
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