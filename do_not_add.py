feature_selection = "none"
number_of_features = 10
loss = "default"
alpha = 10
max_depth = 5

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
        "max_depth": max_depth
    }
}

parameters.update({
    "result": {
        "gini_score": 0.271
    }
})

import json

f = open("results.json", "r")
results_txt = f.read()
f.close()
results = json.loads(results_txt)
print(results)
# décommenter cette ligne si vous voulez sauvegarder les résultats
results.append(parameters)
f = open("results.json", "w")
f.write(json.dumps(results))
f.close()