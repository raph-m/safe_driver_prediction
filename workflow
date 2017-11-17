np.mean(y) = 0.03 donc ça signifie que seul 3% des données sont labellées à 1 ie seuls 3% des personnes ont un accident la première année. Donc attention on peut atteindre une très bonne accuracy alors que la prédiction est nulle à chier. (cf résultat avec alpha=1).
Donc ce que j'ai fait c'est que j'ai mis dans la cross-entropy un facteur alpha qui permet de mettre plus d'importance sur les colonnes labellées à 1. Ce paramètre semble avoir une bonne influence sur le comportement voulu surtout si on fait seulement 10 époques, mais je ne sais pas forcément ce que ca vaut. (en gros l'idée c'est que l'algo se dit: si je baisse ce
poids je vais faire passer 10 prédictions de 0.20 à 0.19 ce qui est bien vu que elles doivent etre à 0.00 mais si je monte ce poids j'ai juste une valeur à 0.5 qui va passer à 0.6. Donc l'algo ne voit pas la diff alors que en fait il faudrait mieux chercher à mettre des valeurs à 1)


Pour l'instant ce qu'il se passe c'est que je prends toutes les colonnes qui sont catégoriques et je garde que celles
qui sont le plus corrélées avec la target. Et je garde toutes les colonnes continues. Puis on peut faire varier
facilement l'architecture dans parameters.py. Donc on peut continuer de tester différents trucs (architecture, fonction
de cout, faire varier alpha, taille des batchs et nombre d'époques).

Ce qu'il faut faire:
- tester d'autres architectures
- implémenter un modèle plus basique (genre multilinéaire quoi)
- faire de la feature selection sur les valeurs continues (une facon de selectionner c'est de faire
    une régression linéaire avec juste une feature et de voir quelles features ont le plus gros taux
    de corrélation) (c'est la priorité la je pense) c'est peut etre possible de faire de la feature
    selection sur les données catégoriques et continues en même temps
- trouver un moyen de gérer le fait qu'il y a très peu de valeurs labellées à 1 (normalement c'est géré
    par l'argument 'weight_labels'
- notamment il faudrait deja voir les resultats avec une simple régression linéaire


une page intéressante pour répondre à notre problème (imbalanced classes):
https://svds.com/learning-imbalanced-classes/

comprendre le Gini coefficient:
https://www.kaggle.com/batzner/gini-coefficient-an-intuitive-explanation
en gros ca représente la quantité de 'swap' qu'il faudrait faire pour remettre nos predictions
dans le bon ordre

une analyse préliminaire des données:
https://www.kaggle.com/headsortails/steering-wheel-of-fortune-porto-seguro-eda

J'ai fait un algo random Forest mais c'est complètement nul en critère de Gini.
Deja si on ajoute l'argument 'class_weight' on a un résultat un peu plus honnête.

Conclusion du fichier random_prediciton.py: en fait ce fichier teste l'index de Gini si jamais on
fait des prédictions random. En trouve qu'on peut monter jusqu'a 0.02 en Gini normalisé avec une
distribution random.

Maintenant je vais tenter XGBoost. Ici un petit lien qui explique comment fonctionnent les arbres de décisions
de XGBoost:
http://xgboost.readthedocs.io/en/latest/model.html
Sinon je vous expliquerai comment ca marche c'est pas très compliqué.

Résultats XGBoost: j'ai fait un commit avec la première fois que j'atteins un score honnete (0.22
un truc comme ca). Ensuite j'obtiens un bon résultat (0.271) avec la loss par défaut et max_depth de 5,
et pas de feature selection + un alpha qui vaut 10.


Dans tools il y a maintenant moyen de faire un tracé de cap curve. Je vous laisse vous renseigner sur
ce qu'est la cap curve, en tout cas il y a le tracé du pire prédicteur (random, c'est la ligne droite)
et du meilleur (celui qui fait un angle). Donc plus l'aire entre le pire et notre modèle est bon et plus
le prédicteur est bon !

Normalement avec XGBoost on peut obtenir les résultats maximum. Donc pour trouver la valeur idéale des
paramètres il faudrait regarder sur Kaggle.
Ce que je constate pour l'instant c'est un sérieux overfitting (0.359 -> 0.271).
Donc la on a un soucis, il faut checker combien d'estimateurs il y a dans ce truc. (apparemment 100)

résultats de référence:
train: 0.359
test: 0.271
avec: feature_selection = "none"
number_of_features = 10
loss = "reg:linear"
alpha = 32
max_depth = 5
n_estimators = 100

en fait si je passe à 200 estimators j'augmente l'overfitting ... bizarre
(et je fais baisser le score de test)
si je passe à 50 estimateurs je fais baisser l'overfitting mais aussi le score...
Donc rester autour de 100 estimateurs parait bien.


quelques notes sur comment gérer l'overfitting:
http://xgboost.readthedocs.io/en/latest/how_to/param_tuning.html


Si on change de modèle et qu'on passe en "rank:pairwise" on obtient des résultats similaires
mais avec moins d'overfitting.

Pour faire baisser l'overfitting le lien plus haut conseille de mettre le subsampling à une valeur
inférieure à 1. Apparemment ca diminue le nombre de données prises en compte ?? wtf je vois
pas le rapport. Bref ca augmente légèrement les resultats de train et de test. Enfin bref je suis
sceptique par rapport à ce truc.


un kernel Kaggle intéressant:
https://www.kaggle.com/tendolkar3/no-magic-0-283-lb-detailed-w-data-exploration/notebook
ce mec donne des paramètres qui marchent bien pour un XGBoost
il donne aussi une méthode intéressante:
faire trois arbres LightGBM assez différents et les faire voter.

un lien sur LightGBM:
https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/

j'ai essayé de faire du OnehotEncoding sur XGBoost mais ca pose un problème de mémoire je pense
ca me met cette erreur:
>>>
terminate called after throwing an instance of 'std::bad_alloc'
  what():  std::bad_alloc

Process finished with exit code 134 (interrupted by signal 6: SIGABRT)
>>>


j'ai réussi à faire tourner un modèle LightGBM qui donne un résultat assez satisfaisant (0.25)
sans que j'ai touché aux paramètres. Ca a l'air d'être une bonne piste (notamment de combiner
différents arbres);
