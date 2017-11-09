np.mean(y) = 0.03 donc ca signifie que seul 3% des données sont labellées à 1 ie seuls 3% des personnes
ont un accident la première année. Donc attention on peut atteindre une très bonne accuracy alors que la prédiction est
nulle à chier. (cf résultat avec alpha=1).
Donc ce que j'ai fait c'est que j'ai mis dans la cross-entropy un facteur alpha qui permet de mettre plus d'importance
sur les colonnes labellées à 1. Ce paramètre semble avoir une bonne influence sur le comportement voulu surtout si on fait
seulement 10 époques, mais je ne sais pas forcément ce que ca vaut. (en gros l'idée c'est que l'algo se dit: si je baisse ce
poids je vais faire passer 10 prédictions de 0.20 à 0.19 ce qui est bien vu que elles doivent etre à 0.00 mais si je
monte ce poids j'ai juste une valeur à 0.5 qui va passer à 0.6. Donc l'algo ne voit pas la diff alors que en fait il
faudrait mieux chercher à mettre des valeurs à 1)


Pour l'instant ce qu'il se passe c'est que je prends toutes les colonnes qui sont catégoriques et je garde que celles
qui sont le plus corrélées avec la target. Et je garde toutes les colonnes continues. Puis on peut faire varier
facilement l'architecture dans parameters.py. Donc on peut continuer de tester différents trucs (architecture, fonction
de cout, faire varier alpha, taille des batchs et nombre d'époques).

Ce qu'il faut faire:
- tester d'autres architectures
- implémenter un modèle plus basique (genre multilinéaire quoi)
- faire de la feature selection sur les valeurs continues
- trouver un moyen de gérer le fait qu'il y a très peu de valeurs labellées à 1