Le fichier main contient le launcher python

Dans ce fichier la variable useAudioAnalysCSV determine si les features audio doivent être regénérées (long) ou si
elles doivent être récupérées dans un fichier csv (plus rapide)


La fonction evaluation.randomizedSearchCV(X,y) permet de trouver les meilleurs paramètres à partir du jeu de donnée

Pour que seeError() affiche le nom des fichiers ce champ doit être conservé assez longtemps dans la chaîne d'appel.
Ne pas oublier de le supprimer avec de fit ou classify.


#Améliorations :

Le champ manually_verified n'est pas pas utilisé, c'est une piste d'amélioration potentiel.
Une réseau neuronal pourrait être mis en place pour améliorer le score.