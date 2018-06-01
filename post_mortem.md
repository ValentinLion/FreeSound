#Exécution
Le fichier main contient le launcher python.

La ligne description.description(pd.read_csv("csv/train.csv", sep=",")) dans main.py lance la phase de description
du jeu de donnée.

L'utilisation du TreeJSON qui permet aussi de décrire les données est expliquée dans TreeJSON/README.md

Toutes les features audio ont leur propre fonction qui renvoie une Serie Pandas. Il est ainsi facile de gérer les
features que l'on souhaite utiliser. La fonction get_features concatène les Serie des différentes features.

L'analyse des features est extrêmement longue (2h). Nous avons donc mis en place un système qui permet d'écrire dans
un fichier csv les features extraites. On peut donc par la suite utiliser directement ce fichier et ainsi passer la
phase d'extraction. Pour cela, il existe la variable useAudioAnalysCSV dans main.py.

La fonction evaluation.randomizedSearchCV(X,y) permet de trouver les meilleurs paramètres à partir du jeu de donnée

Pour que seeError() affiche le nom des fichiers, le champ fname doit être conservé assez longtemps dans la chaîne 
d'appel.
Ne pas oublier de le supprimer avec de fit ou classify.


#Choix des features

Nous utilisons principalement MFCC. Cette technique est largement utilisée dans la reconnaissance vocale.
Nous récupérons la moyenne, la dérivée, le min/max, le skew et le kurtosis de ces coefficient ce qui nous permet
d'avoir une empreinte sonore assez précise de chaque son.

Nous avons rajouté d'autres features à partir de fonctions provenant de la librairie audio librosa comme Zero crossing,
Spectral Contrast, Spectral bandwidth, Spectral centroid,Spectral rolloff.


#Machine learning

Le problème est un choix de label. L'utilisation d'un classifier est donc logique. Après quelques tests nous nous sommes
dirigés vers RandomForest de sklearn qui donnait les meilleurs résultats avec peu de temps de configuration. D'autres
personnes participant à la compétition ont eu d'excellents résultats avec des réseaux de neurones.

Nous transformons les labels en entier pour qu'ils puissent être utilisés par les outils de machine learning.
Nous séparons le jeu de donnée (20% test, 80% train). Nous l'entraînons et affichons le résultat de la prédiction sur
le test ainsi que les errors de prédiction dans un fichier csv. Ensuite, nous entraînons notre classifier sur l'ensemble
des données.

Pour renvoyer trois labels de prédiction nous utilisons la fonction de RandomForrest predict_proba qui renvoie un
tableau des probabilités pour chaque label. Nous trions ensuite ce tableau et récupérons les trois labels avec
la probabilité la plus forte.


#Améliorations :

Nous avons manqué de temps pour pousser les outils de machine learning. La récupération des features ayant été très
longue.

La recherche de meilleurs paramètres pour le RandomForrest ou le passage à un réseau neuronale pourrait
améliorer le score.

Le champ manually_verified n'est pas utilisé, c'est une piste d'amélioration potentielle.