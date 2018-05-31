# FreeSound

Projet ESIR2 pour participer au concours Kaggle Freesound de labellisation de son.
https://www.kaggle.com/c/freesound-audio-tagging

1) Objectifs :

Entraîner un outil de machine learning sur un ensemble de fichiers audio. Labelliser une liste de fichiers audio dont
le label est inconnu. Soumettre le résultat sur le site Kaggle pour obtenir le meilleur score de prédiction
(1 le meilleur, 0 le pire)

Compétences mise en oeuvre :
Utilisation et paramétrage d'outils de machine-learning
Recherche, sélection et extraction des meilleurs features audio
Compréhension des erreurs pour essayer d'améliorer la prédiction
Développement en python

2) Fichiers fournis par la compétition :

41 Labels possibles
9473 fichiers d'entraînement et csv listant les labels
9000 fichiers à labelliser

3) Contraintes

Le programme s'attend à des fichiers csv en entrée de ce format :
fname,label,manually_verified
00044347.wav,Hi-hat,0
001ca53d.wav,Saxophone,1
002d256b.wav,Trumpet,0

Trois propositions de label sont possibles. Le score est ajusté en fonction de la place du bon label dans les trois
labels proposés.


Les fichiers audio doivent être en wav et à 44100 Hz

Le programme doit fournir en sorti un fichier csv qui doit respecter le format attendu par Kaggle. C'est-à-dire le nom
du fichier et les trois labels potentiels.

Extrait du fichier soumis :
ffa502ed.wav,Clarinet Oboe Flute
ffa69cfc.wav,Telephone Electric_piano Snare_drum
ffaca82d.wav,Writing Computer_keyboard Keys_jangling

4) Mise en oeuvre de la solution

L'extraction des features se base sur :
- MFCC
- Zero crossing
- Spectral Contrast
- Spectral bandwidth
- Spectral centroid
- Spectral rolloff

L'extraction des features étant extrêment longue (+ 1H) il est possible de les écrire dans un fichier csv qui peut être
réutilisé à la prochaine exécution.

La prédiction se fait grâce à un classifier RandomForrest.
Il affiche un descriptif du jeu de donnée, renvoi le score de prédiction, un fichier csv soumettable sur Kaggle et un
fichier csv des erreurs dans la prédiction du test set.

5) Dépendances

- python : interpréteur
- pandas : structure facilitant l'analyse de données
- scipy : calcul scientifique
- librosa : extraction de features audio
- sklearn : outil de machine learning
- tqdm : barre de progression
- matplotlib : traçage de graphique

6) Améliorations possibles

Utilisations de features plus pertinentes
Tuning des paramètres de RandomForrest
Utilisation d'un réseau de neurones