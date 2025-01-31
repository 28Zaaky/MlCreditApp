# Application de prédiction pour l'octroi de prêts bancaires

## Description
Ce projet utilise des techniques d'apprentissage automatique pour analyser les demandes de prêts et prédire si un prêt sera accordé ou non. L'objectif est de mieux comprendre les facteurs influençant l'octroi d'un crédit en exploitant des données historiques.

## Approche Scientifique

### 1. Prétraitement des Données
* **Chargement des données** : Lecture du jeu de données contenant les demandes de prêts.
* **Gestion des valeurs manquantes** : Remplacement des données manquantes par des valeurs pertinentes (imputation par la valeur la plus fréquente pour les catégories et par propagation pour les valeurs numériques).
* **Encodage des variables catégoriques** : Transformation des données textuelles en valeurs numériques pour permettre leur utilisation dans les modèles de machine learning.

### 2. Analyse Exploratoire des Données (EDA)
* **Visualisation des distributions** : Étude de la répartition des prêts accordés et refusés.
* **Corrélations entre variables** : Identification des facteurs les plus influents, notamment l'historique de crédit, les revenus et le statut matrimonial.
* **Analyse des tendances** : Observation des caractéristiques différenciant les emprunteurs acceptés et refusés.

### 3. Modélisation et Prédiction
* **Sélection des modèles** : Test de plusieurs algorithmes d'apprentissage supervisé :
   * Régression Logistique
   * K-Nearest Neighbors (KNN)
   * Arbre de Décision
* **Évaluation des performances** : Comparaison des modèles à l'aide de métriques comme l'accuracy score.
* **Optimisation** : Sélection des variables les plus pertinentes pour améliorer la performance du modèle final.

### 4. Sauvegarde et Exploitation du Modèle
* **Entraînement final** : Ajustement du modèle de Régression Logistique avec les meilleures variables.
* **Sauvegarde** : Le modèle est stocké sous forme d'un fichier `model.pkl` pour une réutilisation ultérieure.

## Utilité du Projet
Ce projet permet de :
* Comprendre les facteurs déterminants dans l'octroi d'un prêt.
* Automatiser la prise de décision à partir d'un modèle prédictif.
* Optimiser les critères d'approbation des crédits pour les institutions financières.
