import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Lire la base de données
df = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')

# Voir les valeurs manquantes
print("\nInformation sur le Data Frame:")
df.info()

print("\nValeurs manquantes par colonne:")
missing_values = df.isnull().sum().sort_values(ascending=False)
print(missing_values)

# Statistiques descriptives numériques et catégoriques
print("\nStatistiques descriptives du DataFrame:")
print(df.describe())

print("\nVariables catégoriques")
print(df.describe(include='O'))

# Séparer les données catégoriques et numériques
cat_data = df.select_dtypes(include='object')
num_data = df.select_dtypes(exclude='object')

# Afficher les premières lignes des dataframes catégoriques et numériques
print("\nDonnées catégoriques:")
print(cat_data.head())

print("\nDonnées numériques:")
print(num_data.head())

# Pour les variables catégoriques, remplacer les valeurs manquantes par les valeurs les plus fréquentes
cat_data = cat_data.apply(lambda x: x.fillna(x.value_counts().index[0]))
missing_cat_data = cat_data.isnull().sum().any()
print("Restent-ils des valeurs manquantes dans les données catégoriques ?", missing_cat_data)

# Pour les variables numériques, remplacer les valeurs manquantes par les valeurs précédentes de la même colonne
num_data.fillna(method='bfill', inplace=True)
missing_num_data = num_data.isnull().sum().any()
print("Restent-ils des valeurs manquantes dans les données numériques ?", missing_num_data)

# Transformer la colonne target
target_value = {'Y': 1, 'N': 0}
target = cat_data['Loan_Status']
cat_data.drop('Loan_Status', axis=1, inplace=True)
target = target.map(target_value)
print(target.head())

# Remplacer les valeurs catégoriques par des valeurs numériques
le = LabelEncoder()
for col in cat_data.columns:
    cat_data[col] = le.fit_transform(cat_data[col])
print(cat_data.head())

# Supprimer Loan_ID
cat_data.drop('Loan_ID', axis=1, inplace=True)

# Concaténer cat_data et num_data et spécifier la colonne target
X = pd.concat([cat_data, num_data], axis=1)
y = target
print(X.head(), "\n", y.head())


# ANALYSE EXPLORATOIRE (EDA)

# Fonction pour visualiser la distribution de la variable cible
def visual_distrib_var(target):
    plt.figure(figsize=(6, 5))
    sns.countplot(x=target, order=target.value_counts().index, palette='Set2')
    # Ajout de labels et de titre
    plt.xlabel('Statut du prêt')
    plt.ylabel('Nombre de prêts')
    plt.title('Distribution des prêts accordés et non accordés')
    # Ajout du texte pour afficher le nombre de prêts au-dessus des barres
    for i, value in enumerate(target.value_counts()):
        plt.text(i, value, str(value), ha='center', va='bottom')
    #plt.show()

    yes = target.value_counts()[1] / len(target)
    no = target.value_counts()[0] / len(target)
    print(f'Le pourcentage des crédits accordés est: {yes:.2%}')
    print(f'Le pourcentage des crédits non accordés est: {no:.2%}')


visual_distrib_var(target)


# Fonction pour visualiser l'impact de chaque variable sur l'accord de crédit
def var_relation(df, X, y):
    df['Loan_Status'] = y  # Ajouter Loan_Status à df pour faciliter la visualisation

    # Credit history
    grid = sns.FacetGrid(df, col='Loan_Status', height=3.2, aspect=1.6)
    grid.map(sns.countplot, 'Credit_History')
    #plt.show()

    # Sexe
    grid = sns.FacetGrid(df, col='Loan_Status', height=3.2, aspect=1.6)
    grid.map(sns.countplot, 'Gender')
    #plt.show()

    # Mariage
    grid = sns.FacetGrid(df, col='Loan_Status', height=3.2, aspect=1.6)
    grid.map(sns.countplot, 'Married')
    #plt.show()

    # Education
    grid = sns.FacetGrid(df, col='Loan_Status', height=3.2, aspect=1.6)
    grid.map(sns.countplot, 'Education')
    #plt.show()

    # Revenus
    plt.figure(figsize=(10, 5))
    plt.scatter(X['ApplicantIncome'], y)
    plt.xlabel('Revenu du demandeur')
    plt.ylabel('Statut du prêt')
    plt.title('Relation entre le revenu du demandeur et le statut du prêt')
    #plt.show()

    # Revenus du conjoint
    plt.figure(figsize=(10, 5))
    plt.scatter(X['CoapplicantIncome'], y)
    plt.xlabel('Revenu du co-demandeur')
    plt.ylabel('Statut du prêt')
    plt.title('Relation entre le revenu du co-demandeur et le statut du prêt')
    #plt.show()

    print(df.groupby('Loan_Status').median(numeric_only=True))


var_relation(df, X, y)


def model_prediction():
    # Diviser la base de données en une base de données de test et de training
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    print('X_train taille : ', X_train.shape)
    print('X_test taille : ', X_test.shape)
    print('y_train taille : ', y_train.shape)
    print('y_test taille : ', y_test.shape)

    # Mise en place de 3 algorithmes de ML (LReg, KNN, DTree)
    models = {
        'LogisticRegression': LogisticRegression(random_state=42),
        'KNeighborsClassifier': KNeighborsClassifier(),
        'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=1, random_state=42)
    }

    return models, X_train, y_train, X_test, y_test


models, X_train, y_train, X_test, y_test = model_prediction()


# Fonction pour calculer la précision
def accu(y_true, y_pred, retu=False):
    acc = accuracy_score(y_true, y_pred)
    if retu:
        return acc
    else:
        print(f'La précision du modèle est: {acc}')


# Fonction pour entraîner et évaluer les modèles
def train_test_eval(models, X_train, y_train, X_test, y_test):
    for name, model in models.items():
        print(name, ':')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accu(y_test, y_pred)
        print('-' * 30)


train_test_eval(models, X_train, y_train, X_test, y_test)


def model_prediction_X_2():
    X_2 = X[['Credit_History', 'Married', 'CoapplicantIncome']]

    # Diviser la base de données en une base de données test et d'entrainement
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(X_2, y):
        X_train_2, X_test_2 = X_2.iloc[train_idx], X_2.iloc[test_idx]
        y_train_2, y_test_2 = y.iloc[train_idx], y.iloc[test_idx]
    print('X_train taille: ', X_train_2.shape)
    print('X_test taille: ', X_test_2.shape)
    print('y_train taille: ', y_train_2.shape)
    print('y_test taille: ', y_test_2.shape)

    return X_train_2, X_test_2, y_train_2, y_test_2


X_train_2, X_test_2, y_train_2, y_test_2 = model_prediction_X_2()


def reg_logistique(X_2):
    Classifier = LogisticRegression()
    Classifier.fit(X_2, y_train_2)

    y_pred = Classifier.predict(X_test_2)
    accuracy = accuracy_score(y_test_2, y_pred)
    print(f"La précision du modèle est: {accuracy}")
    pickle.dump(Classifier, open('model.pkl', 'wb'))


reg_logistique(X_train_2)

