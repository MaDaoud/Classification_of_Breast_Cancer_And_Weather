import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier

# --- Chargement du dataset ---
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# --- Prétraitement ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)

# --- Modélisation ---
model1 = SGDClassifier(max_iter=1000, tol=1e-3, random_state=0)
model2 = KNeighborsClassifier(n_neighbors=5)

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)

y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)

# --- Évaluation ---
def eval_classification(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")
    print("Matrice de confusion :\n", cm)
    return acc, prec, rec, f1, cm

eval_classification("SGD Classifier", y_test, y_pred1)
eval_classification("KNN (k=5)", y_test, y_pred2)

# --- Fonction de prédiction ---
def predire_tumeur(modele, scaler, valeurs):
    valeurs = np.array(valeurs).reshape(1, -1)
    valeurs = scaler.transform(valeurs)
    prediction = modele.predict(valeurs)[0]
    return "Maligne" if prediction == 1 else "Bénigne"

# --- Analyse des erreurs et insights ---
def analyse_erreurs(model, X_test, y_test):
    y_pred = model.predict(X_test)
    erreurs = np.where(y_pred != y_test)[0]
    if len(erreurs) == 0:
        return "Le modèle ne fait pas d'erreurs sur le test."
    else:
        return f"Le modèle se trompe sur {len(erreurs)} échantillons sur {len(y_test)}. Ces erreurs peuvent être dues à des mesures proches des frontières entre bénin et malin."
