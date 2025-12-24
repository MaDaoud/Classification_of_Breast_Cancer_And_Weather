import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor

# --- Chargement du dataset ---
df = pd.read_csv("weather.csv")
df = df.drop(columns=["Date"], errors="ignore")
df = df.dropna(axis=1, how="all")
df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna()

y = df["MaxTemp"]
X = df.drop(columns=["MaxTemp"])

# --- Prétraitement ---
scaler2 = StandardScaler()
X_scaled = scaler2.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)

# --- Modélisation ---
reg_linear = LinearRegression()
reg_linear.fit(X_train, y_train)

dummy_reg = DummyRegressor(strategy="mean")
dummy_reg.fit(X_train, y_train)

y_pred_linear = reg_linear.predict(X_test)
y_pred_dummy = dummy_reg.predict(X_test)

# --- Évaluation ---
def eval_regression(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"\n--- {name} ---")
    print(f"R²: {r2:.3f}, MAE: {mae:.3f}, MSE: {mse:.3f}, RMSE: {rmse:.3f}")
    return r2, mae, mse, rmse

eval_regression("Linear Regression", y_test, y_pred_linear)
eval_regression("Dummy Regressor", y_test, y_pred_dummy)

# --- Fonction de prédiction ---
def predire_temperature(modele, scaler, valeurs):
    valeurs = np.array(valeurs).reshape(1, -1)
    valeurs = scaler.transform(valeurs)
    return modele.predict(valeurs)[0]

# --- Analyse des variables influentes ---
def variables_influentes(X, y):
    corr = X.corrwith(y).sort_values(key=abs, ascending=False)
    top3 = corr.head(3)
    return f"Les variables les plus corrélées avec la température maximale sont : {top3.index.tolist()} (corrélation : {top3.values.tolist()})"
