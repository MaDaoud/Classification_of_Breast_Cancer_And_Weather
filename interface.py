import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from partie1_classification import model1, model2, scaler, predire_tumeur, X_test, y_test, X, y, analyse_erreurs
from partie2_regression import reg_linear, scaler2, predire_temperature, X_test as X_test_reg, y_test as y_test_reg, X as X_reg, df as df_weather, variables_influentes

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Mini-Projet ML", layout="wide")
st.markdown("<h1 style='text-align: center; color: darkblue;'>🧠 Mini-Projet ML : Classification & Régression</h1>", unsafe_allow_html=True)

tabs = st.tabs(["🔹 Classification", "🔹 Régression"])

# =====================
# PARTIE 1 - CLASSIFICATION
# =====================
with tabs[0]:
    st.subheader("Classification : Breast Cancer")
    
    with st.expander("🔍 Explorer les données"):
        st.write("Dimensions :", X.shape)
        st.dataframe(X.head())
        st.dataframe(X.describe())
        st.info("Le dataset contient des mesures issues de biopsies. La variable cible indique si la tumeur est bénigne (0) ou maligne (1).")

    st.subheader("📊 Visualisations")
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(y.value_counts(), use_container_width=True)
        st.caption("0 = Bénigne, 1 = Maligne")
    with col2:
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(pd.DataFrame(X_test, columns=scaler.feature_names_in_).corr(), cmap="coolwarm", annot=True, ax=ax)
        st.pyplot(fig)

    st.subheader("📈 Évaluation des modèles")
    y_pred1 = model1.predict(scaler.transform(X_test))
    y_pred2 = model2.predict(scaler.transform(X_test))

    metrics = []
    for name, y_pred in zip(["SGD Classifier", "KNN (k=5)"], [y_pred1, y_pred2]):
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1_ = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        metrics.append((name, acc, prec, rec, f1_, cm))

    for m in metrics:
        st.markdown(f"**{m[0]}**")
        col1, col2 = st.columns([1,2])
        with col1:
            st.metric("Accuracy", f"{m[1]:.3f}")
            st.metric("Precision", f"{m[2]:.3f}")
            st.metric("Recall", f"{m[3]:.3f}")
            st.metric("F1-score", f"{m[4]:.3f}")
        with col2:
            st.dataframe(m[5])
            st.write(analyse_erreurs(model1 if m[0]=="SGD Classifier" else model2, scaler.transform(X_test), y_test))

    st.subheader("🖊️ Prédiction d'une tumeur")
    st.info("Entrez les caractéristiques séparées par des virgules pour obtenir une prédiction.")
    user_input_class = st.text_input("Exemple: " + ",".join([str(round(x,2)) for x in X.iloc[0]]))
    if st.button("Prédire la tumeur"):
        if user_input_class.strip() == "":
            st.error("Veuillez entrer des valeurs pour prédire la tumeur.")
        else:
            try:
                valeurs = [float(x.strip()) for x in user_input_class.split(",") if x.strip() != ""]
                if len(valeurs) != X.shape[1]:
                    st.error(f"Il faut exactement {X.shape[1]} valeurs")
                else:
                    resultat = predire_tumeur(model1, scaler, valeurs)
                    st.success(f"Tumeur prédite : {resultat}")
                    st.info("Cette prédiction peut aider à évaluer rapidement le risque de malignité.")
            except Exception as e:
                st.error(f"Erreur : {e}")

# =====================
# PARTIE 2 - RÉGRESSION
# =====================
with tabs[1]:
    st.subheader("Régression : Weather Dataset")
    
    with st.expander("🔍 Explorer les données"):
        st.write("Dimensions :", X_reg.shape)
        st.dataframe(X_reg.head())
        st.dataframe(X_reg.describe())
        st.info("Chaque ligne correspond aux mesures quotidiennes (température, humidité, vent, etc.).")

    st.subheader("📊 Visualisations")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(pd.DataFrame(X_test_reg, columns=scaler2.feature_names_in_).corr(), cmap="coolwarm", annot=True, ax=ax)
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        ax.scatter(X_test_reg[:,0], y_test_reg, c="blue", alpha=0.5)
        ax.set_xlabel(X_reg.columns[0])
        ax.set_ylabel("MaxTemp")
        st.pyplot(fig)

    st.subheader("📈 Évaluation du modèle")
    y_pred = reg_linear.predict(scaler2.transform(X_test_reg))
    r2 = r2_score(y_test_reg, y_pred)
    mae = mean_absolute_error(y_test_reg, y_pred)
    mse = mean_squared_error(y_test_reg, y_pred)
    rmse = np.sqrt(mse)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R²", f"{r2:.3f}")
    col2.metric("MAE", f"{mae:.3f}")
    col3.metric("MSE", f"{mse:.3f}")
    col4.metric("RMSE", f"{rmse:.3f}")
    st.write(variables_influentes(X_reg, y_test_reg))

    st.subheader("🖊️ Prédiction météo")
    st.info("Entrez les valeurs séparées par des virgules pour obtenir la température prédite.")
    st.write(", ".join(X_reg.columns))
    user_input_reg = st.text_input("Exemple: " + ",".join([str(round(x,2)) for x in X_reg.iloc[0]]), key="reg_input")
    if st.button("Prédire la température"):
        if user_input_reg.strip() == "":
            st.error("Veuillez entrer des valeurs pour prédire la température.")
        else:
            try:
                valeurs = [float(x.strip()) for x in user_input_reg.split(",") if x.strip() != ""]
                if len(valeurs) != X_reg.shape[1]:
                    st.error(f"Il faut exactement {X_reg.shape[1]} valeurs")
                else:
                    resultat = predire_temperature(reg_linear, scaler2, valeurs)
                    st.success(f"Température prédite : {resultat:.2f}°C")
                    st.info("Cette prédiction peut alimenter une application météo ou un système automatisé.")
            except Exception as e:
                st.error(f"Erreur : {e}")
