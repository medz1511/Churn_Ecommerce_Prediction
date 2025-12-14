import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Dashboard Churn E-Commerce", layout="wide", page_icon="📉")

# Chemins relatifs
DATA_PATH = 'data/processed/rfm_churn.csv'
MODEL_PATH = 'models/model_churn.joblib'
SCALER_PATH = 'models/scaler_churn.joblib'

# --- 2. FONCTIONS DE CHARGEMENT ---
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"⚠️ Fichier introuvable : {DATA_PATH}. Lancez le DAG Airflow d'abord.")
        return None
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("⚠️ Modèle introuvable. Lancez le DAG Airflow d'abord.")
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)

# --- 3. INTERFACE PRINCIPALE ---
st.title("📉 Pilotage de la Rétention Client (Churn)")
st.markdown(f"**Source de données :** Pipeline Automatisé Airflow & Docker")

df = load_data()
model, scaler = load_model()

if df is not None and model is not None:

    # Création des onglets pour organiser la page
    tab1, tab2 = st.tabs(["📊 Analyse du Portefeuille", "🔮 Prédiction (Horizon 3 mois)"])

    # === ONGLET 1 : ANALYSE ===
    with tab1:
        st.header("1. Vue d'ensemble")
        
        # KPI
        col1, col2, col3, col4 = st.columns(4)
        churn_rate = df['Is_Churn'].mean()
        nb_clients = len(df)
        ca_moyen = df['Monetary'].mean()
        
        col1.metric("Nombre de Clients", f"{nb_clients:,}")
        col2.metric("Taux de Churn (Futur)", f"{churn_rate:.1%}", delta_color="inverse")
        col3.metric("Panier Moyen (LTV)", f"{ca_moyen:.0f} €")
        col4.metric("Horizon de prédiction", "90 Jours")
        
        st.divider()

        # Graphiques
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Distribution Récence vs Churn")
            st.markdown("Les clients à droite (Récence élevée) sont ceux qui partent.")
            fig, ax = plt.subplots()
            sns.histplot(data=df, x='Recency', hue='Is_Churn', bins=30, multiple="stack", palette="Reds", ax=ax)
            st.pyplot(fig)
            
        with c2:
            st.subheader("Impact de la Fréquence d'achat")
            st.markdown("Comparaison de la fidélité entre les actifs et les partants.")
            fig2, ax2 = plt.subplots()
            
            # CORRECTION SEABORN (Pour éviter les erreurs rouges)
            sns.boxplot(
                x='Is_Churn', 
                y='Frequency', 
                data=df, 
                hue='Is_Churn',     # Ajout obligatoire
                legend=False,       # On cache la légende inutile
                showfliers=False, 
                palette="Set2", 
                ax=ax2
            )
            
            # Correction des labels
            ax2.set_xticks([0, 1])
            ax2.set_xticklabels(["Fidèles", "Futurs Partants"])
            st.pyplot(fig2)

    # === ONGLET 2 : SIMULATEUR ===
    with tab2:
        st.header("Simulateur de Prédiction (Horizon 3 mois)")
        st.markdown("""
        Cet outil utilise l'historique client pour prédire la probabilité qu'il **cesse d'acheter dans les 90 prochains jours**.
        """)
        
        col_input, col_pred = st.columns([1, 2])
        
        with col_input:
            st.info("Entrez les paramètres actuels du client 👇")
            recency = st.number_input("Récence (Jours depuis dernier achat)", min_value=0, max_value=365, value=30)
            frequency = st.number_input("Fréquence (Nombre de commandes)", min_value=1, max_value=500, value=5)
            monetary = st.number_input("Montant Total (€)", min_value=0.0, value=500.0)
            
            predict_btn = st.button("Prédire l'avenir", use_container_width=True, type="primary")

        with col_pred:
            if predict_btn:
                # Préparation
                input_data = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
                input_scaled = scaler.transform(input_data)
                
                # Prédiction
                prediction = model.predict(input_scaled)[0]
                proba = model.predict_proba(input_scaled)[0][1]
                
                st.divider()
                # Logique inversée pour l'affichage (Plus proba est haute, plus le risque est élevé)
                if proba > 0.5:
                    st.error(f"### 🔴 RISQUE D'ABANDON PROBABLE")
                    st.write(f"Ce client a **{proba:.1%}** de risque de ne rien acheter dans les 3 prochains mois.")
                    st.markdown("**Facteur clé :** Son comportement ressemble à ceux qui ont quitté la plateforme par le passé.")
                else:
                    st.success(f"### 🟢 CLIENT FIDÈLE")
                    st.write(f"Ce client a **{1-proba:.1%}** de chance de revenir acheter prochainement.")