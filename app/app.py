import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

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
        # Fallback si le CSV processe n'est pas là, on essaie de le trouver ailleurs ou on stop
        st.error(f"⚠️ Fichier de données introuvable : {DATA_PATH}.")
        return None
    return pd.read_csv(DATA_PATH)

# --- FONCTION INTELLIGENTE : CHARGEMENT OU ENTRAINEMENT ---
@st.cache_resource
def get_model(df):
    """
    Tente de charger le modèle. S'il échoue (problème de version),
    il ré-entraîne le modèle à la volée sur le Cloud.
    """
    try:
        # 1. On essaie de charger le fichier existant
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
        
    except Exception as e:
        # 2. SI ÇA PLANTE : On active le plan B (Ré-entraînement)
        # st.warning(f"⚠️ Le modèle pré-entraîné n'est pas compatible ({e}). Ré-entraînement automatique en cours...")
        
        # Préparation des données
        X = df[['Recency', 'Frequency', 'Monetary']]
        y = df['Is_Churn']
        
        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Entraînement
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        return model, scaler

# --- 3. INTERFACE PRINCIPALE ---
st.title("📉 Pilotage de la Rétention Client (Churn)")
st.markdown(f"**Source de données :** Pipeline Automatisé Airflow & Docker")

df = load_data()

if df is not None:
    # On passe le DF à la fonction pour qu'elle puisse ré-entraîner si besoin
    model, scaler = get_model(df) 

    if model is not None:
        # Création des onglets
        tab1, tab2 = st.tabs(["📊 Analyse du Portefeuille", "🔮 Prédiction (Horizon 3 mois)"])

        # === ONGLET 1 : ANALYSE ===
        with tab1:
            st.header("1. Vue d'ensemble")
            
            col1, col2, col3, col4 = st.columns(4)
            churn_rate = df['Is_Churn'].mean()
            nb_clients = len(df)
            ca_moyen = df['Monetary'].mean()
            
            col1.metric("Nombre de Clients", f"{nb_clients:,}")
            col2.metric("Taux de Churn (Futur)", f"{churn_rate:.1%}", delta_color="inverse")
            col3.metric("Panier Moyen (LTV)", f"{ca_moyen:.0f} €")
            col4.metric("Horizon de prédiction", "90 Jours")
            
            st.divider()

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Distribution Récence vs Churn")
                fig, ax = plt.subplots()
                sns.histplot(data=df, x='Recency', hue='Is_Churn', bins=30, multiple="stack", palette="Reds", ax=ax)
                st.pyplot(fig)
                
            with c2:
                st.subheader("Impact de la Fréquence d'achat")
                fig2, ax2 = plt.subplots()
                sns.boxplot(x='Is_Churn', y='Frequency', data=df, hue='Is_Churn', legend=False, showfliers=False, palette="Set2", ax=ax2)
                ax2.set_xticks([0, 1])
                ax2.set_xticklabels(["Fidèles", "Futurs Partants"])
                st.pyplot(fig2)

        # === ONGLET 2 : SIMULATEUR ===
        with tab2:
            st.header("Simulateur de Prédiction (Horizon 3 mois)")
            st.markdown("Cet outil utilise l'historique client pour prédire la probabilité qu'il **cesse d'acheter dans les 90 prochains jours**.")
            
            col_input, col_pred = st.columns([1, 2])
            
            with col_input:
                st.info("Entrez les paramètres actuels du client 👇")
                recency = st.number_input("Récence (Jours)", min_value=0, max_value=365, value=30)
                frequency = st.number_input("Fréquence (Achats)", min_value=1, max_value=500, value=5)
                monetary = st.number_input("Montant Total (€)", min_value=0.0, value=500.0)
                predict_btn = st.button("Prédire l'avenir", use_container_width=True, type="primary")

            with col_pred:
                if predict_btn:
                    input_data = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
                    input_scaled = scaler.transform(input_data)
                    
                    proba = model.predict_proba(input_scaled)[0][1]
                    
                    st.divider()
                    if proba > 0.5:
                        st.error(f"### 🔴 RISQUE D'ABANDON PROBABLE")
                        st.write(f"Ce client a **{proba:.1%}** de risque de ne rien acheter dans les 3 prochains mois.")
                    else:
                        st.success(f"### 🟢 CLIENT FIDÈLE")
                        st.write(f"Ce client a **{1-proba:.1%}** de chance de revenir acheter prochainement.")