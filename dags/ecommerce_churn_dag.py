from airflow.decorators import dag, task
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# --- CONFIGURATION ---
RAW_DATA_PATH = '/opt/airflow/data/raw/data.csv'
PROCESSED_DATA_PATH = '/opt/airflow/data/processed/rfm_churn.csv'
MODEL_PATH = '/opt/airflow/models/model_churn.joblib'
SCALER_PATH = '/opt/airflow/models/scaler_churn.joblib'

# DATE DE COUPURE (3 mois avant la fin du dataset)
# On utilise les données avant cette date pour prédire ce qui se passe après
CUTOFF_DATE = '2011-09-01'

default_args = {
    'owner': 'DataScientist',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

@dag(
    dag_id='ecommerce_churn_prediction_pro',
    default_args=default_args,
    start_date=datetime(2023, 1, 1),
    schedule_interval='@daily',
    catchup=False,
    tags=['churn', 'advanced_ml']
)
def churn_pipeline():

    @task()
    def ingest_and_clean():
        print(f"📥 Lecture du fichier : {RAW_DATA_PATH}")
        df = pd.read_csv(RAW_DATA_PATH, encoding="ISO-8859-1")
        
        # Nettoyage
        df = df.dropna(subset=['CustomerID'])
        df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
        df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Sauvegarde temporaire
        clean_path = '/opt/airflow/data/processed/clean_transactions.csv'
        df.to_csv(clean_path, index=False)
        return clean_path

    @task()
    def feature_engineering(input_path: str):
        """
        Logique Avancée : Temporal Split
        On apprend sur le passé (Observation) pour prédire le futur (Target)
        """
        print("🔄 Création des features avec coupure temporelle...")
        df = pd.read_csv(input_path)
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        cutoff = pd.to_datetime(CUTOFF_DATE)
        print(f"📅 Date de coupure : {cutoff}")

        # 1. Séparation Observation (Passé) / Target (Futur)
        observation_data = df[df['InvoiceDate'] < cutoff]
        future_data = df[df['InvoiceDate'] >= cutoff]
        
        print(f"   Transactions Passées : {len(observation_data)}")
        print(f"   Transactions Futures : {len(future_data)}")

        # 2. Création des Features sur le PASSE uniquement
        # La 'Recency' est calculée par rapport à la date de coupure (le 'présent' au moment de l'entraînement)
        rfm = observation_data.groupby(['CustomerID']).agg({
            'InvoiceDate': lambda x: (cutoff - x.max()).days, # Jours entre dernier achat et la coupure
            'InvoiceNo': 'count',
            'TotalAmount': 'sum'
        })
        rfm.rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalAmount': 'Monetary'}, inplace=True)

        # 3. Création de la Target (La vérité terrain)
        # Quels clients du passé ont acheté dans le futur ?
        customers_who_returned = future_data['CustomerID'].unique()
        
        # Si le client est dans 'future_data', Is_Churn = 0. Sinon Is_Churn = 1
        rfm['Is_Churn'] = rfm.index.isin(customers_who_returned).astype(int)
        # On inverse la logique : isin = True (Resté) -> Churn = 0
        # Donc isin = False (Pas revenu) -> Churn = 1
        rfm['Is_Churn'] = np.where(rfm['Is_Churn'] == 1, 0, 1)

        print(f"📊 Distribution Churn Réelle :\n{rfm['Is_Churn'].value_counts(normalize=True)}")
        
        rfm.to_csv(PROCESSED_DATA_PATH, index=False)
        return PROCESSED_DATA_PATH

    @task()
    def train_model(input_path: str):
        print("🤖 Entraînement du modèle Prédictif...")
        df = pd.read_csv(input_path)
        
        X = df[['Recency', 'Frequency', 'Monetary']]
        y = df['Is_Churn']
        
        # Train/Test Split classique pour valider le modèle
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # On utilise RandomForest car il gère bien les interactions complexes
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        acc = accuracy_score(y_test, model.predict(scaler.transform(X_test)))
        print(f"✅ Accuracy (Vraie Prédiction) : {acc:.2%}")
        
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)

    path_clean = ingest_and_clean()
    path_rfm = feature_engineering(path_clean)
    train_model(path_rfm)

pipeline = churn_pipeline()