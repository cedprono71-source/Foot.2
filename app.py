import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

st.set_page_config(page_title="Pronostics Foot IA", page_icon="⚽", layout="centered")

st.title("⚽ Mon IA de Pronostics Football")
st.write("Entrez les statistiques récentes et les cotes du match pour obtenir le pronostic de l'IA.")

# --- FORMULAIRE UTILISATEUR ---
st.header("📊 Paramètres du match")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Équipe à Domicile")
    home_form = st.slider("Forme (Buts marqués à dom. / 5 derniers matchs)", 0.0, 4.0, 1.5, 0.1)
    home_concede = st.slider("Défense (Buts encaissés à dom. / 5 derniers matchs)", 0.0, 4.0, 1.0, 0.1)
    cote_h = st.number_input("Cote Bookmaker (Victoire Domicile)", min_value=1.01, value=2.10)

with col2:
    st.subheader("Équipe à l'Extérieur")
    away_form = st.slider("Forme (Buts marqués à ext. / 5 derniers matchs)", 0.0, 4.0, 1.2, 0.1)
    away_concede = st.slider("Défense (Buts encaissés à ext. / 5 derniers matchs)", 0.0, 4.0, 1.3, 0.1)
    cote_a = st.number_input("Cote Bookmaker (Victoire Extérieur)", min_value=1.01, value=3.20)

cote_d = st.number_input("Cote Bookmaker (Match Nul)", min_value=1.01, value=3.40)

# Calcul des probabilités implicites des cotes
prob_h = 1 / cote_h
prob_d = 1 / cote_d
prob_a = 1 / cote_a

# --- CHARGEMENT SIMPLIFIÉ DU MODÈLE POUR LA DÉMO ---
@st.cache_resource
def entrainer_modele_demo():
    url = "https://www.football-data.co.uk/mmz4281/2425/F1.csv"
    df = pd.read_csv(url).dropna(subset=['FTHG', 'FTAG', 'FTR', 'B365H', 'B365D', 'B365A'])
    df['Cible'] = df['FTR'].map({'H': 1, 'D': 0, 'A': 2})
    df['Home_Form_H'] = df.groupby('HomeTeam')['FTHG'].transform(lambda x: x

