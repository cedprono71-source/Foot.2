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

# --- CHARGEMENT DU MODÈLE ---
@st.cache_resource
def entrainer_modele_demo():
    url = "https://www.football-data.co.uk/mmz4281/2425/F1.csv"
    df = pd.read_csv(url).dropna(subset=['FTHG', 'FTAG', 'FTR', 'B365H', 'B365D', 'B365A'])
    df['Cible'] = df['FTR'].map({'H': 1, 'D': 0, 'A': 2})
    df['Home_Form_H'] = df.groupby('HomeTeam')['FTHG'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['Home_Concede_H'] = df.groupby('HomeTeam')['FTAG'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['Away_Form_A'] = df.groupby('AwayTeam')['FTAG'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['Away_Concede_A'] = df.groupby('AwayTeam')['FTHG'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['Prob_H'] = 1 / df['B365H']
    df['Prob_D'] = 1 / df['B365D']
    df['Prob_A'] = 1 / df['B365A']
    
    feats = ['Home_Form_H', 'Home_Concede_H', 'Away_Form_A', 'Away_Concede_A', 'Prob_H', 'Prob_D', 'Prob_A']
    df = df.dropna(subset=feats + ['Cible'])
    
    X = df[feats]
    y = df['Cible']
    
    modele = VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('lr', LogisticRegression(max_iter=500, random_state=42)),
            ('xgb', XGBClassifier(eval_metric='logloss', random_state=42))
        ],
        voting='soft'
    )
    modele.fit(X, y)
    return modele

modele_final = entrainer_modele_demo()

# --- BOUTON DE PRÉDICTION ---
if st.button("Lancer l'analyse de l'IA 🤖", type="primary"):
    nouveau_match = np.array([[home_form, home_concede, away_form, away_concede, prob_h, prob_d, prob_a]])
    prediction = modele_final.predict(nouveau_match)[0]
    probabilites = modele_final.predict_proba(nouveau_match)[0]
    
    traduction = {1: "Victoire de l'équipe à Domicile (1)", 0: "Match Nul (N)", 2: "Victoire de l'équipe à l'Extérieur (2)"}
    
    st.divider()
    st.subheader("🎯 Résultat de la Prédiction")
    st.success(f"**Pronostic principal :** {traduction[prediction]}")
    
    col_p1, col_p2, col_p3 = st.columns(3)
    col_p1.metric("Probabilité Domicile", f"{probabilites[0]*100:.1f}%")
    col_p2.metric("Probabilité Nul", f"{probabilites[1]*100:.1f}%")
    col_p3.metric("Probabilité Extérieur", f"{probabilites[2]*100:.1f}%")
