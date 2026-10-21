import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

st.set_page_config(page_title="Pronostics Foot IA - Automatique", page_icon="⚽", layout="wide")

st.title("⚽ Super-IA : Analyse Automatique des Matchs")
st.write("Le Super-Modèle analyse les derniers matchs de la saison, calcule les statistiques des équipes et prédit les résultats de manière totalement autonome.")

# --- CHARGEMENT ET ENTRAÎNEMENT DU SUPER-MODÈLE ---
@st.cache_resource
def entrainer_et_predire():
    url = "https://www.football-data.co.uk/mmz4281/2425/F1.csv"
    df = pd.read_csv(url).dropna(subset=['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'B365H', 'B365D', 'B365A'])
    
    # Préparation des données d'entraînement
    df['Cible'] = df['FTR'].map({'H': 1, 'D': 0, 'A': 2})
    df['Home_Form_H'] = df.groupby('HomeTeam')['FTHG'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['Home_Concede_H'] = df.groupby('HomeTeam')['FTAG'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['Away_Form_A'] = df.groupby('AwayTeam')['FTAG'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['Away_Concede_A'] = df.groupby('AwayTeam')['FTHG'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['Prob_H'] = 1 / df['B365H']
    df['Prob_D'] = 1 / df['B365D']
    df['Prob_A'] = 1 / df['B365A']
    
    feats = ['Home_Form_H', 'Home_Concede_H', 'Away_Form_A', 'Away_Concede_A', 'Prob_H', 'Prob_D', 'Prob_A']
    df_clean = df.dropna(subset=feats + ['Cible']).copy()
    
    X = df_clean[feats]
    y = df_clean['Cible']
    
    # Entraînement du Stacking Classifier (Random Forest + XGBoost + Meta-LogisticRegression)
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('xgb', XGBClassifier(eval_metric='logloss', random_state=42))
    ]
    modele = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=500)
    )
    modele.fit(X, y)
    
    # Prédiction automatique sur les derniers matchs de la base de données
    derniers_matchs = df_clean.tail(15).copy() # Sélectionne les 15 derniers matchs
    X_pred = derniers_matchs[feats]
    
    preds = modele.predict(X_pred)
    probas = modele.predict_proba(X_pred)
    
    derniers_matchs['Pronostic_IA'] = preds
    derniers_matchs['Pronostic_Texte'] = derniers_matchs['Pronostic_IA'].map({1: "Victoire Domicile (1)", 0: "Match Nul (N)", 2: "Victoire Extérieur (2)"})
    derniers_matchs['Confiance_Max'] = [p.max() * 100 for p in probas]
    
    return derniers_matchs

with st.spinner("🤖 L'IA analyse les matchs et calcule les pronostics en cours..."):
    resultats_df = entrainer_et_predire()

st.success("✅ Analyse automatique terminée avec succès !")

# --- AFFICHAGE DES RÉSULTATS SOUS FORME DE LISTE ---
st.header("📊 Derniers Matchs Analysés par l'IA")

for index, row in resultats_df.iterrows():
    with st.container():
        col1, col2, col3 = st.columns([2, 2, 2])
        col1.markdown(f"**🏠 {row['HomeTeam']}** vs **✈️ {row['AwayTeam']}**")
        col2.markdown(f"🎯 **Pronostic :** `{row['Pronostic_Texte']}`")
        col3.markdown(f"📈 **Confiance :** `{row['Confiance_Max']:.1f}%`")
        st.divider()
