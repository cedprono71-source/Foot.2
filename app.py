import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

st.set_page_config(page_title="Pronostics Foot IA - Match Cible", page_icon="⚽", layout="centered")

st.title("⚽ Super-IA : Analyseur de Match Cible")
st.write("Collez ou importez la capture d'écran de votre match, sélectionnez les équipes, et l'IA calculera le pronostic exact pour ce match.")

# --- 1. SECTION IMAGE (IMPORT / COLLAGE) ---
st.header("📸 1. Image ou Capture du match")

uploaded_file = st.file_uploader("📂 Importer une capture d'écran (ou glisser-déposer)", type=["png", "jpg", "jpeg"])

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Capture d'écran du match en cours d'analyse", use_container_width=True)
    st.success("Image prise en compte !")
else:
    st.info("💡 Astuce : Vous pouvez aussi faire un Ctrl+V ou utiliser le bouton ci-dessus pour charger l'image de votre match.")

st.divider()

# --- 2. CHARGEMENT DU SUPER-MODÈLE ET DES DONNÉES ---
@st.cache_resource
def charger_donnees_et_modele():
    url = "https://www.football-data.co.uk/mmz4281/2425/F1.csv"
    df = pd.read_csv(url).dropna(subset=['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'B365H', 'B365D', 'B365A'])
    
    # Calcul des statistiques de forme et de défense
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
    
    # Entraînement du Stacking Classifier
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('xgb', XGBClassifier(eval_metric='logloss', random_state=42))
    ]
    modele = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=500)
    )
    modele.fit(X, y)
    
    return df_clean, modele

with st.spinner("🤖 Chargement des statistiques et entraînement du Super-Modèle..."):
    df_data, modele_final = charger_donnees_et_modele()

# --- 3. SÉLECTION DU MATCH À ANALYSER ---
st.header("📊 2. Sélection du Match")

equipes_disponibles = sorted(df_data['HomeTeam'].unique())

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("🏠 Équipe à Domicile", equipes_disponibles)
with col2:
    equipes_exterieur = [eq for eq in equipes_disponibles if eq != home_team]
    away_team = st.selectbox("✈️ Équipe à l'Extérieur", equipes_exterieur)

# Récupérer les dernières statistiques enregistrées pour ces deux équipes dans la base
dernier_match_home = df_data[df_data['HomeTeam'] == home_team].iloc[-1]
dernier_match_away = df_data[df_data['AwayTeam'] == away_team].iloc[-1]

# Construction du vecteur de caractéristiques pour ce match précis
features_match = np.array([[
    dernier_match_home['Home_Form_H'],
    dernier_match_home['Home_Concede_H'],
    dernier_match_away['Away_Form_A'],
    dernier_match_away['Away_Concede_A'],
    dernier_match_home['Prob_H'],
    dernier_match_home['Prob_D'],
    dernier_match_home['Prob_A']
]])

# --- 4. BOUTON DE LANCEMENT ---
st.markdown("---")
if st.button("Lancer l'analyse ciblée du match 🎯", type="primary"):
    prediction = modele_final.predict(features_match)[0]
    probabilites = modele_final.predict_proba(features_match)[0]
    
    traduction = {
        1: f"Victoire de **{home_team}** (1)", 
        0: "Match Nul (N)", 
        2: f"Victoire de **{away_team}** (2)"
    }
    
    st.subheader("🎯 Résultat de l'Analyse IA")
    st.success(f"**Pronostic :** {traduction[prediction]}")
    
    col_p1, col_p2, col_p3 = st.columns(3)
    col_p1.metric(f"Victoire {home_team}", f"{probabilites[0]*100:.1f}%")
    col_p2.metric("Match Nul", f"{probabilites[1]*100:.1f}%")
    col_p3.metric(f"Victoire {away_team}", f"{probabilites[2]*100:.1f}%")
