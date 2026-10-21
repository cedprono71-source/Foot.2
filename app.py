import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

st.set_page_config(page_title="Pronostics Foot IA", page_icon="⚽", layout="centered")

st.title("⚽ Mon IA de Pronostics Football")
st.write("Importez ou collez une capture d'écran du match et ajustez les paramètres pour lancer le super-modèle.")

# --- SECTION IMAGE (IMPORT OU COLLAGE) ---
st.header("📸 1. Image du match")

uploaded_file = st.file_uploader("📂 Importer depuis votre appareil (ou coller une capture d'écran)", type=["png", "jpg", "jpeg"])

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Capture d'écran chargée avec succès !", use_container_width=True)
    st.success("Image prête pour l'analyse !")
else:
    st.info("💡 Astuce : Vous pouvez aussi faire un **Copier / Coller (Ctrl+V)** directement sur cette page selon votre navigateur, ou utiliser le bouton ci-dessus.")

st.divider()

# --- FORMULAIRE DES PARAMÈTRES ---
st.header("📊 2. Paramètres du match")

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

# --- CHARGEMENT DU SUPER-MODÈLE (STACKING) ---
@st.cache_resource
def entrainer_super_modele():
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
    
    # Fusion avancée par Stacking : Random Forest + XGBoost supervisés par une Régression Logistique
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('xgb', XGBClassifier(eval_metric='logloss', random_state=42))
    ]
    
    modele = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=500)
    )
    
    modele.fit(X, y)
    return modele

modele_final = entrainer_super_modele()

# --- BOUTON DE PRÉDICTION ---
if st.button("Lancer l'analyse du Super-Modèle 🤖", type="primary"):
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
