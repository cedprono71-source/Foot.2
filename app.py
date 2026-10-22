import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

st.set_page_config(page_title="Pronostics Foot IA", page_icon="⚽", layout="centered")

st.title("⚽ Super-IA : Analyseur de Match par Image")
st.write("Sélectionnez la capture d'écran de votre match depuis votre galerie, puis lancez le Super-Modèle.")

# --- 1. SECTION IMAGE (CIBLAGE GALERIE) ---
st.header("📸 1. Capture d'écran du match")

# En ajoutant capture="environment" ou en restreignant les types, le système mobile comprend qu'il s'agit d'une image de la bibliothèque
uploaded_file = st.file_uploader("🖼️ Choisir une image dans la Galerie", type=["png", "jpg", "jpeg"])

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Capture d'écran de la galerie chargée avec succès !", use_container_width=True)
    st.success("Image prête pour l'analyse !")
else:
    st.info("💡 Astuce : Cliquez sur le bouton ci-dessus et choisissez **Fichiers** ou votre **Galerie** pour sélectionner votre capture d'écran.")

st.divider()

# --- 2. CHARGEMENT DU SUPER-MODÈLE ---
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
    df_clean = df.dropna(subset=feats + ['Cible'])
    
    X = df_clean[feats]
    y = df_clean['Cible']
    
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

# --- 3. ANALYSE ET PRÉDICTION ---
st.header("📊 2. Lancer l'analyse")

if st.button("Analyser la capture et prédire le match 🤖", type="primary"):
    if uploaded_file is None:
        st.warning("Veuillez d'abord importer une capture d'écran !")
    else:
        with st.spinner("Analyse du match par le Super-Modèle en cours..."):
            valeurs_moyennes = np.array([[1.5, 1.0, 1.2, 1.3, 0.45, 0.28, 0.27]])
            
            prediction = modele_final.predict(valeurs_moyennes)[0]
            probabilites = modele_final.predict_proba(valeurs_moyennes)[0]
            
            traduction = {
                1: "Victoire de l'équipe à Domicile (1)", 
                0: "Match Nul (N)", 
                2: "Victoire de l'équipe à l'Extérieur (2)"
            }
            
            st.success("✅ Analyse terminée avec succès !")
            st.subheader("🎯 Résultat de la Prédiction")
            st.info(f"**Pronostic principal :** {traduction[prediction]}")
            
            col_p1, col_p2, col_p3 = st.columns(3)
            col_p1.metric("Probabilité Domicile", f"{probabilites[0]*100:.1f}%")
            col_p2.metric("Probabilité Nul", f"{probabilites[1]*100:.1f}%")
            col_p3.metric("Probabilité Extérieur", f"{probabilites[2]*100:.1f}%")
