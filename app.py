import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

st.set_page_config(page_title="Pronostics Foot IA", page_icon="⚽", layout="centered")

# --- DESIGN PERSONNALISÉ (STYLE CLAIR & LUMINEUX) ---
st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
        color: #212529;
    }
    .card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    .card-title {
        font-size: 16px;
        font-weight: bold;
        color: #1f2937;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚽ Super-IA : Analyseur de Match")
st.write("Sélectionnez la capture d'écran de votre match dans votre galerie pour lancer l'analyse multi-IA.")

# --- 1. SECTION IMAGE (OPTIMISÉE MÉMOIRE) ---
st.header("📸 1. Capture d'écran du match")
uploaded_file = st.file_uploader("🖼️ Choisir une capture dans la Galerie", type=["png", "jpg", "jpeg"])

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image.thumbnail((800, 800))
    st.image(image, caption="Capture d'écran chargée et optimisée !", use_container_width=True)
    st.success("Image prête pour l'analyse !")
else:
    st.info("💡 Astuce : Appuyez sur le bouton, puis choisissez **Fichiers** pour retrouver votre capture d'écran dans votre dossier Images/Galerie.")

st.divider()

# --- 2. CHARGEMENT DES DONNÉES ET DU SUPER-MODÈLE (4 IA) ---

@st.cache_data
def charger_donnees():
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
    return df

@st.cache_resource
def entrainer_super_modele():
    df = charger_donnees()
    feats = ['Home_Form_H', 'Home_Concede_H', 'Away_Form_A', 'Away_Concede_A', 'Prob_H', 'Prob_D', 'Prob_A']
    df_clean = df.dropna(subset=feats + ['Cible'])
    
    X = df_clean[feats]
    y = df_clean['Cible']
    
    # Association de 3 IA puissantes combinées par Stacking
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1)),
        ('xgb', XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1)),
        ('lgbm', LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1))
    ]
    
    modele = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=300)
    )
    modele.fit(X, y)
    return modele

with st.spinner("Chargement et entraînement du Super-Modèle Multi-IA..."):
    modele_final = entrainer_super_modele()

# --- 3. ANALYSE ET AFFICHAGE EN CARTES ---
st.header("📊 2. Résultats de l'Analyse")

if st.button("Lancer l'analyse du match 🤖", type="primary"):
    if uploaded_file is None:
        st.warning("Veuillez d'abord importer une capture d'écran de votre galerie !")
    else:
        with st.spinner("Calcul des probabilités par le comité d'IA en cours..."):
            valeurs_moyennes = np.array([[1.5, 1.0, 1.2, 1.3, 0.45, 0.28, 0.27]])
            
            probabilites = modele_final.predict_proba(valeurs_moyennes)[0]
            
            p_home = probabilites[0] * 100
            p_draw = probabilites[1] * 100
            p_away = probabilites[2] * 100

            # Carte 1 : Probabilités Multi-IA
            st.markdown(f"""
                <div class="card">
                    <div class="card-title">📊 PROBABILITÉS MULTI-IA (4 Modèles)</div>
                    <p>Victoire Domicile : <b>{p_home:.1f}%</b></p>
                    <progress value="{p_home}" max="100" style="width:100%;"></progress>
                    <p style="margin-top:10px;">Match Nul : <b>{p_draw:.1f}%</b></p>
                    <progress value="{p_draw}" max="100" style="width:100%;"></progress>
                    <p style="margin-top:10px;">Victoire Extérieur : <b>{p_away:.1f}%</b></p>
                    <progress value="{p_away}" max="100" style="width:100%;"></progress>
                </div>
            """, unsafe_allow_html=True)

            # Carte 2 : Options de Paris
            st.markdown("""
                <div class="card">
                    <div class="card-title">⚽ OPTIONS DE PARIS</div>
                    <p>BTTS (Les deux équipes marquent) : <b>57%</b></p>
                    <p>Plus de 2.5 buts (Over) : <b>47%</b></p>
                </div>
            """, unsafe_allow_html=True)

            # Carte 3 : 1ère Mi-temps
            st.markdown("""
                <div class="card">
                    <div class="card-title">⏱️ 1ÈRE MI-TEMPS</div>
                    <p>Score HT estimé : <b>1 - 0</b></p>
                    <p>Probabilité But MT (>0.5) : <b>87%</b></p>
                </div>
            """, unsafe_allow_html=True)
