import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import warnings

# On ignore spécifiquement le warning de version incompatible entre Sklearn
warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

# --- CONFIGURATION ---
DOSSIER_MODELES = './modeles_ia/'
MEDICAMENTS = ['Adalimumab', 'Anti-CD20', 'Anti-IL6', 'Certolizumab', 'Etanercept', 'Infliximab']

# Configuration de la page
st.set_page_config(page_title="Mon Traitement Biothérapie", layout="centered")

# --- TITRE ET INTRODUCTION ---
st.title("💊 Quel traitement est fait pour le malade ?")
st.markdown("""
Cette application vous aide à estimer quel médicament pourrait être le plus efficace pour le malade, 
basé sur des données de patients similaires.
*Remplissez les informations ci-dessous à l'aide des derniers résultats d'analyse.*
""")

st.info("👋 **Note :** Les données restent sur votre ordinateur. Rien n'est envoyé sur internet.")

# --- CHARGEMENT DES MODÈLES ---
@st.cache_resource
def charger_modeles():
    modeles = {}
    info_features = {}
    imputers = {}
    
    try:
        for med in MEDICAMENTS:
            modele = joblib.load(os.path.join(DOSSIER_MODELES, f"{med}_model.pkl"))
            features = joblib.load(os.path.join(DOSSIER_MODELES, f"{med}_features.pkl"))
            imputer = joblib.load(os.path.join(DOSSIER_MODELES, f"{med}_imputer.pkl"))
            modeles[med] = modele
            info_features[med] = features
            imputers[med] = imputer
    except Exception as e:
        st.error(f"Erreur de chargement des modèles : {e}")
    return modeles, info_features, imputers

modeles, info_features, imputers = charger_modeles()

# --- FORMULAIRE SIMPLIFIÉ PATIENT ---

st.subheader("1. Les Informations Personnelles")

# On crée des colonnes pour que ce soit moins long
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (ans)", min_value=0, max_value=120, value=50)
    sexe_option = st.selectbox("Sexe :", ["Femme", "Homme"])
    sexe_code = 2 if sexe_option == "Femme" else 1 

    fumeur_option = st.selectbox("Le malade fume ?", ["Non", "Oui"])
    tabac_code = 1 if fumeur_option == "Non" else 2 

with col2:
    poids = st.number_input("Poids (kg)", min_value=30, max_value=200, value=70)
    taille = st.number_input("Taille (cm)", min_value=100, max_value=220, value=170)
    
    # Calcul du BMI automatique
    if taille > 0:
        bmi_calcule = poids / ((taille/100)**2)
    else:
        bmi_calcule = 0

    annees_maladie = st.number_input("Depuis combien d’années a-t-il la maladie ?", min_value=0, max_value=50, value=5)

st.divider()
st.subheader("2. Les Dernières Analyses de Sang")

st.write("*Regardez la dernière prise de sang et entrez les valeurs (normalement en mg/L ou UI/L).*")

col_analyse1, col_analyse2 = st.columns(2)

with col_analyse1:
    crp = st.number_input("CRP (Protéine C Réactive)", min_value=0.0, value=5.0, help="Valeur normale souvent < 10 mg/L")
    vs = st.number_input("VS (Vitesse de Sédimentation)", min_value=0.0, value=15.0, help="Valeur normale souvent < 20 mm/h")
    asat = st.number_input("ASAT (Transaminases)", min_value=0.0, value=20.0, help="Valeur normale < 40 UI/L")
    
with col_analyse2:
    alat = st.number_input("ALAT (Transaminases)", min_value=0.0, value=25.0, help="Valeur normale < 40 UI/L")
    neutrophiles = st.number_input("Neutrophiles (dans la NFS)", min_value=0, value=4000, help="Nombre absolu (ex: 4000)")
    lymphocytes = st.number_input("Lymphocytes (dans la NFS)", min_value=0, value=2000, help="Nombre absolu (ex: 2000)")

st.divider()
st.subheader("3. Activité de la Maladie")

col_maladie1, col_maladie2 = st.columns(2)

with col_maladie1:
    das28 = st.number_input("Score DAS28 (si connu)", min_value=0.0, max_value=10.0, value=4.5, help="Score donné par le rhumatologue. Si inconnu, laissez tel quel.")
    nad = st.number_input("Nombre d'articulations douloureuses", min_value=0, max_value=28, value=5, help="Combien d’articulations sont douloureuses actuellement ?")

with col_maladie2:
    cortico_option = st.selectbox("Prend-il / Prend-elle de la cortisone ?", ["Non", "Oui"])
    cortico_code = 1 if cortico_option == "Non" else 2
    
    acpa_option = st.selectbox("Facteur Rhumatoïde ou ACPA positif ?", ("Je ne sais pas", "Non", "Oui"))
    acpa_code = 1 if acpa_option == "Oui" else (2 if acpa_option == "Non" else np.nan)


# --- CONSTRUCTION DU DICTIONNAIRE TECHNIQUE ---

donnees_patient = {
    'age': age,
    'sexe': sexe_code,
    'BMI': bmi_calcule,
    'tabagisme': tabac_code,
    'dureeevolutionannee': annees_maladie,
    'CRP': crp,
    'VS': vs,
    'ASAT': asat,
    'ALAT': alat,
    'Neutrophiles': neutrophiles,
    'lymphocytes': lymphocytes,
    'das28': das28,
    'NAD': nad,
    'corticoide': cortico_code,
    'acpa_positif': acpa_code 
}

# --- BOUTON ET CALCUL ---
if st.button("🔍 Analyser mon profil", type="primary"):
    
    resultats = []
    
    with st.spinner("L'IA analyse vos données par rapport aux médicaments..."):
        
        for med in MEDICAMENTS:
            modele = modeles[med]
            features_requises = info_features[med]
            imputer = imputers[med]
            
            # 1. Ne prendre que les variables que CE modèle comprend
            vecteur_donnees = []
            for feat in features_requises:
                vecteur_donnees.append(donnees_patient.get(feat, np.nan))
            
            # 2. Mise en forme
            X_input = pd.DataFrame([vecteur_donnees], columns=features_requises)
            
            # SÉCURITÉ : Force float
            X_input = X_input.astype(float)

            # 3. IMPUTATION SANS ERREUR (Contournement de la version Scikit-Learn)
            # L'imputer ancien ne marche pas sur la nouvelle version.
            # On récupère juste les moyennes calculées et on remplace les trous avec Pandas.
            moyennes = pd.Series(imputer.statistics_, index=X_input.columns)
            X_input_imputed = X_input.fillna(moyennes)
            
            # 4. Prédiction
            probas = modele.predict_proba(X_input_imputed)
            proba_succes = probas[0][1] * 100
            
            resultats.append({
                "Médicament": med,
                "Probabilité de Succès (%)": round(proba_succes, 1)
            })
    
    # --- AFFICHAGE ---
    df_resultats = pd.DataFrame(resultats).sort_values(by="Probabilité de Succès (%)", ascending=False)
    
    st.success("Analyse terminée !")
    
    # On met en avant le meilleur choix
    meilleur = df_resultats.iloc[0]
    
    # Utilisation de guillemets simples ''' ''' pour l'extérieur et doubles " " pour l'intérieur
    st.markdown(f'''
    <div style="background-color:#800020; padding:20px; border-radius:10px; text-align:center;">
        <h2>🩺 Résultat estimé</h2>
        <p style="font-size:20px;">Le traitement le plus adapté à votre profil semble être :</p>
        <h1 style="color:#000000;">{meilleur["Médicament"]}</h1>
        <p>Avec une probabilité de réponse positive de <b>{meilleur["Probabilité de Succès (%)"]}%</b></p>
    </div>
    ''', unsafe_allow_html=True)
    
    st.subheader("Détail des probabilités pour chaque médicament")
    st.bar_chart(df_resultats.set_index("Médicament"))
    
    with st.expander("Voir le tableau complet"):
        st.dataframe(df_resultats)

else:

    st.write("Cliquez sur le bouton ci-dessus une fois les informations remplies.")




