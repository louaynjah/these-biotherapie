# 💊 Mon Traitement Biothérapie

Application **Streamlit** d’aide à la décision pour estimer la probabilité de succès de plusieurs biothérapies à partir d’un profil patient.

---

## ✨ Pourquoi ce projet ?

Ce projet permet de :
- centraliser des données cliniques simples (profil patient + bilans biologiques),
- comparer plusieurs médicaments de biothérapie,
- afficher une estimation claire et visuelle des probabilités de réponse.

L’objectif est d’offrir une interface rapide, pédagogique et locale pour explorer des résultats de modèles IA.

---

## 🚀 Fonctionnalités principales

- Formulaire patient intuitif (âge, sexe, IMC, tabagisme, durée de la maladie…)
- Saisie des analyses biologiques essentielles (CRP, VS, ASAT, ALAT, NFS…)
- Évaluation de 6 traitements :
  - Adalimumab
  - Anti-CD20
  - Anti-IL6
  - Certolizumab
  - Etanercept
  - Infliximab
- Classement automatique des probabilités de succès
- Affichage du meilleur candidat thérapeutique
- Visualisation graphique des résultats (bar chart)
- Exécution locale (aucune donnée envoyée sur Internet)

---

## 🧠 Stack technique

- **Python**
- **Streamlit**
- **Pandas / NumPy**
- **Scikit-learn**
- **Joblib** (chargement des modèles et preprocessings)

---

## 📁 Structure du projet

```text
these-biotherapie/
├── app.py
├── requirements.txt
└── modeles_ia/
    ├── *_model.pkl
    ├── *_features.pkl
    └── *_imputer.pkl
```

---

## ⚙️ Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## ▶️ Lancer l’application

```bash
streamlit run app.py
```

Puis ouvrir le lien local affiché dans le terminal (souvent `http://localhost:8501`).

---

## 🖼️ Visuels (à ajouter)

Tu pourras intégrer ici tes images plus tard :

```md
![Capture écran - formulaire](docs/images/formulaire.jpg)
![Capture écran - résultats](docs/images/results.jpg)
```

---

## ⚠️ Avertissement

Cet outil fournit une **aide à l’estimation** basée sur des modèles prédictifs, et ne remplace pas un avis médical spécialisé.

---

## 📌 Statut

Projet opérationnel et prêt à être enrichi (UI, métriques, explications modèle, visualisations avancées).
